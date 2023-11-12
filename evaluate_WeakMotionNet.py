"""
Evaluate WeakMotionNet in Stage2
Some of the code are modified based on 'train_single_seq.py' in MotionNet.

Reference:
MotionNet (https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from weak_model import WeakMotionNet
from data.weak_nuscenes_dataloader import DatasetSingleSeq_Stage2
from data.weak_waymo_dataloader import DatasetSingleSeq_Stage2 as DatasetSingleSeq_Stage2_waymo

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from evaluation_utils import evaluate_FGBG_prediction, evaluate_motion_prediction


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

out_seq_len = 1  # The number of future frames we are going to predict
height_feat_size = 13  # The size along the height dimension

parser = argparse.ArgumentParser()
parser.add_argument('--evaldata', default='/path_to/nuScenes/input-data/test/', type=str, help='The path to the preprocessed sparse BEV training data')

parser.add_argument('--pretrained', default='pretrained/nuscenes_motion_1-0.pth', type=str)
parser.add_argument('--datatype', default='nuScenes', type=str, choices=['Waymo', 'nuScenes'])
parser.add_argument('--log', default=True, action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')
parser.add_argument('--gpu', default='0')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')


args = parser.parse_args()
print(args)


need_log = args.log
pretrained = args.pretrained
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
datatype = args.datatype
num_workers = args.nworker
evaldata = args.evaldata

def main():

    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        logger_path = check_folder(logger_root)
        logger_path = check_folder(os.path.join(logger_path, 'evaluation'))
        logger_path = check_folder(os.path.join(logger_path, time_stamp))
        log_file_name = os.path.join(logger_path, 'log.txt')
        saver = open(log_file_name, "w")

        saver.write(args.__repr__() + "\n\n")
        saver.flush()


    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
    if datatype == 'nuScenes':
        area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
    elif datatype == 'Waymo':
        area_extents = np.array([[-32., 32.], [-32., 32.], [-1., 4.]])

    if datatype == 'nuScenes':
        evalset = DatasetSingleSeq_Stage2(dataset_root=evaldata, split='test', future_frame_skip=0,
                                          voxel_size=voxel_size, area_extents=area_extents)
    elif datatype == 'Waymo':
        evalset = DatasetSingleSeq_Stage2_waymo(dataset_root=evaldata, split='val', future_frame_skip=0,
                                                voxel_size=voxel_size, area_extents=area_extents)

    evalloader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=num_workers)


    model = WeakMotionNet(out_seq_len=out_seq_len, FGBG_category_num=2, height_feat_size=height_feat_size)
    model = nn.DataParallel(model)
    model = model.to(device)

    if args.pretrained != '':
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    me_static, me_slow, me_fast, acc_bg, acc_fg = eval(model, evalloader, device, saver)

    if need_log:
        saver.close()



def eval(model, evalloader, device, saver):

    # Motion
    if datatype == 'nuScenes':
        num_future_sweeps = 20
        frequency = 20.0
        speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])
    elif datatype == 'Waymo':
        num_future_sweeps = 10
        frequency = 10.0
        speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 40.0]])

    selected_future_sweeps = np.arange(0, num_future_sweeps + 1, num_future_sweeps)  # We evaluate predictions at 1s
    selected_future_sweeps = selected_future_sweeps[1:]
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / frequency)

    cell_groups = list()  # grouping the cells with different speeds
    for i in range(distance_intervals.shape[0]):
        cell_statistics = list()

        for j in range(len(selected_future_sweeps)):
            # corresponds to each row, which records the MSE, median etc.
            cell_statistics.append([])
        cell_groups.append(cell_statistics)


    # Foreground/Background Classification
    overall_cls_pred = list()  # to compute FG/BG classification accuracy for each object category
    overall_cls_gt = list()  # to compute FG/BG classification accuracy for each object category

    # for i, data in enumerate(evalloader, 0):
    for i, data in tqdm(enumerate(evalloader, 0), total=len(evalloader), smoothing=0.9):
        padded_voxel_points, all_disp_field_gt, pixel_cat_map_gt, \
        non_empty_map, all_valid_pixel_maps, future_steps,\
        _, _, _, \
        _, _, _, _, _, _ = data

        padded_voxel_points = padded_voxel_points.to(device)

        with torch.no_grad():
            disp_pred, FGBG_pred = model(padded_voxel_points)
            # The predicted displacements are for the next 0.5s.
            # We linearly interpolate the outputs to the next 1s for evaluation.
            disp_pred = disp_pred * 2.0

            non_empty_map_numpy = non_empty_map.numpy()
            pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()

            overall_cls_gt, overall_cls_pred = evaluate_FGBG_prediction(FGBG_pred, non_empty_map_numpy, pixel_cat_map_gt_numpy, overall_cls_gt, overall_cls_pred)
            cell_groups = evaluate_motion_prediction(disp_pred, FGBG_pred,
                                                     all_disp_field_gt, all_valid_pixel_maps, future_steps,
                                                     distance_intervals, selected_future_sweeps, cell_groups)

    me_list = np.zeros([3])
    for i, d in enumerate(speed_intervals):
        group = cell_groups[i]
        print("--------------------------------------------------------------")
        print("For cells within speed range [{}, {}]:\n".format(d[0], d[1]))
        if need_log:
            saver.write("--------------------------------------------------------------\n")
            saver.write("For cells within speed range [{}, {}]:\n\n".format(d[0], d[1]))

        dump_error = []
        dump_error_quantile_50 = []

        for s in range(len(selected_future_sweeps)):
            row = group[s]

            errors = np.concatenate(row) if len(row) != 0 else row

            if len(errors) == 0:
                mean_error = None
                error_quantile_50 = None
            else:
                mean_error = np.average(errors)
                error_quantile_50 = np.quantile(errors, 0.5)

            dump_error.append(mean_error)
            dump_error_quantile_50.append(error_quantile_50)

            msg = "Frame {}:\nThe mean error is {}\nThe 50% error quantile is {}". \
                format(selected_future_sweeps[s], mean_error, error_quantile_50)
            print(msg)
            if need_log:
                saver.write(msg + "\n")
                saver.flush()

        me_list[i] = mean_error

    print("--------------------------------------------------------------")
    if need_log:
        saver.write("--------------------------------------------------------------\n")

    # Compute the mean FG/BG classification accuracy for each category
    overall_cls_gt = np.concatenate(overall_cls_gt)
    overall_cls_pred = np.concatenate(overall_cls_pred)
    cm = confusion_matrix(overall_cls_gt, overall_cls_pred)
    cm_sum = np.sum(cm, axis=1)

    mean_cat = cm[np.arange(2), np.arange(2)] / cm_sum
    cat_map = {0: 'Background', 1: 'Foreground'}
    for i in range(len(mean_cat)):
        print("mean cat accuracy of {}: {}".format(cat_map[i], mean_cat[i]))
        if need_log:
            saver.write("mean cat accuracy of {}: {}".format(cat_map[i], mean_cat[i])+ "\n")

    # Compute the statistics of mean pixel classification accuracy
    pixel_acc = np.sum(cm[np.arange(2), np.arange(2)]) / np.sum(cm_sum)
    print("Mean pixel classification accuracy: {}".format(pixel_acc))
    if need_log:
        saver.write("Mean pixel classification accuracy: {}".format(pixel_acc)+ "\n")
        saver.flush()

    return me_list[0], me_list[1], me_list[2], cm[0, 0] / cm_sum[0], cm[1, 1] / cm_sum[1]

if __name__ == "__main__":
    main()
