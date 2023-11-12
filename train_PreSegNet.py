"""
Train PreSegNet in Stage1
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
from weak_model import PreSegNet
from data.weak_nuscenes_dataloader import DatasetSingleSeq_Stage1
from data.weak_waymo_dataloader import DatasetSingleSeq_Stage1 as DatasetSingleSeq_Stage1_waymo

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from loss_utils import FGBG_seg_loss
from evaluation_utils import evaluate_FGBG_prediction

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


height_feat_size = 13  # The size along the height dimension

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='/path_to/nuScenes/weak-data/train/', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('-t', '--evaldata', default='/path_to/nuScenes/input-data/val/', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--datatype', default='nuScenes', type=str, choices=['Waymo', 'nuScenes'])

parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--batch', default=16, type=int, help='Batch size')
parser.add_argument('--nepoch', default=30, type=int, help='Number of epochs')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')
parser.add_argument('--log', default=True, action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')
parser.add_argument('--gpu', default='0')
parser.add_argument('--annotation_ratio', default=0.01, type=float)


args = parser.parse_args()
print(args)

num_epochs = args.nepoch
need_log = args.log
BATCH_SIZE = args.batch
num_workers = args.nworker
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
datatype = args.datatype
annotation_ratio = args.annotation_ratio

def main():
    start_epoch = 1
    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'Stage1'))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()


        else:
            model_save_path = args.resume

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
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
        trainset = DatasetSingleSeq_Stage1(dataset_root=args.data, split='train',
                                           future_frame_skip=0,voxel_size=voxel_size,
                                           area_extents=area_extents, annotation_ratio=annotation_ratio)
    elif datatype == 'Waymo':
        trainset = DatasetSingleSeq_Stage1_waymo(dataset_root=args.data, split='train',
                                                 future_frame_skip=0,voxel_size=voxel_size,
                                                 area_extents=area_extents, annotation_ratio=annotation_ratio)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    print("Training dataset size:", len(trainset))

    if datatype == 'nuScenes':
        evalset = DatasetSingleSeq_Stage1(dataset_root=args.evaldata, split='val', future_frame_skip=0,
                                          voxel_size=voxel_size, area_extents=area_extents)
    elif datatype == 'Waymo':
        evalset = DatasetSingleSeq_Stage1_waymo(dataset_root=args.evaldata, split='val', future_frame_skip=0,
                                                voxel_size=voxel_size, area_extents=area_extents)

    evalloader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=num_workers)
    print("Validation dataset size:", len(evalset))


    model = PreSegNet(FGBG_category_num=2, height_feat_size=height_feat_size)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        model.train()
        loss_FGBG_seg = train(model, trainloader, optimizer, device, epoch, voxel_size, area_extents)

        model.eval()
        acc_bg, acc_fg, pixel_acc = eval(model, evalloader, device, datatype)

        scheduler.step()


        if need_log:
            saver.write("loss_FGBG_seg: {}\n".format(loss_FGBG_seg))
            saver.write("acc_bg: {}\t acc_fg: {}\t pixel_acc: {}\n".format(acc_bg, acc_fg, pixel_acc))
            saver.flush()

        # save model
        if need_log and (epoch >= 20):
            save_dict = {'epoch': epoch,
                         'model_state_dict': model.state_dict(),
                         'loss': loss_FGBG_seg.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '_%.3f_%.3f_%.3f'%(acc_bg, acc_fg, pixel_acc) + '.pth'))

    if need_log:
        saver.close()


def train(model, trainloader, optimizer, device, epoch, voxel_size, area_extents):
    running_loss_FGBG_seg = AverageMeter('FGBG_Seg', ':.6f')  # for cell FG/BG segmentation error

    # for i, data in enumerate(trainloader, 0):
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader), smoothing=0.9):
        voxel_points, non_empty_map, _, point_FGBG_gt_mask, source_pc, curr_source_num = data
        optimizer.zero_grad()

        # Move to GPU/CPU
        voxel_points = voxel_points.to(device)

        # Make prediction
        FGBG_pred = model(voxel_points)

        # Compute and back-propagate the losses
        loss_FGBG_seg = FGBG_seg_loss(FGBG_pred, point_FGBG_gt_mask, source_pc, curr_source_num, voxel_size, area_extents)
        total_loss = loss_FGBG_seg
        total_loss.backward()
        optimizer.step()

        running_loss_FGBG_seg.update(loss_FGBG_seg.item())


    print("{}, \tat epoch {}, \titerations {}".
          format(running_loss_FGBG_seg, epoch, i))

    return running_loss_FGBG_seg


def eval(model, evalloader, device, datatype):


    overall_cls_pred = list()  # to compute FG/BG classification accuracy for each object category
    overall_cls_gt = list()  # to compute FG/BG classification accuracy for each object category


    # for i, data in enumerate(evalloader, 0):
    for i, data in tqdm(enumerate(evalloader, 0), total=len(evalloader), smoothing=0.9):
        voxel_points, non_empty_map, pixel_cat_map_gt, _, _, _ = data

        voxel_points = voxel_points.to(device)

        with torch.no_grad():
            FGBG_pred = model(voxel_points)

            non_empty_map_numpy = non_empty_map.numpy()
            pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()

            overall_cls_gt, overall_cls_pred = evaluate_FGBG_prediction(FGBG_pred, non_empty_map_numpy, pixel_cat_map_gt_numpy,
                                                                        overall_cls_gt, overall_cls_pred, datatype=datatype)


    # Compute the mean FG/BG classification accuracy for each category
    overall_cls_gt = np.concatenate(overall_cls_gt)
    overall_cls_pred = np.concatenate(overall_cls_pred)
    cm = confusion_matrix(overall_cls_gt, overall_cls_pred)
    cm_sum = np.sum(cm, axis=1)

    mean_cat = cm[np.arange(2), np.arange(2)] / cm_sum
    cat_map = {0: 'Background', 1: 'Foreground'}
    for i in range(len(mean_cat)):
        print("mean cat accuracy of {}: {}".format(cat_map[i], mean_cat[i]))

    # Compute the statistics of mean pixel classification accuracy
    pixel_acc = np.sum(cm[np.arange(2), np.arange(2)]) / np.sum(cm_sum)
    print("Mean pixel classification accuracy: {}".format(pixel_acc))

    return cm[0, 0] / cm_sum[0], cm[1, 1] / cm_sum[1], pixel_acc

if __name__ == "__main__":
    main()
