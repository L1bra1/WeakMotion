

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from weak_model import PreSegNet
from data.weak_utils import remove_close, filter_pc, convert_semantic_to_FGBG, gen_voxel_indices_for_pc, convert_semantic_to_FGBG_waymo

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

height_feat_size = 13  # The size along the height dimension

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='/path_to/nuScenes/weak-data/train', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('-s', '--save_FB', default='/path_to/nuScenes/FGBG-data/', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--datatype', default='nuScenes', type=str, choices=['Waymo', 'nuScenes'])
parser.add_argument('--pretrained', default='pretrained/nuscenes_seg_0-01.pth', type=str)
parser.add_argument('--gpu', default='0')


args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
datatype = args.datatype


def main():

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
    if datatype == 'nuScenes':
        area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
    elif datatype == 'Waymo':
        area_extents = np.array([[-32., 32.], [-32., 32.], [-1., 4.]])
    dims = (256, 256, 13)

    model = PreSegNet(FGBG_category_num=2, height_feat_size=height_feat_size)
    model = nn.DataParallel(model)
    model = model.to(device)

    if args.pretrained != '':
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])

        check_folder(args.save_FB)
        model_name = args.pretrained.split('/')[-1][:-4]
        args.save_FB = check_folder(os.path.join(args.save_FB, model_name))


    # get file list
    seq_dirs = []
    for d in os.listdir(args.data):
        tmp_0 = os.path.join(args.data, d)
        seq_dirs.append(tmp_0)

    # for file in tqdm(seq_dirs, total=len(seq_dirs), smoothing=0.9):
    for idx, file in tqdm(enumerate(seq_dirs, 0), total=len(seq_dirs), smoothing=0.9):

        if datatype == 'nuScenes':
            scene_name = file.split('/')[-1]
            file = os.path.join(file, '0.npy')
        elif datatype == 'Waymo':
            scene_name = file.split('/')[-1].split('.')[0]

        weak_data_handle = np.load(file, allow_pickle=True)
        weak_dict = weak_data_handle.item()

        pc_0, FGBG_gt_0, voxel_points_0 = gen_voxel_for_PreSegNet(weak_dict, area_extents, voxel_size, dims, index=0)
        pc_1, FGBG_gt_1, voxel_points_1 = gen_voxel_for_PreSegNet(weak_dict, area_extents, voxel_size, dims, index=1)
        pc_2, FGBG_gt_2, voxel_points_2 = gen_voxel_for_PreSegNet(weak_dict, area_extents, voxel_size, dims, index=2)

        pred_0 = estimate_FBGB_for_point(model, voxel_points_0, pc_0, area_extents, voxel_size, device)
        pred_1 = estimate_FBGB_for_point(model, voxel_points_1, pc_1, area_extents, voxel_size, device)
        pred_2 = estimate_FBGB_for_point(model, voxel_points_2, pc_2, area_extents, voxel_size, device)

        FG_pred_0_bool = (pred_0 - 1).astype(np.bool_)
        FG_pred_1_bool = (pred_1 - 1).astype(np.bool_)
        FG_pred_2_bool = (pred_2 - 1).astype(np.bool_)


        file_name = os.path.join(args.save_FB, scene_name + '.npz')
        np.savez(file_name, pred_0=FG_pred_0_bool,
                 pred_1=FG_pred_1_bool,
                 pred_2=FG_pred_2_bool)

    print('Finish!')
    return


def gen_voxel_for_PreSegNet(weak_dict, area_extents, voxel_size, dims, index = 0):
    pc = weak_dict['synchronized_pc_' + str(index)].T[:, 0:3]
    label = weak_dict['points_label_' + str(index)]

    if datatype == 'nuScenes':
        FGBG_gt_mask = convert_semantic_to_FGBG(label[:, 0])  # 1: Background; 2: Foreground
    elif datatype == 'Waymo':
        FGBG_gt_mask = convert_semantic_to_FGBG_waymo(label[:, 0])

    # remove close point
    pc, not_close = remove_close(pc, radius=1.0)
    FGBG_gt_mask = FGBG_gt_mask[not_close]

    # Convert point cloud to 3D voxel, which will serve as the input of PreSegNet
    pc, filter_idx = filter_pc(pc, extents=area_extents)
    FGBG_gt_mask = FGBG_gt_mask[filter_idx]

    # Convert 3D coordinate to voxel index
    voxel_indices = gen_voxel_indices_for_pc(pc, voxel_size, area_extents)

    voxel_points = np.zeros(dims, dtype=np.bool_)
    voxel_points[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1
    voxel_points = voxel_points.astype(np.float32)
    voxel_points = np.expand_dims(voxel_points, 0).astype(np.float32)

    return pc, FGBG_gt_mask, voxel_points



def estimate_FBGB_for_point(model, voxel_points, pc, area_extents, voxel_size, device):
    voxel_points = np.expand_dims(voxel_points, 0).astype(np.float32)
    voxel_points = torch.from_numpy(voxel_points)
    voxel_points = voxel_points.to(device)

    with torch.no_grad():
        FGBG_pred = model(voxel_points)

    voxel_indices = gen_voxel_indices_for_pc(pc, voxel_size, area_extents)
    point_FGBG_pred = FGBG_pred[0, :, voxel_indices[:, 0], voxel_indices[:, 1]].permute(1, 0)
    point_FGBG_pred_np = point_FGBG_pred.cpu().numpy()
    point_FGBG_pred_np = np.argmax(point_FGBG_pred_np, axis=1) + 1 # 1 bg, 2 fg

    return point_FGBG_pred_np


if __name__ == "__main__":
    main()
