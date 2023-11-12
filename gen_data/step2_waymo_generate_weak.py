"""
Prepare the input data, motion ground truth, and Foreground/Background information for Waymo data.
"""

import numpy as np
import os
import os.path as osp
from pathlib import Path
from functools import reduce
from gen_data.nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import copy
import tqdm
import pickle
import argparse

from gen_data.waymo_data_utils import process_past_pc_waymo, build_BEV_input_waymo, build_BEV_gt_waymo, convert_to_sparse_bev_waymo
from gen_data.gen_weak_waymo_utils import gen_weak_supervision

obj_class_map = {
    "Vehicle": 1, "Pedestrian":2, "Cyclist": 3, "Others": 4
} # take sign as others

voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-1., 4.]])

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name

def create_waymo_infos(root_path, save_root_path, mode):
    flow_time_gap = 1
    past_time_gap = 1

    check_folder(os.path.join(save_root_path, 'input-data'))
    check_folder(os.path.join(save_root_path, 'input-data', mode))
    if mode == "train":
        check_folder(os.path.join(save_root_path, 'weak-data'))
        check_folder(os.path.join(save_root_path, 'weak-data', mode))
        sample_dir = check_folder(os.path.join(save_root_path, 'weak-data', 'train_sample_info'))

    if mode == "train":
        scene_list_file = "ImageSets/train.txt"
        sample_time_gap = 1.0
        type = 'training'
    elif mode == "val":
        scene_list_file = "ImageSets/val.txt"
        sample_time_gap = 1.0
        type = 'validation'
    else:
        assert Exception

    with open(scene_list_file, 'r') as f:
        scene_list = f.readlines()
        scene_list = [s.strip().split(".")[0]  for s in scene_list] 

    print("finish loading scene list")
    sample_data_step = int(sample_time_gap * 10)
    flow_data_step = int(flow_time_gap * 10)
    past_data_step = int(past_time_gap * 10)
    past_data_sample_index = np.arange(0, past_data_step, 2)
    future_data_sample_index = np.arange(1, 1 + flow_data_step, 1)

    for scene_name in tqdm.tqdm(scene_list):
        lidar_path = root_path / type / scene_name
        assert osp.exists(lidar_path)
        ann_path = lidar_path / f"{scene_name}.pkl"

        pc_random_index_dict = dict()
        pc_down_sample_dict = dict()

        with open(ann_path, 'rb') as f:
            ann_data = pickle.load(f)

        num_lidar = len(ann_data)
        for i in range(0, num_lidar, sample_data_step):
            # remove unenough prev and future sweep
            if i < past_data_step  or i > (num_lidar - 1 - flow_data_step):
                continue

            ''' get current info'''
            ann_i = ann_data[i]

            # extract info about reference key
            pose = ann_i["pose"]  # global_from_car, convert pc in car system to global system
            ts = ann_i["time_stamp"]
            token = "{}_{:04d}".format(scene_name, i)

            ''' get past pc '''
            past_pc_list = process_past_pc_waymo(scene_name, lidar_path, ann_data, i, past_data_sample_index, pose, ts)

            ''' build BEV input & gt '''
            padded_voxel_points, voxel_indices_list = build_BEV_input_waymo(past_pc_list, past_data_sample_index, voxel_size, area_extents)
            all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map, pixel_indices, pixel_instance_map = \
                build_BEV_gt_waymo(past_pc_list, voxel_size[0:2], area_extents, ann_i,
                             future_data_sample_index, ann_data, i, pose, ts)

            dense_bev_data = voxel_indices_list, padded_voxel_points, pixel_indices,\
                             pixel_instance_map, all_disp_field_gt, all_valid_pixel_maps,\
                             non_empty_map, pixel_cat_map

            sparse_bev_data = convert_to_sparse_bev_waymo(dense_bev_data)

            # save the data
            save_name = token + '.npy'
            BEV_save_name = os.path.join(save_root_path, 'input-data', mode, save_name)
            np.save(BEV_save_name, arr=sparse_bev_data)

            if mode == "train":
                # build weak supervision
                weak_dict, pc_random_index_dict, pc_down_sample_dict = gen_weak_supervision(scene_name, lidar_path, ann_data, i, pc_random_index_dict, pc_down_sample_dict)
                save_name = token + '.npy'
                weak_save_name = os.path.join(save_root_path, 'weak-data', mode, save_name)
                np.save(weak_save_name, arr=weak_dict)

        if mode == "train":
            save_file_name = os.path.join(sample_dir, scene_name + '_sample_info.npy')
            np.save(save_file_name, arr=pc_random_index_dict)

            save_file_name = os.path.join(sample_dir, scene_name + '_down_sample_info.npy')
            np.save(save_file_name, arr=pc_down_sample_dict)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR',
                        default='/media/ruibo/cc84e27f-82c0-43ca-8f28-5b4c37369175/liruibo/dataset/AA-motion-data/waymo/waymo-unzipped',
                        type=str)
    parser.add_argument('--SAVE_ROOT_DIR',
                        default='/home/ruibo/Documents/A-Task/Pillar/AA-code-for-release/tmp-data/Waymo',
                        type=str)
    args = parser.parse_args()


    ROOT_DIR = args.DATA_DIR
    SAVE_ROOT_DIR = check_folder(args.SAVE_ROOT_DIR)


    root_path = Path(ROOT_DIR)
    save_root_path = Path(SAVE_ROOT_DIR)

    create_waymo_infos(root_path, save_root_path, mode="train")

    create_waymo_infos(root_path, save_root_path, mode="val")
