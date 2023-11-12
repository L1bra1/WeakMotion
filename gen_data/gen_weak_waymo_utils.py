"""
Prepare the Foreground/Background information for Waymo data.

"""


import numpy as np
from pathlib import Path
from functools import reduce
from gen_data.nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from gen_data.waymo_data_utils import load_waymo_points, point_in_hull_fast

obj_class_map = {
    "Vehicle": 1, "Pedestrian":2, "Cyclist": 3, "Others": 4
} # take sign as others

def gen_weak_supervision(scene_name, lidar_path, ann_data, i, pc_random_index_dict, pc_down_sample_dict, num_down_sample = 50000):
    ''' get current info'''
    ann_i = ann_data[i]

    # extract info about reference key
    lidar_pc_path = lidar_path / "{:04d}.npy".format(i)
    cur_xyz = load_waymo_points(lidar_pc_path)

    ref_pose = ann_i["pose"]
    ref_token = "{}_{:04d}".format(scene_name, i)
    ref_ts = ann_i["time_stamp"]

    save_weak_dict = dict()

    id_list = [-5, 0, 5]
    for j in range(3):
        sweep_index = i + id_list[j]
        sweep_ann = ann_data[sweep_index]
        sweep_lidar_pc_path = lidar_path / "{:04d}.npy".format(sweep_index)
        sweep_pose = sweep_ann["pose"]
        sweep_pc = load_waymo_points(sweep_lidar_pc_path)

        sweep_token = "{}_{:04d}".format(scene_name, sweep_index)
        sweep_ts = sweep_ann["time_stamp"]
        time_lag = sweep_ts - ref_ts

        # ref_from_global * global_from_current = ref_from_current
        tm = reduce(np.dot, [np.linalg.inv(ref_pose), sweep_pose])
        sweep_pc = sweep_pc.T
        sweep_pc[:3, :] = tm.dot(np.vstack((sweep_pc[:3, :], np.ones(sweep_pc.shape[1]))))[:3, :]
        points_label = get_label_info(sweep_ann, lidar_path, sweep_index)

        # down-sample
        down_sample_idx, pc_down_sample_dict = gen_random_index_for_pc(sweep_pc, sweep_token, pc_down_sample_dict)
        sweep_pc_t = sweep_pc.transpose((1, 0))

        # We only preserve a fixed number of points for each point cloud
        if down_sample_idx.shape[0] > num_down_sample:
            sampled_sweep_pc_t = sweep_pc_t[down_sample_idx[:num_down_sample]]
            sampled_points_label = points_label[down_sample_idx[:num_down_sample]].astype(np.int32)
        else:
            sampled_sweep_pc_t = sweep_pc_t[down_sample_idx]
            sampled_points_label = points_label[down_sample_idx].astype(np.int32)
        sampled_sweep_pc = sampled_sweep_pc_t.transpose((1, 0))

        save_weak_dict['synchronized_pc_' + str(j)] = sampled_sweep_pc
        save_weak_dict['frame_id_' + str(j)] = sweep_token
        save_weak_dict['ts_' + str(j)] = time_lag

        save_weak_dict['points_label_' + str(j)] = sampled_points_label

        sample_idx, pc_random_index_dict = gen_random_index_for_pc(sampled_sweep_pc, sweep_token, pc_random_index_dict)
        save_weak_dict['sample_idx_' + str(j)] = sample_idx.astype(np.int32)

    return save_weak_dict, pc_random_index_dict, pc_down_sample_dict

def get_label_info(sweep_ann, lidar_path, sweep_index):

    sweep_nusc_box_dict = {}
    for obj_idx, obj_id in enumerate(sweep_ann["annos"]['obj_ids']):
        # vehicle system
        lwh = sweep_ann["annos"]["dimensions"][obj_idx]  # c_x, c_y, c_z
        ctr = sweep_ann["annos"]["location"][obj_idx]  # l, w, h
        yaw = sweep_ann["annos"]["heading_angles"][obj_idx]
        name = sweep_ann["annos"]["name"][obj_idx]

        nusc_box = Box(
            ctr, [lwh[1], lwh[0], lwh[2]],
            Quaternion(axis=[0, 0, 1], angle=yaw), name=name, token=obj_idx
        )
        sweep_nusc_box_dict[obj_id] = nusc_box


    # # ----------init-------------------
    lidar_pc_path = lidar_path / "{:04d}.npy".format(sweep_index)
    sweep_xyz = load_waymo_points(lidar_pc_path)

    sweep_cls_mask = np.zeros([len(sweep_xyz), 1], dtype=np.int64)

    inbox_idx_dict = {}
    for box_token, sweep_box in sweep_nusc_box_dict.items():
        inbox_idx = point_in_hull_fast(sweep_xyz, sweep_box)

        box_name = sweep_box.name

        if box_name in ["Vehicle", "Pedestrian", "Cyclist"]:
            sweep_cls_mask[inbox_idx] = obj_class_map[box_name]
        elif box_name == "Sign":
            sweep_cls_mask[inbox_idx] = obj_class_map["Others"]
        else:
            raise Exception
        inbox_idx_dict[box_token] = inbox_idx
    return sweep_cls_mask


def gen_random_index_for_pc(pc, token, random_index_dict):
    curr_source_num = pc.shape[1]
    if token in random_index_dict.keys():
        sample_idx = random_index_dict[token]
        assert curr_source_num == len(sample_idx)
    else:
        sample_idx = np.random.choice(curr_source_num, curr_source_num, replace=False)
        random_index_dict[token] = sample_idx.astype(np.int32)
    return sample_idx, random_index_dict