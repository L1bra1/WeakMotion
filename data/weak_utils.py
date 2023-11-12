import numpy as np
import os

def remove_close(points, radius):
    points = points.T
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    points = points.T
    return points, not_close

def filter_pc(pc, extents):
    filter_idx = np.where((extents[0, 0] < pc[:, 0]) & (pc[:, 0] < extents[0, 1]) &
                          (extents[1, 0] < pc[:, 1]) & (pc[:, 1] < extents[1, 1]) &
                          (extents[2, 0] < pc[:, 2]) & (pc[:, 2] < extents[2, 1]))[0]
    pc = pc[filter_idx]
    return pc, filter_idx

def gen_voxel_indices_for_pc(pc, voxel_size, extents):
    # Convert 3D coordinate to voxel index
    discrete_pc = np.floor(pc[:, :3] / voxel_size).astype(np.int32)
    min_voxel_coord = np.floor(extents.T[0] / voxel_size)
    voxel_indices = (discrete_pc - min_voxel_coord).astype(int)
    return voxel_indices

def convert_semantic_to_FGBG(cate):
    # Label ID 0: nose; Label ID 1~23: foreground classes; Label ID 24~31: background classes
    # reference https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md
    # and https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_lidarseg.md

    fg_mask = (0 < cate) & (cate < 24)
    return fg_mask.astype(np.int32) + 1


def convert_semantic_to_FGBG_waymo(cate):
    # Label ID 0: Background; 1: Vehicle; 2: Pedestrian; 3: Cyclist; 4: Sign, regarded as background

    fg_mask = (0 < cate) & (cate < 4)
    return fg_mask.astype(np.int32) + 1