"""
This code is to generate the input data and the motion ground truth for the training set of Waymo data.
When generating the input BEV data, we follow the setting of MotionNet.
And some of the codes are modified based on 'data_utils.py' in MotionNet.

Reference:
MotionNet (https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""


import numpy as np
from pathlib import Path
from functools import reduce
from gen_data.nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

obj_class_map = {
    "Vehicle": 1, "Pedestrian":2, "Cyclist": 3, "Others": 4
} # take sign as others

def load_waymo_points(lidar_path):
    points = np.load(lidar_path).reshape(-1, 6)
    NLZ_flag = points[:, 5]
    points = points[NLZ_flag == -1]
    return points[:, :3]



def point_in_hull_fast(points, bounding_box):
    """
    Check if a point lies in a bounding box. We first rotate the bounding box to align with axis. Meanwhile, we
    also rotate the whole point cloud. Finally, we just check the membership with the aid of aligned axis.
    This implementation is fast.
    :param points: nd.array (N x d); N: the number of points, d: point dimension
    :param bounding_box: the Box object
    return: The membership of points within the bounding box
    """
    # Make sure it is a unit quaternion
    bounding_box.orientation = bounding_box.orientation.normalised

    # Rotate the point clouds
    pc = bounding_box.orientation.inverse.rotation_matrix @ points.T
    pc = pc.T

    orientation_backup = Quaternion(bounding_box.orientation)  # Deep clone it
    bounding_box.rotate(bounding_box.orientation.inverse)
    corners = bounding_box.corners()

    # Test if the points are in the bounding box
    idx = np.where((corners[0, 7] <= pc[:, 0]) & (pc[:, 0] <= corners[0, 0]) &
                   (corners[1, 1] <= pc[:, 1]) & (pc[:, 1] <= corners[1, 0]) &
                   (corners[2, 2] <= pc[:, 2]) & (pc[:, 2] <= corners[2, 0]))[0]

    # recover
    bounding_box.rotate(orientation_backup)
    return idx


def calc_displace_vector(points: np.array, curr_box: Box, next_box: Box):
    """
    Calculate the displacement vectors for the input points.
    This is achieved by **comparing the current and next bounding boxes**.
    Specifically, we **first rotate the input points according to the delta rotation angle**, and then **translate them**. Finally we compute the
    displacement between the transformed points and the input points.
    :param points: The input points, (N x d). **Note that these points should be inside the current bounding box.**
    :param curr_box: **Current bounding box.**
    :param next_box: **The future next bounding box in the temporal sequence.**
    :return: Displacement vectors for the points.
    """
    assert points.shape[1] == 3, "The input points should have dimension 3."

    # Make sure the quaternions are normalized
    curr_box.orientation = curr_box.orientation.normalised
    next_box.orientation = next_box.orientation.normalised

    delta_rotation = curr_box.orientation.inverse * next_box.orientation
    rotated_pc = (delta_rotation.rotation_matrix @ points.T).T
    rotated_curr_center = np.dot(delta_rotation.rotation_matrix, curr_box.center)
    delta_center = next_box.center - rotated_curr_center

    rotated_tranlated_pc = rotated_pc + delta_center
    pc_displace_vectors = rotated_tranlated_pc - points
    RT = np.zeros([4, 4])
    RT[:3, :3] = delta_rotation.rotation_matrix
    RT[:3, -1] = delta_center
    return pc_displace_vectors, RT



def voxelize_occupy(pts, voxel_size, extents=None, return_indices=False):
    """
    Voxelize the input point cloud. We only record if a given voxel is occupied or not, which is just binary indicator.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be 0 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param return_indices: Whether to return the non-empty voxel indices.
    """
    # Function Constants
    VOXEL_EMPTY = 0
    VOXEL_FILLED = 1

    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Create Voxel Object with -1 as empty/occluded
    leaf_layout = VOXEL_EMPTY * np.ones(num_divisions.astype(int), dtype=np.float32)

    # Fill out the leaf layout
    leaf_layout[voxel_indices[:, 0],
                voxel_indices[:, 1],
                voxel_indices[:, 2]] = VOXEL_FILLED

    if return_indices:
        return leaf_layout, voxel_indices
    else:
        return leaf_layout

def remove_close(points, radius):
    points = points.T
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    points = points.T
    return points, not_close

def filter_pc(pc, extents=None):
    filter_idx = np.where((extents[0, 0] < pc[:, 0]) & (pc[:, 0] < extents[0, 1]) &
                          (extents[1, 0] < pc[:, 1]) & (pc[:, 1] < extents[1, 1]) &
                          (extents[2, 0] < pc[:, 2]) & (pc[:, 2] < extents[2, 1]))[0]
    pc = pc[filter_idx]
    return pc, filter_idx



def process_past_pc_waymo(scene_name, lidar_path, ann_data, i, past_data_sample_index, ref_pose, ts):
    past_pc_list = dict()

    for j in past_data_sample_index:
        sweep_index = i - j
        sweep_ann = ann_data[sweep_index]
        sweep_lidar_pc_path = lidar_path / "{:04d}.npy".format(sweep_index)
        sweep_pose = sweep_ann["pose"]
        past_pc = load_waymo_points(sweep_lidar_pc_path)

        sweep_token = "{}_{:04d}".format(scene_name, sweep_index)
        sweep_ts = sweep_ann["time_stamp"]
        time_lag = ts - sweep_ts

        # ref_from_global * global_from_current = ref_from_current
        tm = reduce(np.dot, [np.linalg.inv(ref_pose), sweep_pose])
        past_pc = past_pc.T
        past_pc[:3, :] = tm.dot(np.vstack((past_pc[:3, :], np.ones(past_pc.shape[1]))))[:3, :]
        past_pc_list['synchronized_pc_' + str(j)] = past_pc
        past_pc_list['frame_id_' + str(j)] = sweep_token
        past_pc_list['ts_' + str(j)] = time_lag

    return past_pc_list



def build_BEV_input_waymo(past_pc_list, past_data_sample_index, voxel_size, area_extents):

    voxel_indices_list = list()
    padded_voxel_points_list = list()

    for j in past_data_sample_index[::-1]:
        past_pc = past_pc_list['synchronized_pc_' + str(j)].T

        # remove close point
        past_pc, not_close = remove_close(past_pc, radius=1.0)
        # fixed size
        past_pc, filter_idx = filter_pc(past_pc, extents=area_extents)

        res, voxel_indices = voxelize_occupy(past_pc, voxel_size=voxel_size, extents=area_extents, return_indices=True)

        voxel_indices_list.append(voxel_indices)
        padded_voxel_points_list.append(res)

    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(np.bool)
    return padded_voxel_points, voxel_indices_list



def build_BEV_gt_waymo(past_pc_list, grid_size, extents, ann_i,
                     future_data_sample_index, ann_data, i, ref_pose, ts,
                     category_num=5, one_hot_thresh=0.8, min_point_num_per_voxel=-1, proportion_thresh = 0.5):

    refer_pc = past_pc_list['synchronized_pc_0'].T
    refer_pc, not_close = remove_close(refer_pc, radius=1.0)
    refer_pc, filter_idx = filter_pc(refer_pc, extents=extents)

    # ----------------------------------------------------
    # Filter and sort the reference point cloud

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < refer_pc[:, 0]) & (refer_pc[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < refer_pc[:, 1]) & (refer_pc[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < refer_pc[:, 2]) & (refer_pc[:, 2] < extents[2, 1]))[0]
        refer_pc = refer_pc[filter_idx]

    # -- Discretize pixel coordinates to given quantization size
    discrete_pts = np.floor(refer_pc[:, 0:2] / grid_size).astype(np.int32)

    # -- Use Lex Sort, sort by x, then y
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))

    refer_pc = refer_pc[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # -- The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # -- Sort unique indices to preserve order
    unique_indices.sort()
    pixel_coords = discrete_pts[unique_indices]

    # -- Number of points per voxel, last voxel calculated separately
    num_points_in_pixel = np.diff(unique_indices)
    num_points_in_pixel = np.append(num_points_in_pixel, discrete_pts.shape[0] - unique_indices[-1])

    # -- Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_pixel_coord = np.floor(extents.T[0, 0:2] / grid_size)
        max_pixel_coord = np.ceil(extents.T[1, 0:2] / grid_size) - 1
    else:
        min_pixel_coord = np.amin(pixel_coords, axis=0)
        max_pixel_coord = np.amax(pixel_coords, axis=0)

    # -- Get the voxel grid dimensions
    num_divisions = ((max_pixel_coord - min_pixel_coord) + 1).astype(np.int32)

    # -- Bring the min voxel to the origin
    pixel_indices = (pixel_coords - min_pixel_coord).astype(int)
    # ----------------------------------------------------

    # ----------------------------------------------------
    # Get the point cloud subsets, which are inside different instance bounding boxes
    refer_box_list = list()
    refer_pc_idx_per_bbox = list()
    points_category = np.zeros(refer_pc.shape[0], dtype=np.int)  # store the point categories

    pixel_instance_id = np.zeros(pixel_indices.shape[0], dtype=np.uint8)
    points_instance_id = np.zeros(refer_pc.shape[0], dtype=np.int)

    # box in current frame,
    cur_nusc_box_dict = {}  # t0 global -> t0
    for obj_idx, obj_id in enumerate(ann_i["annos"]['obj_ids']):
        # vehicle system
        lwh = ann_i["annos"]["dimensions"][obj_idx]  # c_x, c_y, c_z
        ctr = ann_i["annos"]["location"][obj_idx]  # l, w, h
        yaw = ann_i["annos"]["heading_angles"][obj_idx]
        name = ann_i["annos"]["name"][obj_idx]

        nusc_box = Box(
            ctr, [lwh[1], lwh[0], lwh[2]],
            Quaternion(axis=[0, 0, 1], angle=yaw), name=name, token=obj_idx
        )
        cur_nusc_box_dict[obj_id] = nusc_box

        box_name = name

        if box_name in ["Vehicle", "Pedestrian", "Cyclist"]:
            instance_cat = obj_class_map[box_name]
        elif box_name == "Sign":
            instance_cat = obj_class_map["Others"]
        else:
            raise Exception

        idx = point_in_hull_fast(refer_pc[:, 0:3], nusc_box)
        refer_pc_idx_per_bbox.append(idx)
        refer_box_list.append(nusc_box)

        points_category[idx] = instance_cat
        points_instance_id[idx] = obj_idx + 1  # object id starts from 1, background has id 0

    # remove the constraint
    # assert np.max(points_instance_id) <= 255, "The instance id exceeds uint8 max."

    if len(refer_pc_idx_per_bbox) > 0:
        refer_pc_idx_inside_box = np.concatenate(refer_pc_idx_per_bbox).tolist()
    else:
        refer_pc_idx_inside_box = []
    refer_pc_idx_outside_box = set(range(refer_pc.shape[0])) - set(refer_pc_idx_inside_box)
    refer_pc_idx_outside_box = list(refer_pc_idx_outside_box)

    # Compute pixel (cell) categories
    pixel_cat = np.zeros([unique_indices.shape[0], category_num], dtype=np.float32)
    most_freq_info = []

    for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):
        pixel_elements_categories = points_category[v[0]:v[0] + v[1]]
        elements_freq = np.bincount(pixel_elements_categories, minlength=category_num)
        assert np.sum(elements_freq) == v[1], "The frequency count is incorrect."

        elements_freq = elements_freq / float(v[1])
        most_freq_cat, most_freq = np.argmax(elements_freq), np.max(elements_freq)
        most_freq_info.append([most_freq_cat, most_freq])

        most_freq_elements_idx = np.where(pixel_elements_categories == most_freq_cat)[0]
        pixel_elements_instance_ids = points_instance_id[v[0]:v[0] + v[1]]
        most_freq_instance_id = pixel_elements_instance_ids[most_freq_elements_idx[0]]

        if most_freq >= one_hot_thresh:
            one_hot_cat = np.zeros(category_num, dtype=np.float32)
            one_hot_cat[most_freq_cat] = 1.0
            pixel_cat[h] = one_hot_cat

            pixel_instance_id[h] = most_freq_instance_id
        else:
            pixel_cat[h] = elements_freq  # use soft category probability vector.

    pixel_cat_map = np.zeros((num_divisions[0], num_divisions[1], category_num), dtype=np.float32)
    pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_cat[:]

    pixel_instance_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.uint8)
    pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_instance_id[:]

    # Set the non-zero pixels to 1.0, which will be helpful for loss computation
    # Note that the non-zero pixels correspond to both the foreground and background objects
    non_empty_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

    # Ignore the voxel/pillar which contains number of points less than min_point_num_per_voxel; only for fg points
    cell_pts_num = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    cell_pts_num[pixel_indices[:, 0], pixel_indices[:, 1]] = num_points_in_pixel[:]
    tmp_pixel_cat_map = np.argmax(pixel_cat_map, axis=2)
    ignore_mask = np.logical_and(cell_pts_num <= min_point_num_per_voxel, tmp_pixel_cat_map != 0)
    ignore_mask = np.logical_not(ignore_mask)
    ignore_mask = np.expand_dims(ignore_mask, axis=2)

    # Compute the displacement vectors w.r.t. the other sweeps
    all_disp_field_gt_list = list()
    all_valid_pixel_maps_list = list()  # valid pixel map will be used for masking the computation of loss
 
    for j in future_data_sample_index:
        curr_disp_vectors = np.zeros_like(refer_pc, dtype=np.float32)
        curr_disp_vectors.fill(np.nan)
        curr_disp_vectors[refer_pc_idx_outside_box,] = 0.0

        # compute flow between t and t + 0,1
        transformed_box_dict = {}  # t1 global -> t0

        next_ann = ann_data[i + j]
        next_pose = next_ann["pose"]
        next_T_cur_tm = reduce(np.dot, [np.linalg.inv(ref_pose), next_pose])  # ref_from_next
        next_ts_lag = next_ann["time_stamp"] - ts
        for obj_idx, obj_id in enumerate(next_ann["annos"]['obj_ids']):
            # vehicle system
            ctr = next_ann["annos"]["location"][obj_idx]
            lwh = next_ann["annos"]["dimensions"][obj_idx]  # l, w, h
            yaw = next_ann["annos"]["heading_angles"][obj_idx]
            name = next_ann["annos"]["name"][obj_idx]

            # transform to t0
            # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/utils/box_utils.py#L196
            yaw_offset = np.arctan2(next_T_cur_tm[1, 0], next_T_cur_tm[0, 0])  # return radian
            yaw = yaw + yaw_offset

            new_ctr = np.einsum('ij,nj->ni', next_T_cur_tm[:3, :3], ctr[None]) + next_T_cur_tm[:3, -1][None]
            new_ctr = new_ctr[0]

            # nuscenes devkit requires width, length, height as inputs
            nusc_box = Box(
                new_ctr, [lwh[1], lwh[0], lwh[2]], Quaternion(axis=[0, 0, 1], angle=yaw), name=name, token=obj_idx
            )
            transformed_box_dict[obj_id] = nusc_box

        cur_xyz = refer_pc
        # # ----------init-------------------
        cls_mask = np.zeros([len(cur_xyz), 1], dtype=np.int64)


        for box_token, cur_box in cur_nusc_box_dict.items():
            inbox_idx = point_in_hull_fast(cur_xyz, cur_box)

            box_name = cur_box.name

            if box_name in ["Vehicle", "Pedestrian", "Cyclist"]:
                cls_mask[inbox_idx] = obj_class_map[box_name]
            elif box_name == "Sign":
                cls_mask[inbox_idx] = obj_class_map["Others"]
            else:
                raise Exception


            cur_xyz_in_box = cur_xyz[inbox_idx]

            if box_name != "Sign" and box_token in transformed_box_dict:  # compute_flow
                transformed_box = transformed_box_dict[box_token]
                in_box_flow, _ = calc_displace_vector(cur_xyz_in_box, cur_box, transformed_box)
                curr_disp_vectors[inbox_idx] = in_box_flow
            else:  # ignore
                curr_disp_vectors[inbox_idx] = 0.

        # Second, compute the mean displacement vector and category for each non-empty pixel
        disp_field = np.zeros([unique_indices.shape[0], 2], dtype=np.float32)  # we only consider the 2D field

        # We only compute loss for valid pixels where there are corresponding box annotations between two frames
        valid_pixels = np.zeros(unique_indices.shape[0], dtype=np.bool)

        for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):

            # Only when the number of majority points exceeds predefined proportion, we compute
            # the displacement vector for this pixel. Otherwise, We consider it is background (possibly ground plane)
            # and has zero displacement.
            pixel_elements_categories = points_category[v[0]:v[0] + v[1]]
            most_freq_cat, most_freq = most_freq_info[h]

            if most_freq >= proportion_thresh:
                most_freq_cat_idx = np.where(pixel_elements_categories == most_freq_cat)[0]
                most_freq_cat_disp_vectors = curr_disp_vectors[v[0]:v[0] + v[1], :3]
                most_freq_cat_disp_vectors = most_freq_cat_disp_vectors[most_freq_cat_idx]

                if np.isnan(most_freq_cat_disp_vectors).any():  # contains invalid disp vectors
                    valid_pixels[h] = 0.0
                else:
                    mean_disp_vector = np.mean(most_freq_cat_disp_vectors, axis=0)
                    disp_field[h] = mean_disp_vector[0:2]  # ignore the z direction

                    valid_pixels[h] = 1.0

        # Finally, assemble to a 2D image
        disp_field_sparse = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
        disp_field_sparse[pixel_indices[:, 0], pixel_indices[:, 1]] = disp_field[:]
        disp_field_sparse = disp_field_sparse * ignore_mask

        valid_pixel_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
        valid_pixel_map[pixel_indices[:, 0], pixel_indices[:, 1]] = valid_pixels[:]

        all_disp_field_gt_list.append(disp_field_sparse)
        all_valid_pixel_maps_list.append(valid_pixel_map)

    all_disp_field_gt_list = np.stack(all_disp_field_gt_list, axis=0)
    all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, axis=0)


    return all_disp_field_gt_list, all_valid_pixel_maps_list, non_empty_map, pixel_cat_map, pixel_indices, pixel_instance_map



# ---------------------- Convert the dense BEV data into sparse format ----------------------
# This will significantly reduce the space used for data storage
def convert_to_sparse_bev_waymo(dense_bev_data):
    save_voxel_indices_list, save_voxel_points, save_pixel_indices, save_pixel_instance_maps, \
        save_disp_field_gt, save_valid_pixel_maps, save_non_empty_maps, save_pixel_cat_maps = dense_bev_data

    save_num_past_pcs = 5
    save_num_future_pcs = 10

    save_valid_pixel_maps = save_valid_pixel_maps.astype(np.bool)
    save_voxel_dims = save_voxel_points.shape[1:]
    num_categories = save_pixel_cat_maps.shape[-1]

    sparse_disp_field_gt = save_disp_field_gt[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :]
    sparse_valid_pixel_maps = save_valid_pixel_maps[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
    sparse_pixel_cat_maps = save_pixel_cat_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
    sparse_pixel_instance_maps = save_pixel_instance_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1]]

    save_data_dict = dict()
    for i in range(len(save_voxel_indices_list)):
        save_data_dict['voxel_indices_' + str(i)] = save_voxel_indices_list[i].astype(np.int32)

    save_data_dict['disp_field'] = sparse_disp_field_gt
    save_data_dict['valid_pixel_map'] = sparse_valid_pixel_maps
    save_data_dict['pixel_cat_map'] = sparse_pixel_cat_maps
    save_data_dict['num_past_pcs'] = save_num_past_pcs
    save_data_dict['num_future_pcs'] = save_num_future_pcs
    # save_data_dict['trans_matrices'] = save_trans_matrices
    save_data_dict['3d_dimension'] = save_voxel_dims
    save_data_dict['pixel_indices'] = save_pixel_indices
    save_data_dict['pixel_instance_ids'] = sparse_pixel_instance_maps

    # -------------------------------- Sanity Check --------------------------------
    dims = save_non_empty_maps.shape

    test_disp_field_gt = np.zeros((save_num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
    test_disp_field_gt[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :] = sparse_disp_field_gt[:]
    assert np.all(test_disp_field_gt == save_disp_field_gt), "Error: Mismatch"

    test_valid_pixel_maps = np.zeros((save_num_future_pcs, dims[0], dims[1]), dtype=np.bool)
    test_valid_pixel_maps[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]
    assert np.all(test_valid_pixel_maps == save_valid_pixel_maps), "Error: Mismatch"

    test_pixel_cat_maps = np.zeros((dims[0], dims[1], num_categories), dtype=np.float32)
    test_pixel_cat_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]
    assert np.all(test_pixel_cat_maps == save_pixel_cat_maps), "Error: Mismatch"

    test_non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
    test_non_empty_map[save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = 1.0
    assert np.all(test_non_empty_map == save_non_empty_maps), "Error: Mismatch"

    test_pixel_instance_map = np.zeros((dims[0], dims[1]), dtype=np.uint8)
    test_pixel_instance_map[save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = sparse_pixel_instance_maps[:]
    assert np.all(test_pixel_instance_map == save_pixel_instance_maps), "Error: Mismatch"

    for i in range(len(save_voxel_indices_list)):
        indices = save_data_dict['voxel_indices_' + str(i)]
        curr_voxels = np.zeros(save_voxel_dims, dtype=np.bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    return save_data_dict
