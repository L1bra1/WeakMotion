"""
Data loader for Waymo data
Some of the code are modified based on 'nuscenes_dataloader.py' in MotionNet.

Reference:
MotionNet (https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""

from torch.utils.data import Dataset
import numpy as np
import os
import warnings
from data.weak_utils import remove_close, filter_pc, convert_semantic_to_FGBG_waymo, gen_voxel_indices_for_pc


class DatasetSingleSeq_Stage1(Dataset):
    """
    Generate the Waymo training dataset for Stage1

    Parameters
    ----------
    dataset_root : Path to dataset root directory
    split : [train/val]
    annotation_ratio: Desired FG/BG annotation ratio.
    num_points: Desired number of points in point clouds.
    """
    def __init__(self, dataset_root=None, split='train', future_frame_skip=0, voxel_size=(0.25, 0.25, 0.4),
                 area_extents=np.array([[-32., 32.], [-32., 32.], [-1., 4.]]), dims=(256, 256, 13), num_category=5,
                 annotation_ratio=0.01, num_points = 40000):

        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(split))

        self.dataset_root = dataset_root
        print("data root:", dataset_root)

        seq_dirs = []
        if split == 'train':
            for d in os.listdir(self.dataset_root):
                # 0: past frame, 1: current frame, 2:future frame
                tmp_0 = os.path.join(self.dataset_root, d) + '/0'
                seq_dirs.append(tmp_0)
                tmp_1 = os.path.join(self.dataset_root, d) + '/1'
                seq_dirs.append(tmp_1)
                tmp_2 = os.path.join(self.dataset_root, d) + '/2'
                seq_dirs.append(tmp_2)
        else:
            for d in os.listdir(self.dataset_root):
                tmp_1 = os.path.join(self.dataset_root, d)
                seq_dirs.append(tmp_1)

        self.seq_files = seq_dirs
        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(split, self.num_sample_seqs))

        # For training, the size of dataset should be 14351 * 3; for validation: 3634;
        if split == 'train' and self.num_sample_seqs != 14351 * 3:
            warnings.warn(">> The size of training dataset is not 17065 * 3.\n")
        elif split == 'val' and self.num_sample_seqs != 3634:
            warnings.warn(">> The size of validation dataset is not 3634.\n")

        self.split = split
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.future_frame_skip = future_frame_skip
        self.dims = dims
        self.annotation_ratio = annotation_ratio
        self.num_points = num_points
        self.num_category = num_category


    def __len__(self):
        return self.num_sample_seqs

    def __getitem__(self, idx):

        seq_file = self.seq_files[idx]

        if self.split == 'train':
            pc_index = seq_file.split('/')[-1] # 0: past frame, 1: current frame, 2:future frame
            scene_name = seq_file.split('/')[-2]
            file_name = os.path.join(self.dataset_root, scene_name)

            weak_data_handle = np.load(file_name, allow_pickle=True)
            weak_dict = weak_data_handle.item()

            pc = weak_dict['synchronized_pc_' + str(pc_index)].T[:, 0:3]
            label = weak_dict['points_label_' + str(pc_index)]
            sample_idx = weak_dict['sample_idx_' + str(pc_index)]

            # Convert semantic label to FB/BG label
            FGBG_gt_mask = convert_semantic_to_FGBG_waymo(label[:, 0])  # 1: Background; 2: Foreground

            # We only annotate partial points according to the given sample_idx;
            # thus, the annotated points for each point cloud is fixed during training
            selected_num = np.floor(self.annotation_ratio * len(sample_idx)).astype(np.int64)
            selected_sample_idx = sample_idx[:selected_num]

            annotation_mask = np.zeros(len(sample_idx), dtype=np.float32)
            annotation_mask[selected_sample_idx] = 1    # 0: point without annotation; 1: point with annotation
            FGBG_gt_mask[annotation_mask == 0] = 3     # 1: Background; 2: Foreground;  3: Unannotated

            # remove close point
            pc, not_close = remove_close(pc, radius=1.0)
            FGBG_gt_mask = FGBG_gt_mask[not_close]

            # Convert point cloud to 3D voxel, which will serve as the input of PreSegNet
            pc, filter_idx = filter_pc(pc, extents=self.area_extents)
            FGBG_gt_mask = FGBG_gt_mask[filter_idx]

            # Convert 3D coordinate to voxel index
            voxel_indices = gen_voxel_indices_for_pc(pc, self.voxel_size, self.area_extents)

            voxel_points = np.zeros(self.dims, dtype=np.bool)
            voxel_points[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1
            voxel_points = voxel_points.astype(np.float32)
            voxel_points = np.expand_dims(voxel_points, 0).astype(np.float32)

            non_empty_map = np.zeros((self.dims[0], self.dims[1]), dtype=np.float32)
            non_empty_map[voxel_indices[:, 0], voxel_indices[:, 1]] = 1.0

            # The number of points is diverse in different scenes;
            # thus, we sample points with a fixed size.
            curr_source_num = pc.shape[0]
            if curr_source_num >= self.num_points:
                pc_sample_idx = np.random.choice(curr_source_num, self.num_points, replace=False)
                curr_source_num = self.num_points
            else:
                pc_sample_idx = np.concatenate((np.arange(curr_source_num),
                                                np.random.choice(curr_source_num, self.num_points - curr_source_num, replace=True)), axis=-1)
            point_FGBG_gt_mask = FGBG_gt_mask[pc_sample_idx]
            source_pc = pc[pc_sample_idx]

            pixel_cat_map = np.zeros(1, dtype=np.float32)


        elif self.split == 'val':
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            gt_dict = gt_data_handle.item()

            dims = gt_dict['3d_dimension']
            num_future_pcs = gt_dict['num_future_pcs']
            num_past_pcs = gt_dict['num_past_pcs']
            pixel_indices = gt_dict['pixel_indices']

            sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
            all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
            all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

            sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
            pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
            pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

            non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
            non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

            # Extract voxel map from the current frame
            indices = gt_dict['voxel_indices_4']
            curr_voxels = np.zeros(dims, dtype=np.bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            voxel_points = np.expand_dims(curr_voxels, 0).astype(np.float32)


            point_FGBG_gt_mask = np.zeros(1, dtype=np.float32)
            source_pc = np.zeros(1, dtype=np.float32)
            curr_source_num = np.zeros(1, dtype=np.float32)

        return voxel_points, non_empty_map, pixel_cat_map, point_FGBG_gt_mask, source_pc, curr_source_num



class DatasetSingleSeq_Stage2(Dataset):
    """
    Generate the Waymo training dataset for Stage2

    Parameters
    ----------
    dataset_root :      Path to input data root directory
    weakdata_root:      Path to weak supervision data root directory
    FBdata_root:        Path to FG/BG masks predicted by PreSegNet in Stage1
    split :             [train/val]
    annotation_ratio:   Desired FG/BG annotation ratio. Should be consistent with the ratio in Stage1
    num_points_seg:     Desired number of points in the current frame. Will be used to train the FG/BG segmentation head
    num_points_motion:  Desired number of FG points in the three frames. Will be used for Chamfer loss
    """
    def __init__(self, dataset_root=None, weakdata_root=None, FBdata_root=None, split='train', future_frame_skip=0, voxel_size=(0.25, 0.25, 0.4),
                 area_extents=np.array([[-32., 32.], [-32., 32.], [-1., 4.]]), dims=(256, 256, 13), num_category=5,
                 annotation_ratio=1.0, num_points_seg = 40000, num_points_motion = 12000):

        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(split))

        self.dataset_root = dataset_root
        print("data root:", dataset_root)
        self.weakdata_root = weakdata_root
        self.FBdata_root = FBdata_root

        seq_dirs = []
        if split == 'train':
            for d in os.listdir(self.dataset_root):
                tmp_0 = os.path.join(self.dataset_root, d)
                seq_dirs.append(tmp_0)
        else:
            for d in os.listdir(self.dataset_root):
                tmp_0 = os.path.join(self.dataset_root, d)
                seq_dirs.append(tmp_0)

        self.seq_files = seq_dirs
        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(split, self.num_sample_seqs))

        # For training, the size of dataset should be 14351; for validation/testing: 3634
        if split == 'train' and self.num_sample_seqs != 14351:
            warnings.warn(">> The size of training dataset is not 14351.\n")
        elif split == 'val' and self.num_sample_seqs != 3634:
            warnings.warn(">> The size of validation dataset is not 3634.\n")


        self.split = split
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.future_frame_skip = future_frame_skip
        self.dims = dims
        self.annotation_ratio = annotation_ratio
        self.num_points_seg = num_points_seg
        self.num_points_motion = num_points_motion
        self.num_category = num_category

    def __len__(self):
        return self.num_sample_seqs

    def sample_foreground_point(self, pc, FGBG_label, use_GT_label=True):
        pc, not_close = remove_close(pc, radius=1.0)
        pc, filter_idx = filter_pc(pc, extents=self.area_extents)

        if use_GT_label:
            FGBG_label = FGBG_label[not_close]
            FGBG_label = FGBG_label[filter_idx]
            FG_mask = FGBG_label == 2
        else:
            FG_mask = FGBG_label

        FG_point = pc[FG_mask]
        FG_point_num = FG_point.shape[0]

        if FG_point_num != 0:
            if FG_point_num >= self.num_points_motion:
                sample_idx = np.random.choice(FG_point_num, self.num_points_motion, replace=False)
                FG_point_num = self.num_points_motion
            else:
                sample_idx = np.concatenate((np.arange(FG_point_num),
                                             np.random.choice(FG_point_num, self.num_points_motion - FG_point_num, replace=True)), axis=-1)
            FG_point = FG_point[sample_idx]
        else:
            FG_point = np.zeros((self.num_points_motion, 3))

        return FG_point, FG_point_num

    def __getitem__(self, idx):
        seq_file = self.seq_files[idx]
        gt_data_handle = np.load(seq_file, allow_pickle=True)
        gt_dict = gt_data_handle.item()

        dims = gt_dict['3d_dimension']
        num_future_pcs = gt_dict['num_future_pcs']
        num_past_pcs = gt_dict['num_past_pcs']
        pixel_indices = gt_dict['pixel_indices']

        sparse_disp_field_gt = gt_dict['disp_field']
        all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
        all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

        sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
        all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
        all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

        sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
        pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
        pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

        non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
        non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

        padded_voxel_points = list()
        for i in range(num_past_pcs):
            indices = gt_dict['voxel_indices_' + str(i)]
            curr_voxels = np.zeros(dims, dtype=np.bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

        # get weak supervision
        if self.split == 'train':
            scene_name = seq_file.split('/')[-1]
            weak_file_name = os.path.join(self.weakdata_root, scene_name)
            weak_data_handle = np.load(weak_file_name, allow_pickle=True)
            weak_dict = weak_data_handle.item()

            # get FG/BG annotations for the current frame,
            # this procedure is the same as the data preparation in Stage1
            # 0: past frame; 1: current frame; 2: future frame

            pc_seg = weak_dict['synchronized_pc_1'].T[:, 0:3]
            label_seg = weak_dict['points_label_1']
            FGBG_gt_mask_seg = convert_semantic_to_FGBG_waymo(label_seg[:, 0])
            sample_idx = weak_dict['sample_idx_1']

            selected_num = np.floor(self.annotation_ratio * len(sample_idx)).astype(np.int64)
            selected_sample_idx = sample_idx[:selected_num]

            annotation_mask = np.zeros(len(sample_idx), dtype=np.float32)
            annotation_mask[selected_sample_idx] = 1    # 0: point without annotation; 1: point with annotation
            FGBG_gt_mask_seg[annotation_mask == 0] = 3     # 1: Background; 2: Foreground;  3: Unlabelled

            pc_seg, not_close = remove_close(pc_seg, radius=1.0)
            FGBG_gt_mask_seg = FGBG_gt_mask_seg[not_close]
            pc_seg, filter_idx = filter_pc(pc_seg, extents=self.area_extents)
            FGBG_gt_mask_seg = FGBG_gt_mask_seg[filter_idx]

            curr_seg_num = pc_seg.shape[0]
            if curr_seg_num >= self.num_points_seg:
                pc_sample_idx = np.random.choice(curr_seg_num, self.num_points_seg, replace=False)
                curr_seg_num = self.num_points_seg
            else:
                pc_sample_idx = np.concatenate((np.arange(curr_seg_num),
                                                np.random.choice(curr_seg_num, self.num_points_seg - curr_seg_num, replace=True)), axis=-1)
            point_FGBG_gt_mask_seg = FGBG_gt_mask_seg[pc_sample_idx]
            pc_seg = pc_seg[pc_sample_idx]

            # get foreground points in three frames for chamfer loss
            if self.annotation_ratio ==1:
                # When using full annotations, we directly extract ground truth foreground points for chamfer loss
                pc_0 = weak_dict['synchronized_pc_0'].T[:, 0:3]
                label_0 = weak_dict['points_label_0']
                FGBG_gt_mask_0 = convert_semantic_to_FGBG_waymo(label_0[:, 0])  # 1: Background; 2: Foreground
                FG_point_0, FG_point_num_0 = self.sample_foreground_point(pc_0, FGBG_gt_mask_0)

                pc_1 = weak_dict['synchronized_pc_1'].T[:, 0:3]
                label_1 = weak_dict['points_label_1']
                FGBG_gt_mask_1 = convert_semantic_to_FGBG_waymo(label_1[:, 0])
                FG_point_1, FG_point_num_1 = self.sample_foreground_point(pc_1, FGBG_gt_mask_1)

                pc_2 = weak_dict['synchronized_pc_2'].T[:, 0:3]
                label_2 = weak_dict['points_label_2']
                FGBG_gt_mask_2 = convert_semantic_to_FGBG_waymo(label_2[:, 0])
                FG_point_2, FG_point_num_2 = self.sample_foreground_point(pc_2, FGBG_gt_mask_2)
            else:
                # When using partial annotations, we extract foreground points predicted by PreSegNet for Chamfer loss
                pred_FGBG_file_name = os.path.join(self.FBdata_root, scene_name.split('.')[0] + '.npz')
                pred_FGBG_data = np.load(pred_FGBG_file_name)

                pc_0 = weak_dict['synchronized_pc_0'].T[:, 0:3]
                pred_FGBG_0 = pred_FGBG_data['pred_0']
                FG_point_0, FG_point_num_0 = self.sample_foreground_point(pc_0, pred_FGBG_0, use_GT_label=False)

                pc_1 = weak_dict['synchronized_pc_1'].T[:, 0:3]
                pred_FGBG_1 = pred_FGBG_data['pred_1']
                FG_point_1, FG_point_num_1 = self.sample_foreground_point(pc_1, pred_FGBG_1, use_GT_label=False)

                pc_2 = weak_dict['synchronized_pc_2'].T[:, 0:3]
                pred_FGBG_2 = pred_FGBG_data['pred_2']
                FG_point_2, FG_point_num_2 = self.sample_foreground_point(pc_2, pred_FGBG_2, use_GT_label=False)

        else:
            pc_seg = np.zeros(1)
            point_FGBG_gt_mask_seg = np.zeros(1)
            curr_seg_num = np.zeros(1)
            FG_point_0 = np.zeros(1)
            FG_point_num_0 = np.zeros(1)
            FG_point_1 = np.zeros(1)
            FG_point_num_1 = np.zeros(1)
            FG_point_2 = np.zeros(1)
            FG_point_num_2 = np.zeros(1)

        return padded_voxel_points, all_disp_field_gt, pixel_cat_map, \
               non_empty_map, all_valid_pixel_maps, num_future_pcs, \
               pc_seg, point_FGBG_gt_mask_seg, curr_seg_num, \
               FG_point_0, FG_point_num_0, FG_point_1, FG_point_num_1, FG_point_2, FG_point_num_2