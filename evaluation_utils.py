"""
Codes to evaluate Foreground/Background segmentation and motion prediction
Some of the code are modified based on 'eval.py' in MotionNet.

Reference:
MotionNet (https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

def evaluate_FGBG_prediction(FGBG_pred, non_empty_map_numpy, pixel_cat_map_gt_numpy, overall_cls_gt, overall_cls_pred,
                             datatype='nuScenes'):

    # Convert the category map
    max_prob = np.amax(pixel_cat_map_gt_numpy, axis=-1)
    filter_mask = max_prob == 1.0  # Note: some of the cell probabilities are soft probabilities
    pixel_cat_map_numpy = np.argmax(pixel_cat_map_gt_numpy,
                                    axis=-1) + 1  # category starts from 1 (background), etc

    # Convert category label to FG/BG label
    pixel_FGBG_map_numpy = pixel_cat_map_numpy.copy()
    if datatype == 'nuScenes':
        # 1: background or empty; 2: Vehicle; 3: Ped; 4: Bike; 5: Others
        pixel_FGBG_map_numpy[pixel_FGBG_map_numpy > 1] = 2
    elif datatype == 'Waymo':
        # 1: background or empty; 2: Vehicle; 3: Ped; 4: Cyclist; 5: Sign, regarded as background
        tmp = pixel_FGBG_map_numpy.copy()
        pixel_FGBG_map_numpy[tmp > 1] = 2
        pixel_FGBG_map_numpy[(tmp == 5)] = 1

    pixel_FGBG_map_numpy = (pixel_FGBG_map_numpy * non_empty_map_numpy * filter_mask).astype(
        np.int32)  # 0: Empty; 1: Background; 2: Foreground

    FGBG_pred_numpy = FGBG_pred.cpu().numpy()
    FGBG_pred_numpy = np.transpose(FGBG_pred_numpy, (0, 2, 3, 1))
    FGBG_pred_numpy = np.argmax(FGBG_pred_numpy, axis=-1) + 1
    FGBG_pred_numpy = (FGBG_pred_numpy * non_empty_map_numpy * filter_mask).astype(np.int32)

    border = 8
    roi_mask = np.zeros_like(non_empty_map_numpy)
    roi_mask[:, border:-border, border:-border] = 1.0

    # For computing confusion matrix, in order to compute FG/BG classification accuracy for each category
    count_mask = non_empty_map_numpy * filter_mask * roi_mask
    idx_fg = np.where(count_mask > 0)

    overall_cls_gt.append(pixel_FGBG_map_numpy[idx_fg])
    overall_cls_pred.append(FGBG_pred_numpy[idx_fg])

    return overall_cls_gt, overall_cls_pred


def evaluate_motion_prediction(disp_pred, FGBG_pred, all_disp_field_gt, all_valid_pixel_maps, future_steps,
                               distance_intervals, selected_future_sweeps, cell_groups,
                               use_FGBG_pred_masking=True, datatype='nuScenes'):

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(all_disp_field_gt.size(0), -1, pred_shape[-3], pred_shape[-2], pred_shape[-1])
    disp_pred = disp_pred.contiguous()
    disp_pred = disp_pred.cpu().numpy()

    if use_FGBG_pred_masking:
        FGBG_pred_numpy = FGBG_pred.cpu().numpy()
        FGBG_pred_numpy = np.argmax(FGBG_pred_numpy, axis=1)
        mask = FGBG_pred_numpy == 0  # predicted Background mask

        # For those with very small movements, we consider them as static
        last_pred = disp_pred[:, -1, :, :, :]
        last_pred_norm = np.linalg.norm(last_pred, ord=2, axis=1)  # out: (batch, h, w)
        thd_mask = last_pred_norm <= 0.2

        cat_weight_map = np.ones_like(FGBG_pred_numpy, dtype=np.float32)
        cat_weight_map[mask] = 0.0
        cat_weight_map[thd_mask] = 0.0
        cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)

        disp_pred = disp_pred * cat_weight_map  # small motion, static, background


    # Pre-processing
    all_disp_field_gt = all_disp_field_gt.numpy()  # (bs, seq, h, w, channel)
    future_steps = future_steps.numpy()[0]

    valid_pixel_maps = all_valid_pixel_maps[:, -future_steps:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.numpy()

    all_disp_field_gt = all_disp_field_gt[:, -future_steps:, ]
    all_disp_field_gt = np.transpose(all_disp_field_gt, (0, 1, 4, 2, 3))
    all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=2)

    upper_thresh = 0.2
    if datatype == 'nuScenes':
        upper_bound = 1 / 20 * upper_thresh
    elif datatype == 'Waymo':
        upper_bound = 1 / 10 * upper_thresh

    static_cell_mask = all_disp_field_gt_norm <= upper_bound
    static_cell_mask = np.all(static_cell_mask, axis=1)  # along the temporal axis
    moving_cell_mask = np.logical_not(static_cell_mask)

    for j, d in enumerate(distance_intervals):
        for slot, s in enumerate((selected_future_sweeps - 1)):  # selected_future_sweeps: [4, 8, ...]
            curr_valid_pixel_map = valid_pixel_maps[:, s]

            if j == 0:  # corresponds to static cells
                curr_mask = np.logical_and(curr_valid_pixel_map, static_cell_mask)
            else:
                # We use the displacement between keyframe and the last sample frame as metrics
                last_gt_norm = all_disp_field_gt_norm[:, -1]
                mask = np.logical_and(d[0] <= last_gt_norm, last_gt_norm < d[1])

                curr_mask = np.logical_and(curr_valid_pixel_map, mask)
                curr_mask = np.logical_and(curr_mask, moving_cell_mask)

            # we evaluate the performance for cells within the range [-30m, 30m] along both x, y dimensions.
            border = 8
            roi_mask = np.zeros_like(curr_mask, dtype=np.bool_)
            roi_mask[:, border:-border, border:-border] = True
            curr_mask = np.logical_and(curr_mask, roi_mask)

            cell_idx = np.where(curr_mask == True)

            gt = all_disp_field_gt[:, s]
            pred = disp_pred[:, -1, :, :, :]
            norm_error = np.linalg.norm(gt - pred, ord=2, axis=1)

            cell_groups[j][slot].append(norm_error[cell_idx])

    return cell_groups
