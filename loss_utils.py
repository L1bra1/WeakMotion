"""
Codes for the FG/BG classification loss and the consistency-aware Chamfer distance loss
"""

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

from data.weak_utils import gen_voxel_indices_for_pc

def FGBG_seg_loss(FGBG_pred, point_FGBG_gt_mask, source_pc, source_num, voxel_size, area_extents):
    """
    Foreground Background segmentation loss
    ----------

    Inputs:
        FGBG_pred: [B, 2, dim_0, dim_1],    predicted Foreground/Background BEV map
        point_FGBG_gt_mask: [B, N],         per-point Foreground/Background ground truth, (1: BG, 2: FG, 3: Unannotated)
        source_pc: [B, N, 3],               point cloud in current frame
        source_num: [B],                    unrepeated point number in each sample
        voxel_size, area_extents:           voxel size and range of area,
    """

    batch_size = FGBG_pred.shape[0]
    device = FGBG_pred.device

    loss_FGBG_seg = torch.zeros((1), device=device, dtype=FGBG_pred.dtype)

    for batch_index in range(batch_size):

        # get current batch
        curr_source_num = source_num[batch_index]
        curr_source_pc_np = source_pc[batch_index, :curr_source_num, :].numpy()
        curr_point_FGBG_gt_mask = point_FGBG_gt_mask[batch_index, :curr_source_num].float().to(device)  # 1: Background; 2: Foreground;  3: Unannotated
        curr_FGBG_pred = FGBG_pred[batch_index]

        # generate FGBG ground truth and weight for each point
        curr_point_BG_gt_mask = (curr_point_FGBG_gt_mask == 1).float().unsqueeze(0)
        curr_point_FG_gt_mask = (curr_point_FGBG_gt_mask == 2).float().unsqueeze(0)

        curr_point_FGBG_gt_map = torch.cat([curr_point_BG_gt_mask, curr_point_FG_gt_mask], 0).permute(1, 0)

        # weight assigned to different categories. 0.005 for BG; 1.0 for FG; 0.0 for unlabelled
        curr_FGBG_weight_map = (curr_point_BG_gt_mask * 0.005 + curr_point_FG_gt_mask * 1.0).squeeze(0)
        curr_annotated_point_num = torch.sum((curr_point_FGBG_gt_mask != 3).float())

        # get FGBG prediction for each point
        curr_voxel_indices = gen_voxel_indices_for_pc(curr_source_pc_np, voxel_size, area_extents)
        curr_point_FGBG_pred = curr_FGBG_pred[:, curr_voxel_indices[:, 0], curr_voxel_indices[:, 1]].permute(1, 0)

        # compute current loss
        curr_log_softmax_FGBG_pred = F.log_softmax(curr_point_FGBG_pred, dim=1)
        curr_loss_FGBG_pred = torch.sum(- curr_point_FGBG_gt_map * curr_log_softmax_FGBG_pred, dim=1) * curr_FGBG_weight_map
        curr_loss_FGBG_predd = torch.sum(curr_loss_FGBG_pred) / (curr_annotated_point_num + 1e-6)

        # accumulate loss
        loss_FGBG_seg = loss_FGBG_seg + curr_loss_FGBG_predd

    loss_FGBG_seg = loss_FGBG_seg / batch_size
    return loss_FGBG_seg




def CCD_loss(disp_pred, pc_0, pc_0_num, pc_1, pc_1_num, pc_2, pc_2_num, non_empty_map, voxel_size, area_extents,
             epoch, epoch_threshold=10, theta2=1):
    """
    Consistency-aware Chamfer Distance loss
    ----------

    Inputs:
        disp_pred: [B, 2, dim_0, dim_1],    predicted 2D displacement BEV map

        pc_0: [B, M, 3],                    predicted foreground points in the past frame (-0.5s)
        pc_0_num: [B],                      unrepeated foreground point number in each past frame

        pc_1: [B, M, 3],                    predicted foreground points in the current frame (0s)
        pc_1_num: [B],                      unrepeated foreground point number in each current frame

        pc_2: [B, M, 3],                    predicted foreground points in the future frame (+0.5s)
        pc_2_num: [B],                      unrepeated foreground point number in each future frame

        non_empty_map: [B, dim_0, dim_1]    nonempty mask
        voxel_size, area_extents:           voxel size and range of area,

        epoch:                              the number of current training epoch
        epoch_threshold:                    After epoch_threshold epochs, we start to reweight multi-frame Chamfer loss
        theta2:                             hyper-parameter in Gaussian kernel, used in Eq.(6)
    """

    batch_size = disp_pred.shape[0]
    device = disp_pred.device
    loss_disp = torch.zeros((1), device=device, dtype=disp_pred.dtype)

    valid_sample_num = 0
    for batch_index in range(batch_size):

        # 0: past frame; 1: current frame; 2: future frame
        curr_pc_0_num = pc_0_num[batch_index]
        curr_pc_1_num = pc_1_num[batch_index]
        curr_pc_2_num = pc_2_num[batch_index]
        if (curr_pc_0_num > 0) and (curr_pc_1_num > 0) and (curr_pc_2_num > 0):
            valid_sample_num = valid_sample_num + 1
            curr_valid_map = non_empty_map[batch_index]

            # get source and target point clouds, predicted 2D BEV flow
            curr_pc_0_np = pc_0[batch_index, :curr_pc_0_num, :].numpy()     # target pc, past frame
            curr_pc_1_np = pc_1[batch_index, :curr_pc_1_num, :].numpy()     # current pc, source frame
            curr_pc_2_np = pc_2[batch_index, :curr_pc_2_num, :].numpy()     # target pc, future frame
            curr_disp_pred = disp_pred[batch_index, :, :, :]

            # get predicted 3D flow for each point
            curr_voxel_indices = gen_voxel_indices_for_pc(curr_pc_1_np, voxel_size, area_extents)
            curr_point_disp_pred = curr_disp_pred[:, curr_voxel_indices[:, 0], curr_voxel_indices[:, 1]].permute(1, 0)

            # get FG and BG map for the current frame, the map is estimated by the PreSegNet in Stage1
            curr_fg_map = torch.zeros_like(curr_valid_map)
            curr_fg_map[curr_voxel_indices[:, 0], curr_voxel_indices[:, 1]] = 1
            curr_fg_map = curr_fg_map * curr_valid_map
            fg_voxel_num = torch.sum(curr_fg_map)

            curr_bg_map = (1 - curr_fg_map) * curr_valid_map
            bg_voxel_num = torch.sum(curr_bg_map)

            curr_pc_0 = torch.from_numpy(curr_pc_0_np).to(device).float()
            curr_pc_1 = torch.from_numpy(curr_pc_1_np).to(device).float()
            curr_pc_2 = torch.from_numpy(curr_pc_2_np).to(device).float()
            curr_point_3d_disp_pred = torch.cat([curr_point_disp_pred, torch.zeros_like(curr_point_disp_pred[:, 0:1])], -1)

            # compute confidence weights for the three point clouds
            if epoch > epoch_threshold:
                # After epoch_threshold epochs, we start to reweight multi-frame Chamfer loss
                weight_P, weight_C, weight_F = gen_confidence_weight(curr_pc_0, curr_pc_1, curr_pc_2, curr_point_3d_disp_pred, theta2=theta2)
            else:
                weight_P, weight_C, weight_F = None, None, None

            # Consistency-aware Chamfer Distance loss function for the foreground points
            # backward term (backward warped current frame, past frame)
            warped_source_pc_backward = curr_pc_1 - curr_point_3d_disp_pred
            fg_loss_backward = weighted_chamfer_loss(warped_source_pc_backward, curr_pc_0, weight_C, weight_P)

            # forward term (forward warped current frame, future frame)
            warped_source_pc_forward = curr_pc_1 + curr_point_3d_disp_pred
            fg_loss_forward = weighted_chamfer_loss(warped_source_pc_forward, curr_pc_2, weight_C, weight_F)

            fg_loss = (fg_loss_backward + fg_loss_forward) / 2.0

            # generate loss for the background points. Eq.(13)
            bg_gt = torch.zeros_like(curr_disp_pred)    # background points are regarded as static
            bg_loss = torch.sum(torch.abs(curr_disp_pred * curr_bg_map.unsqueeze(0) - bg_gt * curr_bg_map.unsqueeze(0)), 0)
            bg_loss = torch.sum(bg_loss) / (torch.sum(curr_bg_map) + 1e-6)

            # combine the losses from the foreground and the background. Eq.(12)
            curr_loss = (fg_loss * fg_voxel_num + 0.005 * bg_loss * bg_voxel_num) \
                        / (fg_voxel_num + bg_voxel_num + 1e-6)

            loss_disp = loss_disp + curr_loss

    loss_disp = loss_disp / valid_sample_num
    return loss_disp


def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.detach().long(), :]
    return new_points


def find_NN_index(source, target):
    # find the NN point of a source point in the target point set
    sqrdist = square_distance(source, target)
    dist, index = torch.topk(sqrdist, 1, dim=-1, largest=False, sorted=False)
    index = index.squeeze(-1).int()
    return index


def gen_confidence_weight(target_p_pc, source_pc, target_f_pc, disp_pred, theta2=1):

    target_p_pc = target_p_pc.unsqueeze(0)
    source_pc = source_pc.unsqueeze(0)
    target_f_pc = target_f_pc.unsqueeze(0)
    disp_pred = disp_pred.unsqueeze(0)

    # Eq.(3) in paper;
    warped_source_pc_forward = source_pc + disp_pred
    warped_source_pc_backward = source_pc - disp_pred

    # Eq.(4); y_f
    index12 = find_NN_index(warped_source_pc_forward, target_f_pc)
    nn_target_f = index_points(target_f_pc, index12)
    pseudo_flow_f = nn_target_f - source_pc

    # Eq.(5); y_b
    index10 = find_NN_index(warped_source_pc_backward, target_p_pc)
    nn_target_p = index_points(target_p_pc, index10)
    pseudo_flow_b = nn_target_p - source_pc

    # Eq.(6); generate  confidence weights for points in the current (source) frame
    square_flow_diff_C = torch.sum((pseudo_flow_f + pseudo_flow_b) ** 2, -1) + 1e-6
    weight_C = torch.exp(torch.neg(square_flow_diff_C / theta2)) + 1e-6


    # generate confidence weights for the past and future frames

    # Eq.(7); confidence weights for the future (target) frame
    index21 = find_NN_index(target_f_pc, warped_source_pc_forward)
    # Eq.(8);
    weight_F = index_points(weight_C.unsqueeze(-1).contiguous(), index21)
    weight_F = weight_F.squeeze(-1)

    # Eq.(7); confidence weights for the past (target) frame
    index01 = find_NN_index(target_p_pc, warped_source_pc_backward)
    # Eq.(8);
    weight_P = index_points(weight_C.unsqueeze(-1).contiguous(), index01)
    weight_P = weight_P.squeeze(-1)

    return weight_P, weight_C, weight_F


def weighted_chamfer_loss(warped_source_pc, target_pc, weight_source, weight_target):
    warped_source_pc = warped_source_pc.unsqueeze(0)
    target_pc = target_pc.unsqueeze(0)

    sqrdist12 = square_distance(warped_source_pc, target_pc)  # 1 N M
    dist1, index1 = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)     # N
    dist2, index2 = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)      # M
    index1 = index1.squeeze(-1).int()
    index2 = index2.squeeze(1).int()

    nn_target_pc = index_points(target_pc, index1)
    nn_warped_source_pc = index_points(warped_source_pc, index2)

    # L1-norm
    loss_s2t = torch.sum(torch.abs(warped_source_pc - nn_target_pc), -1)
    loss_t2s = torch.sum(torch.abs(nn_warped_source_pc - target_pc), -1)

    if weight_source is None:
        fg_loss = torch.sum(loss_s2t) / loss_s2t.shape[1] + torch.sum(loss_t2s) / loss_t2s.shape[1]
        fg_loss = fg_loss / 2.0
    else:
        fg_loss = torch.sum(loss_s2t * weight_source) /torch.sum(weight_source) +\
                  torch.sum(loss_t2s * weight_target) /torch.sum(weight_target)
        fg_loss = fg_loss / 2.0
    return fg_loss
