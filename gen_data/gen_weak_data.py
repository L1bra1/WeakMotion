"""
This code is to generate Foreground/Background information for the training set of nuScenes data.
And the code is modified based on 'gen_data.py' in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""

from gen_data.nuscenes.nuscenes import NuScenes
import os
from gen_data.nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import argparse
from functools import reduce

from gen_data.nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='/media/ruibo/cc84e27f-82c0-43ca-8f28-5b4c37369175/liruibo/dataset/AA-motion-data/nuScenes/nuScenes-unzipped', type=str, help='Root path to nuScenes dataset')
parser.add_argument('-s', '--split', default='train', type=str, help='The data split [train/val]')
parser.add_argument('-p', '--savepath', default='/home/ruibo/Documents/A-Task/Pillar/AA-code-for-release/tmp-data/nuScenes/weak-data', type=str, help='Directory for saving the generated data')

args = parser.parse_args()

nusc = NuScenes(version='v1.0-trainval', dataroot=args.root, verbose=True)
print("Total number of scenes:", len(nusc.scene))

class_map = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'human.pedestrian': 2,
             'vehicle.bicycle': 3}  # background: 0, other: 4


if args.split == 'train':
    num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
    nsweeps_back = 30  # Number of frames back to the history (including the current timestamp)
    nsweeps_forward = 20  # Number of frames into the future (does not include the current timestamp)
    skip_frame = 0  # The number of frames skipped for the adjacent sequence
    num_adj_seqs = 2  # number of adjacent sequences, among which the time gap is \delta t
else:
    num_keyframe_skipped = 1
    nsweeps_back = 25  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
    nsweeps_forward = 20
    skip_frame = 0
    num_adj_seqs = 1


# The specifications for BEV maps
voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
num_past_frames_for_bev_seq = 5  # the number of past frames for BEV map sequence


scenes = np.load('split.npy', allow_pickle=True).item().get(args.split)
print("Split: {}, which contains {} scenes.".format(args.split, len(scenes)))

args.savepath = check_folder(args.savepath)
sample_info_directory = check_folder(os.path.join(args.savepath, args.split + '_sample_info'))
args.savepath = check_folder(os.path.join(args.savepath, args.split))

def gen_data():
    res_scenes = list()
    for s in scenes:
        s_id = s.split('_')[1]
        res_scenes.append(int(s_id))

    for scene_idx in res_scenes:
        curr_scene = nusc.scene[scene_idx]

        first_sample_token = curr_scene['first_sample_token']
        curr_sample = nusc.get('sample', first_sample_token)
        curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])


        adj_seq_cnt = 0
        save_seq_cnt = 0  # only used for save data file name

        save_weak_dict_list = list()
        pc_random_index_dict = dict()

        # Iterate each sample data
        print("Processing scene {} ...".format(scene_idx))
        while curr_sample_data['next'] != '':

            all_times = \
                LidarPointCloud.from_file_multisweep_bf_sample_data_return_times(nusc, curr_sample_data,
                                                                    nsweeps_back=nsweeps_back,
                                                                    nsweeps_forward=nsweeps_forward)

            _, sort_idx = np.unique(all_times, return_index=True)
            unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
            num_sweeps = len(unique_times)

            # Make sure we have sufficient past and future sweeps
            if num_sweeps != (nsweeps_back + nsweeps_forward):

                # Skip some keyframes if necessary
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample['next'] != '':
                        curr_sample = nusc.get('sample', curr_sample['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])

                # Reset
                adj_seq_cnt = 0
                save_weak_dict_list = list()
                continue


            if adj_seq_cnt == 0:

                save_weak_dict = dict()

                lidar_curr_sample = curr_sample
                key_timestamps = np.zeros(3)

                lidar_sd_token_data = nusc.get('sample_data', lidar_curr_sample['data']['LIDAR_TOP'])
                _, ref_from_car, car_from_global, ref_time = get_pc_pose(lidar_sd_token_data, inverse=True)

                lidar_curr_sample = nusc.get('sample', lidar_curr_sample['prev'])
                # 0 past (-0.5s); 1 current (0s); 2 future (+0.5s)
                for key_frame_index in range(3):

                    lidar_sd_token_data = nusc.get('sample_data', lidar_curr_sample['data']['LIDAR_TOP'])
                    lidar_sd_token = lidar_sd_token_data['token']
                    save_weak_dict['token_' + str(key_frame_index)] = lidar_sd_token

                    current_pc, car_from_current, global_from_car, timestamp = get_pc_pose(lidar_sd_token_data, inverse=False)
                    trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                    current_pc[:3, :] = trans_matrix.dot(np.vstack((current_pc[:3, :], np.ones(current_pc.shape[1]))))[:3, :]
                    save_weak_dict['synchronized_pc_' + str(key_frame_index)] = current_pc[:3, :]

                    lidarseg_labels_filename = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_sd_token)['filename'])

                    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                    key_timestamps[key_frame_index] = 1e-6 * lidar_sd_token_data['timestamp']

                    save_weak_dict['points_label_' + str(key_frame_index)] = points_label

                    sample_idx, pc_random_index_dict = gen_random_index_for_pc(current_pc, lidar_sd_token, pc_random_index_dict)
                    save_weak_dict['sample_idx_' + str(key_frame_index)] = sample_idx

                    if key_frame_index != 2:
                        lidar_curr_sample = nusc.get('sample', lidar_curr_sample['next'])

                save_weak_dict['key_timestamp'] = key_timestamps
                save_weak_dict_list.append(save_weak_dict)

            adj_seq_cnt += 1
            if adj_seq_cnt == num_adj_seqs:

                for seq_idx, seq_weak_dict in enumerate(save_weak_dict_list):

                    # save the data
                    save_directory = check_folder(os.path.join(args.savepath, str(scene_idx) + '_' + str(save_seq_cnt)))
                    save_file_name = os.path.join(save_directory, str(seq_idx) + '.npy')
                    np.save(save_file_name, arr=seq_weak_dict)

                    print("  >> {} - {} Finish sample: {}, sequence {}".format(seq_weak_dict['key_timestamp'][0], seq_weak_dict['key_timestamp'][1], save_seq_cnt, seq_idx))

                save_seq_cnt += 1
                adj_seq_cnt = 0
                save_weak_dict_list = list()


                # Skip some keyframes if necessary
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample['next'] != '':
                        curr_sample = nusc.get('sample', curr_sample['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            else:
                flag = False
                for _ in range(skip_frame + 1):
                    if curr_sample_data['next'] != '':
                        curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more sample frames
                    break

        save_file_name = os.path.join(sample_info_directory, str(scene_idx) + '_sample_info.npy')
        np.save(save_file_name, arr=pc_random_index_dict)


def get_pc_pose(ref_sd_rec, inverse = True):
    # Get reference pose and timestamp
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']),
                                    inverse=inverse)

    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=inverse)

    scan = np.fromfile((os.path.join(nusc.dataroot, ref_sd_rec['filename'])), dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]

    return points.T, ref_from_car, car_from_global, ref_time


def gen_random_index_for_pc(pc, token, pc_random_index_dict):
    curr_source_num = pc.shape[1]
    if token in pc_random_index_dict.keys():
        sample_idx = pc_random_index_dict[token]
        assert curr_source_num == len(sample_idx)
    else:
        sample_idx = np.random.choice(curr_source_num, curr_source_num, replace=False)
        pc_random_index_dict[token] = sample_idx.astype(np.int32)
    return sample_idx, pc_random_index_dict

if __name__ == "__main__":
    gen_data()

