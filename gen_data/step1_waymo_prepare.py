"""
This code is to convert the '.tfrecord' files to '.npy' files in Waymo data.

Reference:
https://github.com/open-mmlab/OpenPCDet
"""


# test on python3.7, tensorflow2.6, waymo-open-dataset-tf-2-6-0
# waymo v1.3.2
import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import pickle
from tqdm import tqdm
import argparse

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--waymo_tfrecord_dir', default='/path_to/Waymo/Waymo-tf-data/', type=str)
parser.add_argument('--split', default='training', type=str, help='The data split [training/validation]')
parser.add_argument('--waymo_save_dir', default='/path_to/Waymo/Waymo-npy-data/', type=str)

args = parser.parse_args()

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name

args.waymo_save_dir = check_folder(args.waymo_save_dir)
waymo_tfrecord_dir = check_folder(os.path.join(args.waymo_tfrecord_dir, args.split))
waymo_save_dir = check_folder(os.path.join(args.waymo_save_dir, args.split))


# https://github.com/open-mmlab/OpenPCDet/blob/aa753ec0e941ddb117654810b7e6c16f2efec2f9/pcdet/datasets/waymo/waymo_utils.py#L73
def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.
    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation


FILENAMES = [i for i in os.listdir(waymo_tfrecord_dir) if i.endswith("tfrecord")]
WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


for idx, FILENAME in tqdm(enumerate(FILENAMES, 0), total=len(FILENAMES), smoothing=0.9):
    dataset = tf.data.TFRecordDataset(os.path.join(waymo_tfrecord_dir, FILENAME), compression_type='')
    scene_name = FILENAME.split(".")[0]
    scene_dir = os.path.join(waymo_save_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    segment_info = []
    save_pkl_name = os.path.join(scene_dir, scene_name + ".pkl")

    frame_index = -1
    for data in dataset:
        frame_index += 1

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        frame_time = frame.timestamp_micros * 1e-6
        (range_images, camera_projections, segmentation_labels,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        # ---------------------------- save clouds -----------------------------------
        # only use first return
        points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=(0,)
        )

        points_all = np.concatenate(points, axis=0)
        points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
        points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
        points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)
        save_points = np.concatenate([
            points_all, points_intensity, points_elongation, points_in_NLZ_flag
        ], axis=-1).astype(np.float32)

        save_pc_path = os.path.join(scene_dir, "{:04d}.npy".format(frame_index))

        np.save(save_pc_path, save_points)
        # pc = np.load(save_pc_path)
        # ---------------------------- save anns -----------------------------------

        obj_name, dimensions, locations, heading_angles, obj_ids = [], [], [], [], []

        laser_labels = frame.laser_labels
        for i in range(len(laser_labels)):
            box = laser_labels[i].box
            class_ind = laser_labels[i].type

            loc = [box.center_x, box.center_y, box.center_z]
            heading_angles.append(box.heading)
            class_name = WAYMO_CLASSES[class_ind]
            if class_name == 'unknown':
                continue

            obj_name.append(class_name)

            dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
            locations.append(loc)
            obj_ids.append(laser_labels[i].id)

        annotations = {}
        annotations['name'] = np.array(obj_name)

        annotations['dimensions'] = np.array(dimensions)
        annotations['location'] = np.array(locations)
        annotations['heading_angles'] = np.array(heading_angles)

        annotations['obj_ids'] = np.array(obj_ids)

        frame_info = {}

        frame_info['point_cloud'] = {'num_features': 5, 'lidar_sequence': scene_name, 'sample_idx': frame_index}
        frame_info["time_stamp"] = frame_time
        frame_info['annos'] = annotations
        frame_info['frame_id'] = scene_name + ('_%03d' % frame_index)

        frame_info['pose'] = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)

        segment_info.append(frame_info)

    with open(save_pkl_name, 'wb') as f:
        pickle.dump(segment_info, f)

