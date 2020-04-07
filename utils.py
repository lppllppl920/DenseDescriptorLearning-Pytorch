'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

# Uncomment these three lines if OPENCV error related to ROS happens
# import sys
# if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from plyfile import PlyData, PlyElement
import yaml
import random
import torch
import torchvision.utils as vutils
import tqdm
import matplotlib.pyplot as plt

import dataset
import models


def write_point_cloud(path, point_cloud):
    point_clouds_list = []
    for i in range(point_cloud.shape[0]):
        point_clouds_list.append((point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2], point_cloud[i, 3],
                                  point_cloud[i, 4], point_cloud[i, 5]))

    vertex = np.array(point_clouds_list,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(path)
    return


def scatter_points_to_image(image, visible_locations_x, visible_locations_y, invisible_locations_x,
                            invisible_locations_y, only_visible, point_size):
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(image.shape[1] / 100, image.shape[0] / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image, zorder=1)
    plt.scatter(x=visible_locations_x, y=visible_locations_y, s=point_size, alpha=0.5, c='b', zorder=2)
    if not only_visible:
        plt.scatter(x=invisible_locations_x, y=invisible_locations_y, s=point_size, alpha=0.5, c='y', zorder=3)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def get_color_file_names_by_bag(root, training_patient_id, validation_patient_id, testing_patient_id):
    training_image_list = []
    validation_image_list = []
    testing_image_list = []

    if not isinstance(training_patient_id, list):
        training_patient_id = [training_patient_id]
    if not isinstance(validation_patient_id, list):
        validation_patient_id = [validation_patient_id]
    if not isinstance(testing_patient_id, list):
        testing_patient_id = [testing_patient_id]

    for id in training_patient_id:
        training_image_list += list(root.glob('{:d}/*/images/0*.jpg'.format(id)))
    for id in testing_patient_id:
        testing_image_list += list(root.glob('{:d}/*/images/0*.jpg'.format(id)))
    for id in validation_patient_id:
        validation_image_list += list(root.glob('{:d}/*/images/0*.jpg'.format(id)))

    training_image_list.sort()
    testing_image_list.sort()
    validation_image_list.sort()
    return training_image_list, validation_image_list, testing_image_list


def get_parent_folder_names(root, id_range):
    folder_list = []
    for id in id_range:
        folder_list += list(root.glob('{:d}/*/'.format(id)))
    folder_list.sort()
    return folder_list


def downsample_and_crop_mask(mask, downsampling_factor, divide, suggested_h=None, suggested_w=None):
    downsampled_mask = cv2.resize(mask, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    end_h_index = downsampled_mask.shape[0]
    end_w_index = downsampled_mask.shape[1]
    # divide is related to the pooling times of the teacher model
    indexes = np.where(downsampled_mask >= 200)
    h = indexes[0].max() - indexes[0].min()
    w = indexes[1].max() - indexes[1].min()

    remainder_h = h % divide
    remainder_w = w % divide

    increment_h = divide - remainder_h
    increment_w = divide - remainder_w

    target_h = h + increment_h
    target_w = w + increment_w

    start_h = max(indexes[0].min() - increment_h // 2, 0)
    end_h = start_h + target_h

    start_w = max(indexes[1].min() - increment_w // 2, 0)
    end_w = start_w + target_w

    if suggested_h is not None:
        if suggested_h != h:
            remain_h = suggested_h - target_h
            start_h = max(start_h - remain_h // 2, 0)
            end_h = min(suggested_h + start_h, end_h_index)
            start_h = end_h - suggested_h

    if suggested_w is not None:
        if suggested_w != w:
            remain_w = suggested_w - target_w
            start_w = max(start_w - remain_w // 2, 0)
            end_w = min(suggested_w + start_w, end_w_index)
            start_w = end_w - suggested_w

    kernel = np.ones((5, 5), np.uint8)
    downsampled_mask_erode = cv2.erode(downsampled_mask, kernel, iterations=1)
    cropped_mask = downsampled_mask_erode[start_h:end_h, start_w:end_w]
    return cropped_mask, start_h, end_h, start_w, end_w


def read_visible_view_indexes(prefix_seq):
    path = prefix_seq / 'visible_view_indexes'
    if not path.exists():
        return []

    visible_view_indexes = []
    with open(str(path)) as fp:
        for line in fp:
            visible_view_indexes.append(int(line))
    return visible_view_indexes


def read_selected_indexes(prefix_seq):
    selected_indexes = []
    with open(str(prefix_seq / 'selected_indexes')) as fp:
        for line in fp:
            selected_indexes.append(int(line))
    return selected_indexes


def read_camera_intrinsic_per_view(prefix_seq):
    camera_intrinsics = []
    param_count = 0
    temp_camera_intrincis = np.zeros((3, 4))
    with open(str(prefix_seq / 'camera_intrinsics_per_view')) as fp:
        for line in fp:
            # Focal length
            if param_count == 0:
                temp_camera_intrincis[0][0] = float(line)
                param_count += 1
            elif param_count == 1:
                temp_camera_intrincis[1][1] = float(line)
                param_count += 1
            elif param_count == 2:
                temp_camera_intrincis[0][2] = float(line)
                param_count += 1
            elif param_count == 3:
                temp_camera_intrincis[1][2] = float(line)
                temp_camera_intrincis[2][2] = 1.0
                camera_intrinsics.append(temp_camera_intrincis)
                temp_camera_intrincis = np.zeros((3, 4))
                param_count = 0
    return camera_intrinsics


def modify_camera_intrinsic_matrix(intrinsic_matrix, start_h, start_w, downsampling_factor):
    intrinsic_matrix_modified = np.copy(intrinsic_matrix)
    intrinsic_matrix_modified[0][0] = intrinsic_matrix[0][0] / downsampling_factor
    intrinsic_matrix_modified[1][1] = intrinsic_matrix[1][1] / downsampling_factor
    intrinsic_matrix_modified[0][2] = intrinsic_matrix[0][2] / downsampling_factor - start_w
    intrinsic_matrix_modified[1][2] = intrinsic_matrix[1][2] / downsampling_factor - start_h
    return intrinsic_matrix_modified


def read_point_cloud(path):
    lists_3D_points = []
    plydata = PlyData.read(path)
    for n in range(plydata['vertex'].count):
        temp = list(plydata['vertex'][n])
        lists_3D_points.append([temp[0], temp[1], temp[2], 1.0])
    return lists_3D_points


def read_view_indexes_per_point(prefix_seq, visible_view_indexes, point_cloud_count):
    # Read the view indexes per point into a 2-dimension binary matrix
    view_indexes_per_point = np.zeros((point_cloud_count, len(visible_view_indexes)))
    point_count = -1
    with open(str(prefix_seq / 'view_indexes_per_point')) as fp:
        for line in fp:
            if int(line) < 0:
                point_count = point_count + 1
            else:
                view_indexes_per_point[point_count][visible_view_indexes.index(int(line))] = 1
    return view_indexes_per_point


def overlapping_visible_view_indexes_per_point(visible_view_indexes_per_point, visible_interval):
    temp_array = np.copy(visible_view_indexes_per_point)
    view_count = visible_view_indexes_per_point.shape[1]
    for i in range(view_count):
        visible_view_indexes_per_point[:, i] = \
            np.sum(temp_array[:, max(0, i - visible_interval):min(view_count, i + visible_interval)], axis=1)

    return visible_view_indexes_per_point


def read_pose_data(prefix_seq):
    stream = open(str(prefix_seq / "motion.yaml"), 'r')
    doc = yaml.load(stream)
    keys, values = doc.items()
    poses = values[1]
    return poses


def get_extrinsic_matrix_and_projection_matrix(poses, intrinsic_matrix, visible_view_count):
    projection_matrices = []
    extrinsic_matrices = []
    for i in range(visible_view_count):
        rigid_transform = quaternion_matrix(
            [poses["poses[" + str(i) + "]"]['orientation']['w'], poses["poses[" + str(i) + "]"]['orientation']['x'],
             poses["poses[" + str(i) + "]"]['orientation']['y'],
             poses["poses[" + str(i) + "]"]['orientation']['z']])
        rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
        rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
        rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']

        transform = np.asmatrix(rigid_transform)
        extrinsic_matrices.append(transform)
        projection_matrices.append(np.dot(intrinsic_matrix, transform))

    return extrinsic_matrices, projection_matrices


def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def global_scale_estimation(extrinsics, point_cloud):
    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)

    for i, extrinsic in enumerate(extrinsics):
        if i == 0:
            max_bound = extrinsic[:3, 3]
            min_bound = extrinsic[:3, 3]
        else:
            temp = extrinsic[:3, 3]
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_1 = np.linalg.norm(max_bound - min_bound, ord=2)

    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)
    for i, point in enumerate(point_cloud):
        if i == 0:
            max_bound = np.asarray(point[:3], dtype=np.float32)
            min_bound = np.asarray(point[:3], dtype=np.float32)
        else:
            temp = np.asarray(point[:3], dtype=np.float32)
            if np.any(np.isnan(temp)):
                continue
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_2 = np.linalg.norm(max_bound - min_bound, ord=2)

    return max(norm_1, norm_2)


def get_color_imgs(prefix_seq, visible_view_indexes, start_h, end_h, start_w, end_w, downsampling_factor):
    imgs = []
    for i in visible_view_indexes:
        img = cv2.imread(str(prefix_seq / "{:08d}.jpg".format(i)))
        downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
        cropped_downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        imgs.append(cropped_downsampled_img)
    height, width, channel = imgs[0].shape
    imgs = np.array(imgs, dtype="float32")
    imgs = np.reshape(imgs, (-1, height, width, channel))
    return imgs


def get_clean_point_list(imgs, point_cloud, view_indexes_per_point, mask_boundary, inlier_percentage,
                         projection_matrices,
                         extrinsic_matrices):
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    assert (inlier_percentage > 0.0 and inlier_percentage <= 1.0)
    point_cloud_contamination_accumulator = np.zeros(array_3D_points.shape[0], dtype=np.int32)
    point_cloud_appearance_count = np.zeros(array_3D_points.shape[0], dtype=np.int32)
    height, width, channel = imgs[0].shape
    valid_frame_count = 0
    mask_boundary = mask_boundary.reshape((-1, 1))
    for i in range(len(projection_matrices)):
        img = imgs[i]
        projection_matrix = projection_matrices[i]
        extrinsic_matrix = extrinsic_matrices[i]
        img = np.array(img, dtype=np.float32) / 255.0
        img_filtered = cv2.bilateralFilter(src=img, d=7, sigmaColor=25, sigmaSpace=25)
        img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL)

        view_indexes_frame = np.asarray(view_indexes_per_point[:, i]).reshape((-1))
        visible_point_indexes = np.where(view_indexes_frame > 0.5)
        visible_point_indexes = visible_point_indexes[0]
        points_3D_camera = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
        points_3D_camera = points_3D_camera / points_3D_camera[:, 3].reshape((-1, 1))

        points_2D_image = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
        points_2D_image = points_2D_image / points_2D_image[:, 2].reshape((-1, 1))

        visible_points_2D_image = points_2D_image[visible_point_indexes, :].reshape((-1, 3))
        visible_points_3D_camera = points_3D_camera[visible_point_indexes, :].reshape((-1, 4))
        indexes = np.where((visible_points_2D_image[:, 0] <= width - 1) & (visible_points_2D_image[:, 0] >= 0) &
                           (visible_points_2D_image[:, 1] <= height - 1) & (visible_points_2D_image[:, 1] >= 0)
                           & (visible_points_3D_camera[:, 2] > 0))
        indexes = indexes[0]
        in_image_point_1D_locations = (np.round(visible_points_2D_image[indexes, 0]) +
                                       np.round(visible_points_2D_image[indexes, 1]) * width).astype(
            np.int32).reshape((-1))
        temp_mask = mask_boundary[in_image_point_1D_locations, :]
        indexes_2 = np.where(temp_mask[:, 0] == 255)
        indexes_2 = indexes_2[0]
        in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2]
        points_depth = visible_points_3D_camera[indexes[indexes_2], 2]
        img_hsv = img_hsv.reshape((-1, 3))
        points_brightness = img_hsv[in_mask_point_1D_locations, 2]
        sanity_array = points_depth ** 2 * points_brightness
        point_cloud_appearance_count[visible_point_indexes[indexes[indexes_2]]] += 1
        if sanity_array.shape[0] < 2:
            continue
        valid_frame_count += 1
        sanity_threshold_min, sanity_threshold_max = compute_sanity_threshold(sanity_array, inlier_percentage)
        indexes_3 = np.where((sanity_array <= sanity_threshold_min) | (sanity_array >= sanity_threshold_max))
        indexes_3 = indexes_3[0]
        point_cloud_contamination_accumulator[visible_point_indexes[indexes[indexes_2[indexes_3]]]] += 1

    clean_point_cloud_array = (point_cloud_contamination_accumulator < point_cloud_appearance_count / 2).astype(
        np.float32)
    print("{} points eliminated".format(int(clean_point_cloud_array.shape[0] - np.sum(clean_point_cloud_array))))
    return clean_point_cloud_array


def compute_sanity_threshold(sanity_array, inlier_percentage):
    # Use histogram to cluster into different contaminated levels
    hist, bin_edges = np.histogram(sanity_array, bins=np.arange(1000) * np.max(sanity_array) / 1000.0,
                                   density=True)
    histogram_percentage = hist * np.diff(bin_edges)
    percentage = inlier_percentage
    # Let's assume there are a certain percent of points in each frame that are not contaminated
    # Get sanity threshold from counting histogram bins
    max_index = np.argmax(histogram_percentage)
    histogram_sum = histogram_percentage[max_index]
    pos_counter = 1
    neg_counter = 1
    # Assume the sanity value is a one-peak distribution
    while True:
        if max_index + pos_counter < len(histogram_percentage):
            histogram_sum = histogram_sum + histogram_percentage[max_index + pos_counter]
            pos_counter = pos_counter + 1
            if histogram_sum >= percentage:
                sanity_threshold_max = bin_edges[max_index + pos_counter]
                sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                break

        if max_index - neg_counter >= 0:
            histogram_sum = histogram_sum + histogram_percentage[max_index - neg_counter]
            neg_counter = neg_counter + 1
            if histogram_sum >= percentage:
                sanity_threshold_max = bin_edges[max_index + pos_counter]
                sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                break

        if max_index + pos_counter >= len(histogram_percentage) and max_index - neg_counter < 0:
            sanity_threshold_max = np.max(bin_edges)
            sanity_threshold_min = np.min(bin_edges)
            break
    return sanity_threshold_min, sanity_threshold_max


def generating_pos_and_increment(idx, visible_view_indexes, adjacent_range):
    # We use the remainder of the overall idx to retrieve the visible view
    visible_view_idx = idx % len(visible_view_indexes)

    adjacent_range_list = []
    adjacent_range_list.append(adjacent_range[0])
    adjacent_range_list.append(adjacent_range[1])

    if len(visible_view_indexes) <= 2 * adjacent_range_list[0]:
        adjacent_range_list[0] = len(visible_view_indexes) // 2

    if visible_view_idx <= adjacent_range_list[0] - 1:
        increment = random.randint(adjacent_range_list[0],
                                   min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
    elif visible_view_idx >= len(visible_view_indexes) - adjacent_range_list[0]:
        increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    else:
        # which direction should we increment
        direction = random.randint(0, 1)
        if direction == 1:
            increment = random.randint(adjacent_range_list[0],
                                       min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
        else:
            increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    return [visible_view_idx, increment]


def get_pair_color_imgs(prefix_seq, pair_indexes, start_h, end_h, start_w, end_w, downsampling_factor):
    imgs = []
    for i in pair_indexes:
        img = cv2.imread(str(prefix_seq / "{:08d}.jpg".format(i)))
        downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
        downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
        imgs.append(downsampled_img)

    height, width, channel = imgs[0].shape
    imgs = np.asarray(imgs, dtype=np.float32)
    imgs = imgs.reshape((-1, height, width, channel))
    return imgs


def get_torch_training_data_feature_matching(height, width, pair_projections, pair_indexes, point_cloud,
                                             mask_boundary, view_indexes_per_point, clean_point_list,
                                             visible_view_indexes):
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    for i in range(2):
        projection_matrix = pair_projections[i]
        if i == 0:
            points_2D_image_1 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_1 = np.round(points_2D_image_1 / points_2D_image_1[:, 2].reshape((-1, 1)))
        else:
            points_2D_image_2 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_2 = np.round(points_2D_image_2 / points_2D_image_2[:, 2].reshape((-1, 1)))

    point_visibility_1 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[0])]).reshape(
        (-1))
    point_visibility_2 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[1])]).reshape(
        (-1))

    visible_point_indexes_1 = np.where((point_visibility_1 > 0.5) & (clean_point_list > 0.5))
    visible_point_indexes_1 = visible_point_indexes_1[0]

    visible_point_indexes_2 = np.where((point_visibility_2 > 0.5) & (clean_point_list > 0.5))
    visible_point_indexes_2 = visible_point_indexes_2[0]

    visible_points_2D_image_1 = points_2D_image_1[visible_point_indexes_1, :].reshape((-1, 3))
    visible_points_2D_image_2 = points_2D_image_2[visible_point_indexes_2, :].reshape((-1, 3))

    in_image_indexes_1 = np.where(
        (visible_points_2D_image_1[:, 0] <= width - 1) & (visible_points_2D_image_1[:, 0] >= 0) &
        (visible_points_2D_image_1[:, 1] <= height - 1) & (visible_points_2D_image_1[:, 1] >= 0))
    in_image_indexes_1 = in_image_indexes_1[0]

    in_image_indexes_2 = np.where(
        (visible_points_2D_image_2[:, 0] <= width - 1) & (visible_points_2D_image_2[:, 0] >= 0) &
        (visible_points_2D_image_2[:, 1] <= height - 1) & (visible_points_2D_image_2[:, 1] >= 0))
    in_image_indexes_2 = in_image_indexes_2[0]

    in_image_point_1D_locations_1 = (np.round(visible_points_2D_image_1[in_image_indexes_1, 0]) +
                                     np.round(visible_points_2D_image_1[in_image_indexes_1, 1]) * width).astype(
        np.int32).reshape((-1))

    in_image_point_1D_locations_2 = (np.round(visible_points_2D_image_2[in_image_indexes_2, 0]) +
                                     np.round(visible_points_2D_image_2[in_image_indexes_2, 1]) * width).astype(
        np.int32).reshape((-1))

    mask_boundary = mask_boundary.reshape((-1, 1))
    temp_mask_1 = mask_boundary[in_image_point_1D_locations_1, :]
    in_mask_indexes_1 = np.where(temp_mask_1[:, 0] == 255)
    in_mask_indexes_1 = in_mask_indexes_1[0]

    temp_mask_2 = mask_boundary[in_image_point_1D_locations_2, :]
    in_mask_indexes_2 = np.where(temp_mask_2[:, 0] == 255)
    in_mask_indexes_2 = in_mask_indexes_2[0]

    common_visible_point_indexes = list(
        np.intersect1d(np.asarray(visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]]),
                       np.asarray(visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]]), assume_unique=True))

    feature_matches = np.concatenate(
        [points_2D_image_1[common_visible_point_indexes, :2], points_2D_image_2[common_visible_point_indexes, :2]],
        axis=1)

    return feature_matches


def get_torch_testing_data_feature_matching(height, width, pair_projections, pair_indexes, point_cloud,
                                            mask_boundary, view_indexes_per_point, clean_point_list):
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    for i in range(2):
        projection_matrix = pair_projections[i]
        if i == 0:
            points_2D_image_1 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_1 = np.round(points_2D_image_1 / points_2D_image_1[:, 2].reshape((-1, 1)))
        else:
            points_2D_image_2 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_2 = np.round(points_2D_image_2 / points_2D_image_2[:, 2].reshape((-1, 1)))

    point_visibility_1 = np.asarray(view_indexes_per_point[:, pair_indexes[0]]).reshape(
        (-1))
    point_visibility_2 = np.asarray(view_indexes_per_point[:, pair_indexes[1]]).reshape(
        (-1))

    visible_point_indexes_1 = np.where((point_visibility_1 > 0.5) & (clean_point_list > 0.5))
    visible_point_indexes_1 = visible_point_indexes_1[0]

    visible_point_indexes_2 = np.where((point_visibility_2 > 0.5) & (clean_point_list > 0.5))
    visible_point_indexes_2 = visible_point_indexes_2[0]

    visible_points_2D_image_1 = points_2D_image_1[visible_point_indexes_1, :].reshape((-1, 3))
    visible_points_2D_image_2 = points_2D_image_2[visible_point_indexes_2, :].reshape((-1, 3))

    in_image_indexes_1 = np.where(
        (visible_points_2D_image_1[:, 0] <= width - 1) & (visible_points_2D_image_1[:, 0] >= 0) &
        (visible_points_2D_image_1[:, 1] <= height - 1) & (visible_points_2D_image_1[:, 1] >= 0))
    in_image_indexes_1 = in_image_indexes_1[0]

    in_image_indexes_2 = np.where(
        (visible_points_2D_image_2[:, 0] <= width - 1) & (visible_points_2D_image_2[:, 0] >= 0) &
        (visible_points_2D_image_2[:, 1] <= height - 1) & (visible_points_2D_image_2[:, 1] >= 0))
    in_image_indexes_2 = in_image_indexes_2[0]

    in_image_point_1D_locations_1 = (np.round(visible_points_2D_image_1[in_image_indexes_1, 0]) +
                                     np.round(visible_points_2D_image_1[in_image_indexes_1, 1]) * width).astype(
        np.int32).reshape((-1))

    in_image_point_1D_locations_2 = (np.round(visible_points_2D_image_2[in_image_indexes_2, 0]) +
                                     np.round(visible_points_2D_image_2[in_image_indexes_2, 1]) * width).astype(
        np.int32).reshape((-1))

    mask_boundary = mask_boundary.reshape((-1, 1))
    temp_mask_1 = mask_boundary[in_image_point_1D_locations_1, :]
    in_mask_indexes_1 = np.where(temp_mask_1[:, 0] == 255)
    in_mask_indexes_1 = in_mask_indexes_1[0]

    temp_mask_2 = mask_boundary[in_image_point_1D_locations_2, :]
    in_mask_indexes_2 = np.where(temp_mask_2[:, 0] == 255)
    in_mask_indexes_2 = in_mask_indexes_2[0]

    common_visible_point_indexes = list(
        np.intersect1d(np.asarray(visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]]),
                       np.asarray(visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]]), assume_unique=True))

    feature_matches = np.concatenate(
        [points_2D_image_1[common_visible_point_indexes, :2], points_2D_image_2[common_visible_point_indexes, :2]],
        axis=1)

    return feature_matches


def generate_heatmap_from_locations(feature_2D_locations, height, width, sigma):
    sample_size, _ = feature_2D_locations.shape

    feature_2D_locations = np.reshape(feature_2D_locations, (sample_size, 4))

    source_heatmaps = []
    target_heatmaps = []

    sigma_2 = sigma ** 2
    for i in range(sample_size):
        x = feature_2D_locations[i, 0]
        y = feature_2D_locations[i, 1]

        x_2 = feature_2D_locations[i, 2]
        y_2 = feature_2D_locations[i, 3]

        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), sparse=False, indexing='ij')

        source_grid_x = x_grid - x
        source_grid_y = y_grid - y

        target_grid_x = x_grid - x_2
        target_grid_y = y_grid - y_2

        heatmap = np.exp(-(source_grid_x ** 2 + source_grid_y ** 2) / (2.0 * sigma_2))
        heatmap_2 = np.exp(-(target_grid_x ** 2 + target_grid_y ** 2) / (2.0 * sigma_2))

        source_heatmaps.append(heatmap)
        target_heatmaps.append(heatmap_2)

    source_heatmaps = np.asarray(source_heatmaps, dtype=np.float32).reshape((sample_size, height, width))
    target_heatmaps = np.asarray(target_heatmaps, dtype=np.float32).reshape((sample_size, height, width))

    return source_heatmaps, target_heatmaps


def type_float_and_reshape(array, shape):
    array = array.astype(np.float32)
    return array.reshape(shape)


def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal", gpu_id=0):
    assert (torch.cuda.is_available())
    net = net.cuda(gpu_id)
    if type == "glorot":
        glorot_weight_zero_bias(net, distribution=distribution)
    else:
        kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net


def glorot_weight_zero_bias(model, distribution="uniform"):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    distribution: string
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    torch.nn.init.xavier_normal_(module.weight, gain=1)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    if activation_mode == "leaky_relu":
        print("Leaky relu is not supported yet")
        assert False

    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=activation_mode)
                else:
                    torch.nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=activation_mode)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def display_results(colors_1, colors_2, feature_maps_1, feature_maps_2, boundaries, response_map_1, gt_heatmaps_1,
                    response_map_2, gt_heatmaps_2, descriptor, cross_check_distance, step, writer, phase="Train"):
    batch_size, _, height, width = colors_1.shape
    x = keypoints_descriptors_extraction(descriptor=descriptor, color_1=colors_1[0],
                                         color_2=colors_2[0],
                                         boundary=boundaries[0])

    if x is None:
        return -1

    display_1 = boundaries * response_map_1[:, 0].view(batch_size, 1, height, width)
    max_1, _ = torch.max(display_1.view(-1, height * width), dim=1)
    display_1 = display_1 / max_1.view(-1, 1, 1, 1)

    display_2 = boundaries * response_map_2[:, 0].view(batch_size, 1, height, width)
    max_2, _ = torch.max(display_2.view(-1, height * width), dim=1)
    display_2 = display_2 / max_2.view(-1, 1, 1, 1)

    target_colors_1_display = \
        display_colors(1, step, writer,
                       colors_1 - 1.0 * gt_heatmaps_1[:, 0].view(batch_size, 1, height,
                                                                 width),
                       is_return_image=True, phase=phase)
    target_colors_2_display = \
        display_colors(2, step, writer,
                       colors_2 - 1.0 * gt_heatmaps_2[:, 0].view(batch_size, 1, height,
                                                                 width),
                       is_return_image=True, phase=phase)
    detected_colors_1_display = display_colors(1, step, writer,
                                               colors_1 - 1.0 * display_1,
                                               is_return_image=True,
                                               phase=phase)
    detected_colors_2_display = display_colors(2, step, writer,
                                               colors_2 - 1.0 * display_2,
                                               is_return_image=True,
                                               phase=phase)
    final_heatmap_1_display = display_feature_response_map(1, step, "pred_final_heatmap",
                                                           writer,
                                                           display_1,
                                                           phase=phase,
                                                           color_reverse=True,
                                                           is_return_image=True)
    final_heatmap_2_display = display_feature_response_map(2, step, "pred_final_heatmap",
                                                           writer,
                                                           display_2,
                                                           phase=phase,
                                                           color_reverse=True,
                                                           is_return_image=True)

    stack_and_display(phase=phase, title="Target_and_detected_t1_d2_t2_d1", step=step,
                      writer=writer,
                      image_list=[target_colors_1_display, detected_colors_2_display,
                                  target_colors_2_display, detected_colors_1_display])

    stack_and_display(phase=phase, title="Heatmap_pattern_r1_f1_e1_r2_f2_e2", step=step,
                      writer=writer,
                      image_list=[final_heatmap_1_display,
                                  final_heatmap_2_display])

    kps_1, kps_2, des_1, des_2, kps_1D_1, kps_1D_2 = x
    feature_matches_display_dl, feature_matches_display_sift = feature_matching_single(
        color_1=colors_1[0], color_2=colors_2[0],
        feature_map_1=feature_maps_1[0],
        feature_map_2=feature_maps_2[0],
        kps_1D_1=kps_1D_1,
        des_1=des_1, des_2=des_2,
        kps_1=kps_1, kps_2=kps_2,
        cross_check_distance=cross_check_distance)

    stack_and_display(phase=phase, title="Feature_matches_dl_sift", step=step,
                      writer=writer,
                      image_list=[feature_matches_display_dl,
                                  feature_matches_display_sift])

    return 0


def display_colors(idx, step, writer, colors_1, phase, is_return_image):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display = np.moveaxis(colors_display.data.cpu().numpy(),
                                 source=[0, 1, 2], destination=[2, 0, 1])
    colors_display[colors_display < 0.0] = 0.0
    colors_display[colors_display > 1.0] = 1.0

    if is_return_image:
        return colors_display
    else:
        writer.add_image(phase + '/Images/Color_' + str(idx), colors_display, step, dataformats="HWC")
        return


def stack_and_display(phase, title, step, writer, image_list):
    writer.add_image(phase + '/Images/' + title, np.vstack(image_list), step, dataformats='HWC')
    return


def keypoints_descriptors_extraction(descriptor, color_1, color_2, boundary):
    color_1 = color_1.data.cpu().numpy()
    boundary = boundary.data.cpu().numpy()
    _, height, width = color_1.shape
    color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
    color_1 = np.asarray(255 * (color_1 * 0.5 + 0.5), dtype=np.uint8)
    boundary = np.asarray(255 * boundary.reshape((height, width)), dtype=np.uint8)
    kps_1, des_1 = descriptor.detectAndCompute(color_1, mask=boundary)

    color_2 = color_2.data.cpu().numpy()
    color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])
    color_2 = np.asarray(255 * (color_2 * 0.5 + 0.5), dtype=np.uint8)
    kps_2, des_2 = descriptor.detectAndCompute(color_2, mask=boundary)

    kps_1D_1 = []
    kps_1D_2 = []

    if kps_1 is None or kps_2 is None or len(kps_1) == 0 or len(kps_2) == 0:
        return None

    for point in kps_1:
        kps_1D_1.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
    for point in kps_2:
        kps_1D_2.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)

    return kps_1, kps_2, des_1, des_2, np.asarray(kps_1D_1), np.asarray(kps_1D_2)


def display_feature_response_map(idx, step, title, writer, feature_response_heat_map, phase,
                                 color_reverse=True, is_return_image=False):
    batch_size, _, height, width = feature_response_heat_map.shape
    feature_response_heat_map = feature_response_heat_map.view(batch_size, 1, height, width)
    heatmap_display = vutils.make_grid(feature_response_heat_map, normalize=False, scale_each=True)
    heatmap_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(heatmap_display.data.cpu().numpy(),
                                                                   source=[0, 1, 2], destination=[2, 0, 1])),
                                        cv2.COLORMAP_HOT)
    if color_reverse:
        heatmap_display = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2RGB)

    if is_return_image:
        return heatmap_display
    else:
        writer.add_image(phase + '/Images/' + title + str(idx), heatmap_display, step, dataformats="HWC")
        return


def feature_matching_single(color_1, color_2, feature_map_1, feature_map_2, kps_1D_1, des_1, des_2,
                            cross_check_distance, kps_1, kps_2, gpu_id=0):
    with torch.no_grad():
        color_1 = color_1.data.cpu().numpy()
        color_2 = color_2.data.cpu().numpy()
        # Color image 3 x H x W
        # Feature map C x H x W
        feature_length, height, width = feature_map_1.shape

        # Extend 1D locations to B x C x Sampling_size
        keypoint_number = len(kps_1D_1)
        source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
            1, 1,
            keypoint_number).expand(
            -1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors = torch.gather(
            feature_map_1.view(1, feature_length, height * width), 2,
            source_feature_1d_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
                                                               keypoint_number,
                                                               1,
                                                               1).permute(0, 2, 1, 3,
                                                                          4).view(1,
                                                                                  keypoint_number,
                                                                                  feature_length,
                                                                                  1, 1)

        filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_2.view(1, feature_length, height, width),
            weight=sampled_feature_vectors.view(keypoint_number,
                                                feature_length,
                                                1, 1), padding=0)

        # Cleaning used variables to save space
        del sampled_feature_vectors
        del source_feature_1d_locations

        max_reponses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)

        del filter_response_map
        # query is 1 and train is 2 here
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_reponses.view(-1)
        # Do cross check
        feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, feature_length, -1)
        keypoint_number = keypoint_number

        # Sampled rough locator feature vectors
        sampled_feature_vectors_2 = torch.gather(
            feature_map_2.view(1, feature_length, height * width), 2,
            feature_1d_locations_2.long())
        sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
                                                                   keypoint_number,
                                                                   1,
                                                                   1).permute(0, 2, 1, 3,
                                                                              4).view(1,
                                                                                      keypoint_number,
                                                                                      feature_length,
                                                                                      1, 1)

        source_filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_1.view(1, feature_length, height, width),
            weight=sampled_feature_vectors_2.view(keypoint_number,
                                                  feature_length,
                                                  1, 1), padding=0)

        del feature_1d_locations_2
        del sampled_feature_vectors_2

        max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)

        del source_filter_response_map

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()

        # We will accept the feature matches if the max indexes here is
        # not far away from the original key point location from descriptor
        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())
        valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
                                           valid_correspondence_indexes.long())

        valid_detected_1d_locations_2 = valid_detected_1d_locations_2.data.cpu().numpy()
        valid_max_responses = valid_max_responses.data.cpu().numpy()
        valid_correspondence_indexes = valid_correspondence_indexes.data.cpu().numpy()

        detected_keypoints_2 = []
        for index in valid_detected_1d_locations_2:
            detected_keypoints_2.append(
                cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))

        matches = []
        for i, (query_index, response) in enumerate(
                zip(valid_correspondence_indexes, valid_max_responses)):
            matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i, _distance=response))

        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

        # Extract corner points
        color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
        color_2 = np.uint8(255 * (color_2 * 0.5 + 0.5))

        display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
                                             flags=2,
                                             outImg=None)

        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        feature_matches_craft = bf.knnMatch(des_1, des_2, k=1)

        good = []
        for m in feature_matches_craft:
            if len(m) != 0:
                good.append(m[0])
        display_matches_craft = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
                                                outImg=None)
        return display_matches_ai, display_matches_craft


def gather_feature_matching_data(feature_descriptor_model_path, sub_folder, data_root, image_downsampling,
                                 network_downsampling, load_intermediate_data, precompute_root,
                                 batch_size, id_range, filter_growth_rate, feature_length, gpu_id):
    feature_descriptor_model = models.FCDenseNet(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length)

    # Multi-GPU running
    feature_descriptor_model = torch.nn.DataParallel(feature_descriptor_model, device_ids=[gpu_id])
    feature_descriptor_model.eval()

    if feature_descriptor_model_path.exists():
        print("Loading {:s} ...".format(str(feature_descriptor_model_path)))
        state = torch.load(str(feature_descriptor_model_path), map_location='cuda:{}'.format(gpu_id))
        feature_descriptor_model.load_state_dict(state["model"])
    else:
        print("No pre-trained model detected")
        raise OSError
    del state

    video_frame_filenames = get_all_color_image_names_in_sequence(sub_folder)
    print("Gathering feature matching data for {}".format(str(sub_folder)))
    folder_list = get_all_subfolder_names(data_root, id_range)
    video_dataset = dataset.SfMDataset(image_file_names=video_frame_filenames,
                                       folder_list=folder_list,
                                       image_downsampling=image_downsampling,
                                       network_downsampling=network_downsampling,
                                       load_intermediate_data=load_intermediate_data,
                                       intermediate_data_root=precompute_root,
                                       phase="image_loading")
    video_loader = torch.utils.data.DataLoader(dataset=video_dataset, batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=batch_size)

    colors_list = []
    feature_maps_list = []
    with torch.no_grad():
        # Update progress bar
        tq = tqdm.tqdm(total=len(video_loader) * batch_size)
        for batch, (colors_1, boundaries, image_names,
                    folders, starts_h, starts_w) in enumerate(video_loader):
            tq.update(batch_size)
            colors_1 = colors_1.cuda(gpu_id)
            if batch == 0:
                boundary = boundaries[0].data.numpy()
                start_h = starts_h[0].item()
                start_w = starts_w[0].item()

            feature_maps_1 = feature_descriptor_model(colors_1)
            for idx in range(colors_1.shape[0]):
                colors_list.append(colors_1[idx].data.cpu().numpy())
                feature_maps_list.append(feature_maps_1[idx].data.cpu())
    tq.close()
    del feature_descriptor_model
    return colors_list, boundary, feature_maps_list, start_h, start_w


def feature_matching_single_generation(feature_map_1, feature_map_2,
                                       kps_1D_1, cross_check_distance, gpu_id):
    with torch.no_grad():
        # Feature map C x H x W
        feature_length, height, width = feature_map_1.shape

        # Extend 1D locations to B x C x Sampling_size
        keypoint_number = len(kps_1D_1)
        source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
            1, 1,
            keypoint_number).expand(
            -1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors = torch.gather(
            feature_map_1.view(1, feature_length, height * width), 2,
            source_feature_1d_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
                                                               keypoint_number,
                                                               1,
                                                               1).permute(0, 2, 1, 3,
                                                                          4).view(1,
                                                                                  keypoint_number,
                                                                                  feature_length,
                                                                                  1, 1)

        # 1 x Sampling_size x H x W
        filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_2.view(1, feature_length, height, width),
            weight=sampled_feature_vectors.view(keypoint_number,
                                                feature_length,
                                                1, 1), padding=0)

        max_reponses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)
        del sampled_feature_vectors, filter_response_map, source_feature_1d_locations
        # query is 1 and train is 2 here
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_reponses.view(-1)
        # Do cross check
        feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors_2 = torch.gather(
            feature_map_2.view(1, feature_length, height * width), 2,
            feature_1d_locations_2.long())
        sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
                                                                   keypoint_number,
                                                                   1,
                                                                   1).permute(0, 2, 1, 3,
                                                                              4).view(1,
                                                                                      keypoint_number,
                                                                                      feature_length,
                                                                                      1, 1)

        # 1 x Sampling_size x H x W
        source_filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_1.view(1, feature_length, height, width),
            weight=sampled_feature_vectors_2.view(keypoint_number,
                                                  feature_length,
                                                  1, 1), padding=0)

        max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)
        del sampled_feature_vectors_2, source_filter_response_map, feature_1d_locations_2

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()

        # We will accept the feature matches if the max indexes here is
        # not far away from the original key point location from descriptor
        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())

        valid_detected_target_2d_locations = torch.cat(
            [torch.fmod(valid_detected_1d_locations_2.float(), width).view(-1, 1),
             torch.floor(valid_detected_1d_locations_2.float() / width).view(-1, 1)],
            dim=1).view(-1, 2).float()
        valid_source_keypoint_indexes = valid_correspondence_indexes.view(-1).data.cpu().numpy()
        valid_detected_target_2d_locations = valid_detected_target_2d_locations.view(-1, 2).data.cpu().numpy()
        return valid_source_keypoint_indexes, valid_detected_target_2d_locations


def extract_keypoints(descriptor, colors_list, boundary, height, width):
    keypoints_list = []
    descriptions_list = []
    keypoints_list_1D = []
    keypoints_list_2D = []

    boundary = np.uint8(255 * boundary.reshape((height, width)))
    for i in range(len(colors_list)):
        color_1 = colors_list[i]
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_RGB2BGR)
        kps, des = descriptor.detectAndCompute(color_1, mask=boundary)
        keypoints_list.append(kps)
        descriptions_list.append(des)
        temp = np.zeros((len(kps)))
        temp_2d = np.zeros((len(kps), 2))

        for j, point in enumerate(kps):
            temp[j] = np.round(point.pt[0]) + np.round(point.pt[1]) * width
            temp_2d[j, 0] = np.round(point.pt[0])
            temp_2d[j, 1] = np.round(point.pt[1])

        keypoints_list_1D.append(temp)
        keypoints_list_2D.append(temp_2d)
    return keypoints_list, keypoints_list_1D, keypoints_list_2D, descriptions_list


def get_all_subfolder_names(root, id_range):
    folder_list = []
    for i in id_range:
        folder_list += list(root.glob('{}/*/'.format(i)))
    folder_list.sort()
    return folder_list


def get_all_color_image_names_in_sequence(sequence_root):
    view_indexes = read_selected_indexes(sequence_root / "colmap" / "0")
    filenames = []
    for index in view_indexes:
        filenames.append(sequence_root / "images" / "{:08d}.jpg".format(index))
    return filenames


def get_file_names_in_sequence(sequence_root):
    path = sequence_root / "colmap" / "0" / 'visible_view_indexes'
    if not path.exists():
        return []

    visible_view_indexes = read_visible_view_indexes(sequence_root / "colmap" / "0")
    filenames = []
    for index in visible_view_indexes:
        filenames.append(sequence_root / "images" / "{:08d}.jpg".format(index))
    return filenames


def read_color_img(image_path, start_h, end_h, start_w, end_w, downsampling_factor):
    img = cv2.imread(str(image_path))
    downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
    downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
    downsampled_img = downsampled_img.astype(np.float32)
    return downsampled_img


def save_model(model, optimizer, epoch, step, model_path, validation_loss):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'validation': validation_loss
    }, str(model_path))
    return
