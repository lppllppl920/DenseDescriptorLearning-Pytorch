'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import matplotlib

matplotlib.use('agg')
import cv2
import yaml
import numpy as np
from plyfile import PlyData
from pathlib import Path
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense Descriptor Learning -- point cloud - video overlay generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sequence_root", type=str, required=True, help='root of video sequence')
    parser.add_argument("--display_image", action="store_true")
    parser.add_argument("--display_visible_only", action="store_true")
    parser.add_argument("--write_image", action="store_true")
    parser.add_argument("--write_video", action="store_true")

    args = parser.parse_args()

    display_image = args.display_image
    display_visible_only = args.display_visible_only
    write_image = args.write_image
    write_video = args.write_video
    sequence_root = Path(args.sequence_root)

    path_list = list(sequence_root.glob("colmap/*"))
    path_list.sort()
    num_points_per_seq = []
    num_points_per_img = []
    for prefix_seq in path_list:
        print("Processing {}...".format(str(prefix_seq)))

        if (prefix_seq / "point_cloud_overlay.avi").exists():
            continue
        # Read sparse point cloud from SfM
        if not (prefix_seq / "structure.ply").exists():
            continue
        lists_3D_points = []
        plydata = PlyData.read(str(prefix_seq / "structure.ply"))
        for i in range(plydata['vertex'].count):
            temp = list(plydata['vertex'][i])
            temp = temp[:3]
            temp.append(1.0)
            lists_3D_points.append(temp)

        lists_colors = [[255, 0, 0] for i in range(len(lists_3D_points))]

        # Read camera poses from SfM
        stream = open(str(prefix_seq / "motion.yaml"), 'r')
        doc = yaml.load(stream)
        keys, values = doc.items()
        poses = values[1]

        # Read indexes of visible views
        visible_view_indexes = []
        with open(str(prefix_seq / 'visible_view_indexes')) as fp:
            for line in fp:
                visible_view_indexes.append(int(line))

        # Read view indexes per point
        view_indexes_per_point = np.zeros((plydata['vertex'].count, len(visible_view_indexes)))
        point_count = -1
        with open(str(prefix_seq / 'view_indexes_per_point')) as fp:
            for line in fp:
                if int(line) == -1:
                    point_count = point_count + 1
                else:
                    view_indexes_per_point[point_count][visible_view_indexes.index(int(line))] = 1

        view_indexes_per_point = utils.overlapping_visible_view_indexes_per_point(
            view_indexes_per_point, 1)

        # Read camera intrinsics used by SfM
        camera_intrinsics = []
        param_count = 0
        temp_camera_intrincis = np.zeros((3, 4))
        with open(str(prefix_seq / 'camera_intrinsics_per_view')) as fp:
            for line in fp:
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

        # Generating projection and extrinsic matrices
        projection_matrices = []
        extrinsic_matrices = []
        projection_matrix = np.zeros((3, 4))
        for i in range(len(visible_view_indexes)):
            rigid_transform = utils.quaternion_matrix(
                [poses["poses[" + str(i) + "]"]['orientation']['w'], poses["poses[" + str(i) + "]"]['orientation']['x'],
                 poses["poses[" + str(i) + "]"]['orientation']['y'],
                 poses["poses[" + str(i) + "]"]['orientation']['z']])
            rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
            rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
            rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']

            transform = np.asmatrix(rigid_transform)
            extrinsic_matrices.append(transform)

            projection_matrix = np.dot(camera_intrinsics[0], transform)
            projection_matrices.append(projection_matrix)

        array_3D_points = np.asarray(lists_3D_points).reshape((-1, 4))
        # Read mask image
        img_mask = cv2.imread(str(prefix_seq / 'undistorted_mask.bmp'), cv2.IMREAD_GRAYSCALE)
        img_mask = img_mask.reshape((-1, 1))
        overlay_image_list = []

        view_indexes_per_point = np.moveaxis(view_indexes_per_point, source=[0, 1], destination=[1, 0])
        # Drawing 2D overlay of sparse point cloud onto every image plane
        for i in range(len(visible_view_indexes)):
            print("Process {}...".format(i))
            img = cv2.imread(str(prefix_seq.parents[1] / ("{:08d}.jpg".format(visible_view_indexes[i]))))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]

            projection_matrix = projection_matrices[i]
            extrinsic_matrix = extrinsic_matrices[i]

            points_3D_camera = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera = points_3D_camera / points_3D_camera[:, 3].reshape((-1, 1))

            points_2D_image = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image = points_2D_image / points_2D_image[:, 2].reshape((-1, 1))

            view_indexes_frame = np.asarray(view_indexes_per_point[i, :]).reshape((-1))
            visible_point_indexes = np.where(view_indexes_frame > 0.5)
            invisible_point_indexes = np.where(view_indexes_frame <= 0.5)
            visible_point_indexes = visible_point_indexes[0]
            invisible_point_indexes = invisible_point_indexes[0]
            visible_points_2D_image = points_2D_image[visible_point_indexes, :].reshape((-1, 3))
            invisible_points_2D_image = points_2D_image[invisible_point_indexes, :].reshape((-1, 3))
            visible_points_3D_camera = points_3D_camera[visible_point_indexes, :].reshape((-1, 4))
            invisible_points_3D_camera = points_3D_camera[invisible_point_indexes, :].reshape((-1, 4))

            indexes = np.where((visible_points_2D_image[:, 0] <= width - 1) & (visible_points_2D_image[:, 0] >= 0) &
                               (visible_points_2D_image[:, 1] <= height - 1) & (visible_points_2D_image[:, 1] >= 0) &
                               (visible_points_3D_camera[:, 2] >= 0))
            indexes = indexes[0]

            in_image_point_1D_locations = (np.round(visible_points_2D_image[indexes, 0]) +
                                           np.round(visible_points_2D_image[indexes, 1]) * width).astype(
                np.int32).reshape((-1))
            temp_mask = img_mask[in_image_point_1D_locations, :]
            indexes_2 = np.where(temp_mask[:, 0] == 255)
            indexes_2 = indexes_2[0]

            visible_in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2]

            indexes = np.where((invisible_points_2D_image[:, 0] <= width - 1) & (invisible_points_2D_image[:, 0] >= 0) &
                               (invisible_points_2D_image[:, 1] <= height - 1) & (invisible_points_2D_image[:, 1] >= 0)
                               & (invisible_points_3D_camera[:, 2] > 0))
            indexes = indexes[0]
            in_image_point_1D_locations = (np.round(invisible_points_2D_image[indexes, 0]) +
                                           np.round(invisible_points_2D_image[indexes, 1]) * width).astype(
                np.int32).reshape((-1))
            temp_mask = img_mask[in_image_point_1D_locations, :]
            indexes_2 = np.where(temp_mask[:, 0] == 255)
            indexes_2 = indexes_2[0]

            invisible_in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2]

            visible_locations_y = list(visible_in_mask_point_1D_locations / width)
            visible_locations_x = list(visible_in_mask_point_1D_locations % width)

            invisible_locations_y = list(invisible_in_mask_point_1D_locations / width)
            invisible_locations_x = list(invisible_in_mask_point_1D_locations % width)

            img = utils.scatter_points_to_image(img, visible_locations_x=visible_locations_x,
                                                visible_locations_y=visible_locations_y,
                                                invisible_locations_x=invisible_locations_x,
                                                invisible_locations_y=invisible_locations_y,
                                                only_visible=display_visible_only,
                                                point_size=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if write_video:
                overlay_image_list.append(img)
            if display_image:
                cv2.imshow("projected spatial points", img)
                cv2.waitKey(10)
        if write_video:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            size = (overlay_image_list[0].shape[1], overlay_image_list[0].shape[0])
            out = cv2.VideoWriter(str(prefix_seq / "point_cloud_overlay.avi"), fourcc, fps=20.0,
                                  frameSize=size)

            for image in overlay_image_list:
                out.write(image)
            out.release()
