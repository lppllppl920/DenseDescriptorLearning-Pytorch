'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import os
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import shutil
import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dense Descriptor Learning -- converting COLMAP format to SfMDataset format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--colmap_exe_path", type=str, required=True, help='executable path of COLMAP')
    parser.add_argument("--sequence_root", type=str, required=True, help='root of video sequence')
    parser.add_argument("--overwrite_output", action="store_true")
    args = parser.parse_args()

    colmap_exe_path = args.colmap_exe_path
    sequence_root = Path(args.sequence_root)
    overwrite_output = args.overwrite_output

    database_path = sequence_root / "database.db"
    result_root = sequence_root / "colmap"
    mask_path = sequence_root / "undistorted_mask.bmp"

    if not result_root.exists():
        print("ERROR: COLMAP sparse reconstruction does not exist")
    result_path_list = list(result_root.glob("*"))

    image_root = sequence_root / "images"
    image_path_list = list(image_root.glob("0*.jpg"))

    selected_indexes = list()
    for image_path in image_path_list:
        selected_indexes.append(int(image_path.name[:-4]))
    selected_indexes.sort()

    for result_path in result_path_list:
        if not overwrite_output:
            if len(list(result_path.glob("*"))) > 0:
                print("ERROR: output files already exist in {}".format(str(result_path)))
                continue

        os.system(
            "{} model_converter --input_path \"{}\" --output_path \"{}\" --output_type TXT".format(
                colmap_exe_path, str(result_path), str(result_path)))

        # Convert the text files to formats that can be read by the SfMDataset class
        camera_file_path = result_path / "cameras.txt"
        observation_file_path = result_path / "images.txt"
        point_cloud_file_path = result_path / "points3D.txt"

        # output file paths
        ply_file_path = result_path / "structure.ply"
        view_indexes_per_point_path = result_path / "view_indexes_per_point"
        camera_intrinsics_per_view_path = result_path / "camera_intrinsics_per_view"
        visible_view_indexes_path = result_path / "visible_view_indexes"
        selected_indexes_path = result_path / "selected_indexes"
        camera_trajectory_path = result_path / "motion.yaml"

        shutil.copy(src=str(mask_path), dst=str(result_path / mask_path.name))

        f_selected_indexes = open(str(selected_indexes), "w")
        for index in selected_indexes:
            f_selected_indexes.write("{}\n".format(index))
        f_selected_indexes.close()

        # Assuming there is only one PINHOLE camera in SfM estimates
        f_camera = open(str(camera_file_path), "r")
        for i in range(4):
            line = f_camera.readline()
            if i == 3:
                words = line.split(sep=" ")
                width = int(words[2])
                height = int(words[3])
                fx = float(words[4])
                fy = float(words[5])
                cx = float(words[6])
                cy = float(words[7])
        f_camera.close()

        # Write camera intrinsics per view (only one is enough in this case)
        f_camera_intrinsics_per_view = open(str(camera_intrinsics_per_view_path), "w")
        f_camera_intrinsics_per_view.write("{}\n{}\n{}\n{}\n".format(fx, fy, cx, cy))
        f_camera_intrinsics_per_view.close()

        f_point_cloud = open(str(point_cloud_file_path), "r")

        for i in range(3):
            _ = f_point_cloud.readline()

        points_list = []
        point_cloud_dict = dict()
        point_3d_id_list = list()
        while True:
            line = f_point_cloud.readline()
            if line is None:
                break
            words = line.split(sep=" ")
            if len(words) <= 1:
                break
            point_3d_id = words[0]
            point_3d_id_list.append(point_3d_id)
            point_cloud_dict[point_3d_id] = dict()
            point_cloud_dict[point_3d_id]["position"] = np.array([float(words[1]), float(words[2]), float(words[3])])
            point_cloud_dict[point_3d_id]["color"] = np.array([int(words[4]), int(words[5]), int(words[6])])

            # 0 is image ID, 1 is 2D point index
            temp_track = np.zeros(((len(words) - 8) // 2, 2), dtype=np.int32)
            for i in range((len(words) - 8) // 2):
                temp_track[i, 0] = int(words[2 * i + 8])
                temp_track[i, 1] = int(words[2 * i + 8 + 1])
            point_cloud_dict[point_3d_id]["track"] = temp_track

            points_list.append(
                (float(words[1]), float(words[2]), float(words[3]), int(words[4]), int(words[5]), int(words[6])))
        f_point_cloud.close()

        vertex = np.array(points_list,
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(str(ply_file_path))

        # Read images.txt
        f_observation = open(str(observation_file_path), "r")

        for i in range(4):
            _ = f_observation.readline()

        images_dict = dict()
        frame_index_dict = dict()
        visible_view_index_list = list()
        while True:
            line = f_observation.readline()
            if line is None:
                break
            # skip the points2D line
            _ = f_observation.readline()

            words = line.split(sep=" ")
            if len(words) <= 1:
                break

            # image ID as the key
            images_dict[words[0]] = dict()

            # qw, qx, qy, qz, tx, ty, tz
            images_dict[words[0]]["extrinsics"] = np.array(
                [float(words[1]), float(words[2]), float(words[3]), float(words[4]),
                 float(words[5]), float(words[6]), float(words[7])])

            frame_index = words[9]
            ind = frame_index.find(".")
            frame_index = int(frame_index[:ind])
            images_dict[words[0]]["frame_index"] = frame_index
            frame_index_dict[str(frame_index)] = int(words[0])
            visible_view_index_list.append(frame_index)

        f_view_indexes_per_point = open(str(view_indexes_per_point_path), "w")
        # Write view_indexes_per_point
        for point_3d_id in point_3d_id_list:
            image_id_2d_obs_id_pair = point_cloud_dict[point_3d_id]["track"]

            image_index_list = list()
            for i in range(image_id_2d_obs_id_pair.shape[0]):
                image_id = image_id_2d_obs_id_pair[i, 0]
                image_index_list.append(images_dict[str(image_id)]["frame_index"])

            image_index_list.sort()
            f_view_indexes_per_point.write("{}\n".format(-1))
            for frame_index in image_index_list:
                f_view_indexes_per_point.write("{}\n".format(frame_index))
        f_view_indexes_per_point.close()

        # Write visible_view_indexes
        f_visible_view_indexes = open(str(visible_view_indexes_path), "w")
        visible_view_index_list.sort()
        for image_index in visible_view_index_list:
            f_visible_view_indexes.write("{}\n".format(image_index))
        f_visible_view_indexes.close()

        # Write motion.yaml
        f_camera_trajectory = open(str(camera_trajectory_path), "w")
        camera_trajectory_dict = dict()
        camera_trajectory_dict["header"] = {"seq": 0, "stamp": 0.0, "frame_id": 0}
        camera_trajectory_dict["poses[]"] = dict()

        keys = frame_index_dict.keys()
        vals = frame_index_dict.values()

        image_ids = np.array([val for val in vals])
        frame_indexes = np.array([int(key) for key in keys])
        sorted_indexes = frame_indexes.argsort()

        sorted_frame_indexes = frame_indexes[sorted_indexes]
        sorted_image_ids = image_ids[sorted_indexes]
        pose_dict = dict()
        for i in range(sorted_frame_indexes.shape[0]):
            extrinsics = images_dict[str(sorted_image_ids[i])]["extrinsics"]
            extrinsics = extrinsics.tolist()
            position_dict = {"x": float(extrinsics[4]), "y": float(extrinsics[5]), "z": float(extrinsics[6])}
            orientation_dict = {"x": float(extrinsics[1]), "y": float(extrinsics[2]), "z": float(extrinsics[3]),
                                "w": float(extrinsics[0])}
            pose_dict["poses[{}]".format(i)] = {"position": position_dict, "orientation": orientation_dict}

        camera_trajectory_dict["poses[]"] = pose_dict
        yaml.dump(camera_trajectory_dict, f_camera_trajectory)
        f_camera_trajectory.close()
