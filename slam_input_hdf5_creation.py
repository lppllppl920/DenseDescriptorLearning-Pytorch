import os
from pathlib import Path
import numpy as np
import shutil
import yaml
import argparse
import cv2
import h5py
import tqdm
import torch

import dataset
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dense SLAM -- create hdf5 file used for slam system input',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True, help='root of data')
    parser.add_argument("--sequence_root", type=str, default=None, help='root of video sequence')
    parser.add_argument('--load_intermediate_data', action='store_true', help='whether to load intermediate data')
    parser.add_argument('--image_downsampling', type=float, default=4.0,
                        help='input image downsampling rate')
    parser.add_argument('--network_downsampling', type=int, default=64, help='network bottom layer downsampling')
    parser.add_argument("--overwrite_hdf5", action="store_true")
    args = parser.parse_args()

    if args.sequence_root is not None:
        sequence_root_list = [Path(args.sequence_root)]
    else:
        sequence_root_list = sorted(list(Path(args.data_root).rglob("_start*")))

    for sequence_root in sequence_root_list:
        print(f"Processing {str(sequence_root)} for hdf5 creation")
        hdf5_path = sequence_root / "slam_input_data.hdf5"

        if args.overwrite_hdf5 and hdf5_path.exists():
            os.remove(str(hdf5_path))
        elif not args.overwrite_hdf5 and hdf5_path.exists():
            continue

        hf = h5py.File(str(hdf5_path), 'w')

        colmap_result_root = sequence_root / "colmap" / "0"

        # Read mask
        original_mask = cv2.imread(str(colmap_result_root / "sparse" / "mask.bmp"))
        ori_height, ori_width = original_mask.shape[:2]
        # Read camera intrinsics
        original_camera_intrinsics = np.zeros((3, 3), dtype=np.float32)
        original_camera_intrinsics[2][2] = 1.0
        with open(str(colmap_result_root / "sparse" / "undistorted_camera_intrinsics")) as fp:
            for i in range(4):
                line = fp.readline()
                if i == 0:
                    original_camera_intrinsics[0][0] = float(line)
                if i == 1:
                    original_camera_intrinsics[1][1] = float(line)
                if i == 2:
                    original_camera_intrinsics[0][2] = float(line)
                if i == 3:
                    original_camera_intrinsics[1][2] = float(line)

        # Read camera poses
        camera_pose_list = list()
        stream = open(str(colmap_result_root / "sparse" / "motion.yaml"), 'r')
        doc = yaml.load(stream)
        keys, values = doc.items()
        poses = values[1]

        # Read indexes of visible views
        visible_view_indexes = []
        with open(str(colmap_result_root / "sparse" / 'visible_view_indexes')) as fp:
            for line in fp:
                visible_view_indexes.append(int(line))

        camera_pose_list = []
        for i in range(len(visible_view_indexes)):
            rigid_transform = utils.quaternion_matrix(
                [poses["poses[" + str(i) + "]"]['orientation']['w'],
                 poses["poses[" + str(i) + "]"]['orientation']['x'],
                 poses["poses[" + str(i) + "]"]['orientation']['y'],
                 poses["poses[" + str(i) + "]"]['orientation']['z']])
            rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
            rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
            rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']
            transform = np.asmatrix(rigid_transform)
            camera_pose_list.append(transform)

        images_root = colmap_result_root / "images"
        image_path_list = sorted(list(images_root.glob("*.png")))
        assert (len(image_path_list) == len(visible_view_indexes))

        video_dataset = dataset.SfMDataset(image_file_names=image_path_list,
                                           folder_list=[sequence_root],
                                           image_downsampling=args.image_downsampling,
                                           network_downsampling=args.network_downsampling,
                                           load_intermediate_data=args.load_intermediate_data,
                                           intermediate_data_root=sequence_root,
                                           phase="image_loading")
        video_loader = torch.utils.data.DataLoader(dataset=video_dataset, batch_size=1,
                                                   shuffle=False,
                                                   num_workers=0)

        mask = None
        camera_intrinsics = None
        colors_list = []
        with torch.no_grad():
            for batch, (colors_1, boundaries, image_names,
                        folders, starts_h, starts_w) in enumerate(video_loader):
                color = colors_1[0].permute(1, 2, 0).data.numpy()
                color = np.asarray((color + 1.0) * 0.5 * 255, dtype=np.uint8)
                # cv2.imshow("color", color)
                # cv2.waitKey()
                colors_list.append(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                if mask is None:
                    mask = np.asarray(boundaries[0].permute(1, 2, 0).data.numpy(), dtype=np.uint8)
                    # cv2.imshow("mask", mask)
                    # cv2.waitKey()

                if camera_intrinsics is None:
                    camera_intrinsics = np.copy(original_camera_intrinsics)
                    camera_intrinsics[0][0] = original_camera_intrinsics[0][0] / args.image_downsampling
                    camera_intrinsics[1][1] = original_camera_intrinsics[1][1] / args.image_downsampling
                    camera_intrinsics[0][2] = original_camera_intrinsics[0][2] / args.image_downsampling - starts_w[0]
                    camera_intrinsics[1][2] = original_camera_intrinsics[1][2] / args.image_downsampling - starts_h[0]

        height, width = colors_list[0].shape[:2]
        dataset_extrinsics = hf.create_dataset('extrinsics', (0, 4, 4),
                                               maxshape=(None, 4, 4), chunks=(4096, 4, 4),
                                               compression="gzip", compression_opts=4, dtype='float32')
        dataset_intrinsics = hf.create_dataset('intrinsics', (0, 3, 3),
                                               maxshape=(None, 3, 3), chunks=(4096, 3, 3),
                                               compression="gzip", compression_opts=4, dtype='float32')
        dataset_color = hf.create_dataset('color', (0, height, width, 3),
                                          maxshape=(None, height, width, 3), chunks=(1, height, width, 3),
                                          compression="gzip", compression_opts=9, dtype='uint8')
        dataset_mask = hf.create_dataset('mask', (0, height, width, 1),
                                         maxshape=(None, height, width, 1), chunks=(1, height, width, 1),
                                         compression="gzip", compression_opts=9, dtype='uint8')
        dataset_frame_index = hf.create_dataset('frame_index', (0, 1),
                                                maxshape=(None, 1), chunks=(40960, 1),
                                                compression="gzip", compression_opts=4, dtype='int32')

        dataset_extrinsics.resize((len(camera_pose_list), 4, 4))
        dataset_extrinsics[:, :, :] = np.asarray(camera_pose_list).reshape((len(camera_pose_list), 4, 4))

        dataset_intrinsics.resize((1, 3, 3))
        dataset_intrinsics[0, :, :] = camera_intrinsics

        dataset_color.resize((len(image_path_list), height, width, 3))
        dataset_color[:, :, :, :] = np.asarray(colors_list).reshape((len(camera_pose_list), height, width, 3))

        dataset_mask.resize((1, height, width, 1))
        dataset_mask[0, :, :, :] = mask.reshape((height, width, 1))

        dataset_frame_index.resize((len(visible_view_indexes), 1))
        dataset_frame_index[:, :] = np.asarray(visible_view_indexes).reshape((-1, 1))

        hf.close()
