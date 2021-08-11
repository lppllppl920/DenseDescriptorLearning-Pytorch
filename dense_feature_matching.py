'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import cv2
import numpy as np
from pathlib import Path
import argparse
import h5py
import multiprocessing
import tqdm
import torch
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
# Local
import utils
import models

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(
        description='Dense Descriptor Learning -- dense feature matching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_downsampling', type=float, default=4.0,
                        help='input image downsampling rate')
    parser.add_argument('--network_downsampling', type=int, default=64, help='network bottom layer downsampling')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for input data loader')
    parser.add_argument('--load_intermediate_data', action='store_true', help='whether to load intermediate data')
    parser.add_argument('--overwrite_matches', action='store_true', help='whether to overwrite matches')
    parser.add_argument('--data_root', type=str, required=True, help='path to the data for '
                                                                     'feature matching')
    parser.add_argument("--sequence_root", type=str, default=None, help='root of video sequence')
    parser.add_argument('--trained_model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--feature_length', type=int, default=256, help='output channel dimension of network')
    parser.add_argument('--filter_growth_rate', type=int, default=10, help='filter growth rate of network')
    parser.add_argument('--max_feature_detection', type=int, default=3000,
                        help='max allowed number of detected features per frame')
    parser.add_argument('--cross_check_distance', type=float, default=5.0,
                        help='max cross check distance for valid matches')
    parser.add_argument('--id_range', nargs='+', type=int,
                        help='range of patient ids')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id for matching generation')
    parser.add_argument('--temporal_range', type=int, default=30, help='range for temporal sampling')
    parser.add_argument('--test_keypoint_num', type=int, default=200, help='number of keypoints used for quick '
                                                                           'spatial testing')
    parser.add_argument('--residual_threshold', type=float, default=5.0, help='pixel threshold for ransac estimation')
    parser.add_argument('--octave_layers', type=int, default=8)
    parser.add_argument('--contrast_threshold', type=float, default=0.00005)
    parser.add_argument('--edge_threshold', type=float, default=100)
    parser.add_argument('--sigma', type=float, default=1.1)
    parser.add_argument('--skip_interval', type=int, default=5,
                        help="number of skipping frames in searching state")
    parser.add_argument('--min_inlier_ratio', type=float, default=0.2,
                        help="minimum inlier ratio of ransac")
    parser.add_argument('--hysterisis_factor', type=float, default=0.7,
                        help="factor of the inlier ratio in the spatial_range state")
    args = parser.parse_args()

    # Hyper-parameters
    id_range = args.id_range
    image_downsampling = args.image_downsampling
    batch_size = args.batch_size
    num_workers = args.num_workers
    network_downsampling = args.network_downsampling
    load_intermediate_data = args.load_intermediate_data
    overwrite_matches = args.overwrite_matches

    if args.sequence_root is not None:
        sequence_root_list = [Path(args.sequence_root)]
    else:
        sequence_root_list = sorted(list(Path(args.data_root).rglob("_start*")))

    trained_model_path = Path(args.trained_model_path)
    data_root = Path(args.data_root)
    max_feature_detection = args.max_feature_detection
    cross_check_distance = args.cross_check_distance
    gpu_id = args.gpu_id
    temporal_range = args.temporal_range
    test_keypoint_num = args.test_keypoint_num
    residual_threshold = args.residual_threshold
    feature_length = args.feature_length
    filter_growth_rate = args.filter_growth_rate
    octave_layers = args.octave_layers
    contrast_threshold = args.contrast_threshold
    edge_threshold = args.edge_threshold
    sigma = args.sigma
    skip_interval = args.skip_interval
    min_inlier_ratio = args.min_inlier_ratio
    hysterisis_factor = args.hysterisis_factor

    precompute_root = data_root / "precompute"
    if not precompute_root.exists():
        precompute_root.mkdir(parents=True)
    print("SIFT detector creating...")
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=max_feature_detection, nOctaveLayers=octave_layers,
                                       contrastThreshold=contrast_threshold,
                                       edgeThreshold=edge_threshold, sigma=sigma)

    feature_descriptor_model = models.FCDenseNet(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length)

    # Multi-GPU running
    feature_descriptor_model = torch.nn.DataParallel(feature_descriptor_model, device_ids=[gpu_id])
    feature_descriptor_model.eval()

    if trained_model_path.exists():
        print("Loading {:s} ...".format(str(trained_model_path)))
        state = torch.load(str(trained_model_path), map_location='cuda:{}'.format(gpu_id))
        feature_descriptor_model.load_state_dict(state["model"])
    else:
        print("No pre-trained model detected")
        raise OSError
    del state

    for sequence_root in sequence_root_list:
        print(f"Processing {str(sequence_root)} for dense feature matching")

        if not overwrite_matches and (sequence_root / "feature_matches.hdf5").exists():
            continue

        colors_list, boundary, start_h, start_w = \
            utils.gather_color_data(sub_folder=sequence_root,
                                    data_root=data_root, image_downsampling=image_downsampling,
                                    network_downsampling=network_downsampling,
                                    load_intermediate_data=load_intermediate_data,
                                    precompute_root=precompute_root,
                                    batch_size=batch_size, id_range=id_range, gpu_id=gpu_id)

        # Erode the boundary to remove near-boundary matches
        kernel = np.ones((5, 5), np.uint8)
        boundary = cv2.erode(boundary, kernel, iterations=3)

        print("Extracting keypoint locations...")
        _, height, width = colors_list[0].shape
        sift_keypoints_list, sift_keypoint_location_list_1D, sift_keypoint_location_list_2D, sift_descriptions_list = \
            utils.extract_keypoints(sift, colors_list, boundary, height, width)

        f_matches = h5py.File(str(sequence_root / "feature_matches.hdf5"), 'w')
        dataset_matches = f_matches.create_dataset('matches', (0, 4, 1),
                                                   maxshape=(None, 4, 1), chunks=(40960, 4, 1),
                                                   compression="gzip", compression_opts=9, dtype='int16')

        frame_count_in_total = len(colors_list)

        mean_inlier_ratio = None
        tq = tqdm.tqdm(total=frame_count_in_total * (frame_count_in_total - 1) // 2)
        for i in range(frame_count_in_total - 1):
            color_1 = colors_list[i]
            feature_map_1 = feature_descriptor_model(color_1.reshape(1, *color_1.shape))
            feature_map_1 = feature_map_1.reshape(*feature_map_1.shape[1:])
            # feature_map_1 = torch.from_numpy(dataset_feature_map[i]).cuda(gpu_id)

            sift_keypoint_1 = sift_keypoints_list[i]
            np.random.seed(10086)
            random_indexes = list(np.random.choice(range(0, len(sift_keypoint_1)), test_keypoint_num,
                                                   replace=False))
            sift_keypoint_locations_1D_1 = sift_keypoint_location_list_1D[i]
            sift_keypoint_locations_2D_1 = sift_keypoint_location_list_2D[i]
            cur_state = "temporal_range"
            for j in range(1, len(colors_list) - i):
                tq.set_description(cur_state)

                if cur_state == "temporal_range":
                    if j > temporal_range:
                        cur_state = "searching"

                if cur_state == "searching":
                    if j % skip_interval != 0:
                        tq.update(1)
                        continue

                if cur_state == "spatial_range":
                    pass

                color_2 = colors_list[i + j]
                # feature_map_2 = torch.from_numpy(dataset_feature_map[i + j]).cuda(gpu_id)
                feature_map_2 = feature_descriptor_model(color_2.reshape(1, *color_2.shape))
                feature_map_2 = feature_map_2.reshape(*feature_map_2.shape[1:])

                if cur_state == "temporal_range" or cur_state == "spatial_range":
                    x = utils.feature_matching_single_generation(
                        feature_map_1=feature_map_1,
                        feature_map_2=feature_map_2,
                        kps_1D_1=sift_keypoint_locations_1D_1,
                        cross_check_distance=cross_check_distance,
                        gpu_id=gpu_id)
                elif cur_state == "searching":
                    x = utils.feature_matching_single_generation(
                        feature_map_1=feature_map_1,
                        feature_map_2=feature_map_2,
                        kps_1D_1=sift_keypoint_locations_1D_1[random_indexes],
                        cross_check_distance=cross_check_distance,
                        gpu_id=gpu_id)
                else:
                    x = None

                if x is None:
                    tq.update(1)
                    continue

                source_keypoint_indexes, target_keypoint_locations = x

                if cur_state == "searching":
                    source_keypoint_indexes = [random_indexes[source_keypoint_index] for
                                               source_keypoint_index in source_keypoint_indexes]

                source_keypoint_locations = sift_keypoint_locations_2D_1[source_keypoint_indexes, :].reshape((-1, 2))

                #  Only keep the locations that are within the original unpadded image and the locations
                #  below should be the one within the original image instead of the padded one
                source_keypoint_locations[:, 0] = image_downsampling * (
                        source_keypoint_locations[:, 0] + start_w)
                source_keypoint_locations[:, 1] = image_downsampling * (
                        source_keypoint_locations[:, 1] + start_h)

                target_keypoint_locations[:, 0] = image_downsampling * (
                        target_keypoint_locations[:, 0] + start_w)
                target_keypoint_locations[:, 1] = image_downsampling * (
                        target_keypoint_locations[:, 1] + start_h)

                try:
                    model, inliers = ransac((source_keypoint_locations,
                                             target_keypoint_locations),
                                            FundamentalMatrixTransform, min_samples=8,
                                            residual_threshold=residual_threshold, max_trials=5)
                except ValueError:
                    tq.set_postfix(
                        source_frame_index='{:d}'.format(i),
                        target_frame_index='{:d}'.format(i + j),
                        point_num='{:d}'.format(target_keypoint_locations.shape[0]))
                    tq.update(1)
                    continue

                inlier_ratio = np.sum(inliers) / source_keypoint_locations.shape[0]

                # if j == 1:
                #     mean_inlier_ratio = inlier_ratio
                # elif j <= temporal_range:
                #     mean_inlier_ratio = mean_inlier_ratio * ((j - 1) / j) + inlier_ratio * (1 / j)
                # elif j == temporal_range + 1:
                #     mean_inlier_ratio = max(min_inlier_ratio, mean_inlier_ratio)

                if cur_state == "temporal_range":
                    if inlier_ratio >= min_inlier_ratio:
                        start_index = dataset_matches.shape[0]
                        dataset_matches.resize(
                            (dataset_matches.shape[0] + target_keypoint_locations.shape[0] + 1, 4, 1))
                        dataset_matches[start_index, :, :] = np.asarray(
                            [target_keypoint_locations.shape[0], i, i + j, -1]).reshape((4, 1))

                        dataset_matches[start_index + 1:start_index + 1 + target_keypoint_locations.shape[0],
                        :] = \
                            np.concatenate([source_keypoint_locations.reshape((-1, 2)),
                                            target_keypoint_locations.reshape((-1, 2))], axis=1).reshape(
                                (-1, 4, 1)).astype(np.int16)
                        tq.set_description(cur_state)
                        tq.set_postfix(
                            source_frame_index='{:d}'.format(i),
                            target_frame_index='{:d}'.format(i + j),
                            inlier_ratio='{:.3f}'.format(inlier_ratio))
                    else:
                        tq.set_postfix(
                            source_frame_index='{:d}'.format(i),
                            target_frame_index='{:d}'.format(i + j))

                elif cur_state == "searching":
                    if inlier_ratio >= min_inlier_ratio:
                        cur_state = "spatial_range"
                        # Redo the feature matching with full set of keypoints
                        x = utils.feature_matching_single_generation(
                            feature_map_1=feature_map_1,
                            feature_map_2=feature_map_2,
                            kps_1D_1=sift_keypoint_locations_1D_1,
                            cross_check_distance=cross_check_distance,
                            gpu_id=gpu_id)

                        if x is None:
                            tq.update(1)
                            continue

                        source_keypoint_indexes, target_keypoint_locations = x
                        source_keypoint_locations = sift_keypoint_locations_2D_1[source_keypoint_indexes,
                                                    :].reshape((-1, 2))
                        source_keypoint_locations[:, 0] = image_downsampling * (
                                source_keypoint_locations[:, 0] + start_w)
                        source_keypoint_locations[:, 1] = image_downsampling * (
                                source_keypoint_locations[:, 1] + start_h)

                        target_keypoint_locations[:, 0] = image_downsampling * (
                                target_keypoint_locations[:, 0] + start_w)
                        target_keypoint_locations[:, 1] = image_downsampling * (
                                target_keypoint_locations[:, 1] + start_h)

                        start_index = dataset_matches.shape[0]
                        dataset_matches.resize(
                            (dataset_matches.shape[0] + target_keypoint_locations.shape[0] + 1, 4, 1))
                        dataset_matches[start_index, :, :] = np.asarray(
                            [target_keypoint_locations.shape[0], i, i + j, -1]).reshape((4, 1))

                        dataset_matches[start_index + 1:start_index + 1 + target_keypoint_locations.shape[0],
                        :] = \
                            np.concatenate([source_keypoint_locations.reshape((-1, 2)),
                                            target_keypoint_locations.reshape((-1, 2))], axis=1).reshape(
                                (-1, 4, 1)).astype(np.int16)
                        tq.set_description(cur_state)
                        tq.set_postfix(
                            source_frame_index='{:d}'.format(i),
                            target_frame_index='{:d}'.format(i + j),
                            inlier_ratio='{:.3f}'.format(inlier_ratio))
                    else:
                        tq.set_description(cur_state)
                        tq.set_postfix(
                            source_frame_index='{:d}'.format(i),
                            target_frame_index='{:d}'.format(i + j))
                elif cur_state == "spatial_range":
                    # Leave a bit of hyterisis space for spatial_range state to allow for more frame matches
                    if inlier_ratio >= hysterisis_factor * min_inlier_ratio:
                        start_index = dataset_matches.shape[0]
                        dataset_matches.resize(
                            (dataset_matches.shape[0] + target_keypoint_locations.shape[0] + 1, 4, 1))
                        dataset_matches[start_index, :, :] = np.asarray(
                            [target_keypoint_locations.shape[0], i, i + j, -1]).reshape((4, 1))

                        dataset_matches[start_index + 1:start_index + 1 + target_keypoint_locations.shape[0],
                        :] = \
                            np.concatenate([source_keypoint_locations.reshape((-1, 2)),
                                            target_keypoint_locations.reshape((-1, 2))], axis=1).reshape(
                                (-1, 4, 1)).astype(np.int16)
                        tq.set_description(cur_state)
                        tq.set_postfix(
                            source_frame_index='{:d}'.format(i),
                            target_frame_index='{:d}'.format(i + j),
                            inlier_ratio='{:.3f}'.format(inlier_ratio))
                    else:
                        cur_state = "searching"
                        tq.set_description(cur_state)
                        tq.set_postfix(
                            source_frame_index='{:d}'.format(i),
                            target_frame_index='{:d}'.format(i + j))
                        tq.update(1)
                        continue

                tq.update(1)

        tq.close()
        if f_matches is not None:
            f_matches.close()
