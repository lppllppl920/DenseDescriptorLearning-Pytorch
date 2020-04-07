'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import tqdm
import numpy as np
from pathlib import Path
import torchsummary
import torch
import random
from tensorboardX import SummaryWriter
import argparse
import datetime
import multiprocessing

# Local
import models
import losses
import utils
import dataset

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(
        description='Dense Descriptor Learning -- pair-wise feature matching evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--adjacent_range', nargs='+', type=int, required=True,
                        help='interval range for a pair of video frames')
    parser.add_argument('--image_downsampling', type=float, default=4.0,
                        help='input image downsampling rate')
    parser.add_argument('--network_downsampling', type=int, default=64,
                        help='network downsampling rate')
    parser.add_argument('--input_size', nargs='+', type=int, required=True,
                        help='input size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
    parser.add_argument('--num_pre_workers', type=int, default=8,
                        help='number of pre-processing workers for data loader')
    parser.add_argument('--inlier_percentage', type=float, default=0.998,
                        help='percentage of inliers of SfM point clouds (for pruning extreme outliers)')
    parser.add_argument('--testing_patient_id', nargs='+', type=int, required=True, help='id of the testing patient')
    parser.add_argument('--load_intermediate_data', action='store_true',
                        help='whether or not to load intermediate data')
    parser.add_argument('--visibility_overlap', type=int, default=20, help='overlap of point visibility information')
    parser.add_argument('--display_architecture', action='store_true', help='display the network architecture')
    parser.add_argument('--trained_model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--testing_data_root', type=str, required=True, help='path to the sfm testing data')
    parser.add_argument('--log_root', type=str, required=True, help='root of logging')
    parser.add_argument('--feature_length', type=int, default=256, help='output channel dimension of network')
    parser.add_argument('--filter_growth_rate', type=int, default=10, help='filter growth rate of network')
    parser.add_argument('--keypoints_per_iter', type=int, default=200, help='number of keypoints per iteration')
    parser.add_argument('--gpu_id', type=int, default=0, help='id of selected GPU')
    args = parser.parse_args()

    trained_model_path = Path(args.trained_model_path)
    log_root = Path(args.log_root)
    adjacent_range = args.adjacent_range
    image_downsampling = args.image_downsampling
    height, width = args.input_size
    num_workers = args.num_workers
    num_pre_workers = args.num_pre_workers
    inlier_percentage = args.inlier_percentage
    testing_patient_id = args.testing_patient_id
    load_intermediate_data = args.load_intermediate_data
    display_architecture = args.display_architecture
    testing_data_root = Path(args.testing_data_root)
    feature_length = args.feature_length
    filter_growth_rate = args.filter_growth_rate
    network_downsampling = args.network_downsampling
    visibility_overlap = args.visibility_overlap
    keypoints_per_iter = args.keypoints_per_iter
    gpu_id = args.gpu_id
    current_date = datetime.datetime.now()

    if not log_root.exists():
        log_root.mkdir()
    log_root = log_root / "dense_descriptor_test_{}_{}_{}_{}".format(current_date.month, current_date.day,
                                                                     current_date.hour,
                                                                     current_date.minute)
    writer = SummaryWriter(logdir=str(log_root))
    print("Created tensorboard visualization at {}".format(str(log_root)))

    # Fix randomness for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(10086)
    np.random.seed(10086)
    random.seed(10086)

    precompute_root = testing_data_root / "precompute"
    try:
        precompute_root.mkdir(mode=0o777, parents=True)
    except OSError:
        pass

    feature_descriptor_model = models.FCDenseNet(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length)
    # Initialize the network with Kaiming He initialization
    utils.init_net(feature_descriptor_model, type="kaiming", mode="fan_in", activation_mode="relu",
                   distribution="normal")
    # Multi-GPU running
    feature_descriptor_model = torch.nn.DataParallel(feature_descriptor_model)

    # Custom layer
    response_map_generator = models.FeatureResponseGeneratorNoSoftThresholding()
    # Evaluation metric
    matching_accuracy_metric = losses.MatchingAccuracyMetric(threshold=3)

    if trained_model_path.exists():
        print("Loading {:s} ...".format(str(trained_model_path)))
        pre_trained_state = torch.load(str(trained_model_path))
        step = pre_trained_state['step']
        epoch = pre_trained_state['epoch']
        model_state = feature_descriptor_model.state_dict()
        trained_model_state = {k: v for k, v in pre_trained_state["model"].items() if k in model_state}
        model_state.update(trained_model_state)
        feature_descriptor_model.load_state_dict(model_state)
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No trained model detected")
        raise OSError

    # Validation
    feature_descriptor_model.eval()
    feature_descriptor_model = feature_descriptor_model.module
    feature_descriptor_model = feature_descriptor_model.cuda(gpu_id)
    # Summary network architecture
    if display_architecture:
        torchsummary.summary(feature_descriptor_model, input_size=(3, height, width))

    total_query = 0
    folder_list = utils.get_parent_folder_names(testing_data_root, id_range=testing_patient_id)

    mean_accuracy_1 = None
    mean_accuracy_2 = None
    mean_accuracy_3 = None
    for patient_id in testing_patient_id:
        data_root = Path(testing_data_root) / "{:d}".format(patient_id)
        sub_folders = list(data_root.glob("*/"))
        sub_folders.sort()
        for folder in sub_folders:
            # Get color image filenames
            test_filenames = utils.get_file_names_in_sequence(sequence_root=folder)
            if len(test_filenames) == 0:
                print("Sequence {} does not have relevant files".format(str(folder)))
                continue

            test_dataset = dataset.SfMDataset(image_file_names=test_filenames,
                                              folder_list=folder_list,
                                              adjacent_range=adjacent_range,
                                              image_downsampling=image_downsampling,
                                              inlier_percentage=inlier_percentage,
                                              network_downsampling=network_downsampling,
                                              load_intermediate_data=load_intermediate_data,
                                              intermediate_data_root=precompute_root,
                                              phase="test",
                                              pre_workers=num_pre_workers,
                                              visible_interval=visibility_overlap)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                                      num_workers=num_workers)

            # Update progress bar
            tq = tqdm.tqdm(total=len(test_loader), dynamic_ncols=True, ncols=40)
            tq.set_description('Test')
            with torch.no_grad():
                for batch, (colors, feature_matches, boundaries) in enumerate(test_loader):
                    colors = colors[0]
                    _, _, height, width = boundaries.shape
                    boundaries = boundaries.cuda(gpu_id)
                    feature_maps_1 = feature_descriptor_model(colors[0].cuda(gpu_id).unsqueeze(dim=0))
                    for i in range(1, colors.shape[0]):
                        feature_maps_2 = feature_descriptor_model(colors[i].cuda(gpu_id).unsqueeze(dim=0))
                        feature_1D_locations_1 = torch.round(feature_matches[i - 1][0, :, 0]) + \
                                                 torch.round(feature_matches[i - 1][0, :, 1]) * width
                        feature_1D_locations_1 = feature_1D_locations_1.cuda(gpu_id).view(1, -1, 1)
                        feature_2D_locations_2 = feature_matches[i - 1][0, :, 2:4].cuda(gpu_id).view(1, -1, 2)
                        query_size = feature_1D_locations_1.shape[1]
                        if query_size == 0:
                            continue

                        batch_num = (query_size - 1) // keypoints_per_iter + 1
                        accuracy_1 = 0
                        accuracy_2 = 0
                        accuracy_3 = 0
                        for k in range(batch_num):
                            sub_1D_locations_1 = feature_1D_locations_1[:,
                                                 keypoints_per_iter * k: min(query_size,
                                                                             keypoints_per_iter * (
                                                                                     k + 1))]
                            sub_2D_locations_2 = feature_2D_locations_2[:,
                                                 keypoints_per_iter * k: min(query_size,
                                                                             keypoints_per_iter * (
                                                                                     k + 1))]
                            response_map_2 = response_map_generator(
                                [feature_maps_1, feature_maps_2, sub_1D_locations_1, boundaries])
                            ratio_1, ratio_2, ratio_3 = matching_accuracy_metric(
                                [response_map_2, sub_2D_locations_2, boundaries])
                            accuracy_1 += ratio_1 * sub_1D_locations_1.shape[1]
                            accuracy_2 += ratio_2 * sub_1D_locations_1.shape[1]
                            accuracy_3 += ratio_3 * sub_1D_locations_1.shape[1]

                        accuracy_1 /= query_size
                        accuracy_2 /= query_size
                        accuracy_3 /= query_size
                        if mean_accuracy_1 is None:
                            mean_accuracy_1 = np.mean(accuracy_1.item())
                            mean_accuracy_2 = np.mean(accuracy_2.item())
                            mean_accuracy_3 = np.mean(accuracy_3.item())
                        else:
                            mean_accuracy_1 = mean_accuracy_1 * (total_query / (total_query + query_size)) + \
                                              accuracy_1 * (query_size / (total_query + query_size))
                            mean_accuracy_2 = mean_accuracy_2 * (total_query / (total_query + query_size)) + \
                                              accuracy_2 * (query_size / (total_query + query_size))
                            mean_accuracy_3 = mean_accuracy_3 * (total_query / (total_query + query_size)) + \
                                              accuracy_3 * (query_size / (total_query + query_size))
                        total_query += query_size

                        step += 1
                        tq.set_postfix(
                            accuracy_1='average: {:.5f}, current: {:.5f}'.format(mean_accuracy_1, accuracy_1.item()),
                            accuracy_2='average: {:.5f}, current: {:.5f}'.format(mean_accuracy_2, accuracy_2.item()),
                            accuracy_3='average: {:.5f}, current: {:.5f}'.format(mean_accuracy_3, accuracy_3.item())
                        )
                        writer.add_scalars('Test', {'accuracy_1': mean_accuracy_1, 'accuracy_2': mean_accuracy_2,
                                                    'accuracy_3': mean_accuracy_3}, step)
                    tq.update(1)
                tq.close()

    writer.close()
