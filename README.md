# DenseDescriptorLearning-Pytorch

![Ours](point_cloud_overlay_fm_only_spatial_grouping.gif) ![SIFT](point_cloud_overlay_SIFT.gif)

*The video on the left is the video overlay of the SfM results estimated with our proposed dense descriptor. The video on the right is the SfM results using SIFT"


This codebase implements the method described in the paper:

***Extremely Dense Point Correspondences using a Learned Feature Descriptor***

Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, Mathias Unberath

In ***Conference on Computer Vision and Pattern Recognition (CVPR) 2020***

Please contact [**Xingtong Liu**](http://www.cs.jhu.edu/~xingtongl/) (xliu89@jh.edu) if you have any questions.

We kindly ask you to cite [this paper]() if the code is used in your own work.
```
```

## Instructions

1. Install all necessary python packages: ```torch, torchvision, opencv-python, numpy, tqdm, pathlib, torchsummary, tensorboardX, albumentations, argparse, pickle, plyfile, pyyaml, datetime, random, shutil, matplotlib, tensorflow, ```.

2. Generate training data from training videos using Structure from Motion (SfM) or Simultaneous Localization and Mapping (SLAM). In terms of the format, please refer to one training data example in this repository. We use SfM to generate training data in this work. Color images with the format of "{:08d}.jpg" are extracted from the video sequence where SfM is applied. ```camer_intrinsics_per_view``` stores the estimated camera intrinsic matrices for all registered views. In this example, since all images are from the same video sequence, we assume the intrinsic matrices are the same for all images. The first three rows in this file are focal length, x and y of the principal point of the camera of the first image. ```motion.yaml``` stores the estimated poses of the world coordinate system w.r.t. the corresponding camera coordinate system. ```selected_indexes``` stores all frame indexes of the video sequence. ```structure.ply``` stores the estimated sparse 3D reconstruction from SfM. ```undistorted_mask.bmp``` is a binary mask used to mask out blank regions of the video frames. ```view_indexes_per_point``` stores the indexes of the frames that each point in the sparse reconstruction gets triangulated with. The views per point are separated by -1 and the order of the points is the same as that in ```structure.ply```. ```visible_view_indexes``` stores the original frame indexes of the registered views where valid camera poses are successfully estimated by SfM.

3. Run ```train.py``` with proper arguments for dense descriptor learning. One example is:
```
/path/to/python /path/to/train.py --adjacent_range 1 50 --image_downsampling 4.0 --network_downsampling 64 --input_size 256 320 --id_range 1 --batch_size 4 --num_workers 4 --num_pre_workers 4 --lr_range 1.0e-4 1.0e-3 --validation_interval 1 --display_interval 20 --rr_weight 1.0 --inlier_percentage 0.99 --training_patient_id 1 --testing_patient_id 1 --validation_patient_id 1 --num_epoch 100 --num_iter 3000 --display_architecture --load_intermediate_data --sampling_size 10 --log_root "/path/to/training/directory" --training_data_root "/path/to/training/data" --feature_length 256 --filter_growth_rate 10 --matching_scale 20.0 --matching_threshold 0.9 --cross_check_distance 5.0 --heatmap_sigma 5.0 --visibility_overlap 20 
```
 Add additional arguments ```--load_trained_model --trained_model_path "\path\to\trained\model"``` to continue previous training. Run ```tensorboard``` to visualize training progress. One example is: ```tensorboard --logdir="/path/to/training/directory/"```.


4. Run ```test.py``` with proper arguments to evaluate the pair-wise feature matching performance of the learned dense descriptor model. One example is:
```
/path/to/python /path/to/test.py --adjacent_range 1 50 --image_downsampling 4.0 --network_downsampling 64 --input_size 256 320 --num_workers 4 --num_pre_workers 4 --inlier_percentage 0.99 --testing_patient_id 1 --load_intermediate_data --visibility_overlap 20
--display_architecture --trained_model_path "/path/to/trained/model" --testing_data_root "/path/to/testing/data" --log_root "/path/to/testing/directory" --feature_length 256 --filter_growth_rate 10 --keypoints_per_iter 3000 --gpu_id 0
```


## Disclaimer

This codebase is only experimental and not ready for clinical applications.

Authors are not responsible for any accidents related to this repository.

This codebase is only allowed for non-commercial usage.

