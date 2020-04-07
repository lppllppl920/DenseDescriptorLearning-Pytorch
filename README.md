# DenseDescriptorLearning-Pytorch

![Ours](point_cloud_overlay_fm_only_spatial_grouping.gif) ![SIFT](point_cloud_overlay_SIFT.gif)

*The video on the left is the video overlay of the SfM results estimated with our proposed dense descriptor. The video on the right is the SfM results using SIFT*


This codebase implements the method described in the paper:

***Extremely Dense Point Correspondences using a Learned Feature Descriptor***

Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, Mathias Unberath

In ***2020 Conference on Computer Vision and Pattern Recognition (CVPR)***

Please contact [**Xingtong Liu**](http://www.cs.jhu.edu/~xingtongl/) (xingtongliu@jhu.edu) or [**Mathias Unberath**](https://www.cs.jhu.edu/faculty/mathias-unberath/) (unberath@jhu.edu) if you have any questions.

We kindly ask you to cite [this paper](https://arxiv.org/abs/2003.00619) if the code is used in your own work.
```
@misc{liu2020extremely,
Author = {Xingtong Liu and Yiping Zheng and Benjamin Killeen and Masaru Ishii and Gregory D. Hager and Russell H. Taylor and Mathias Unberath},
Title = {Extremely Dense Point Correspondences using a Learned Feature Descriptor},
Year = {2020},
Eprint = {arXiv:2003.00619},
}
```

## Instructions

1. Install all necessary python packages: ```torch, torchvision, opencv-python, numpy, tqdm, pathlib, torchsummary, tensorboardX, albumentations, argparse, pickle, plyfile, pyyaml, datetime, random, shutil, matplotlib, tensorflow```.
2. Generate training data from training videos using Structure from Motion (SfM). Please refer to one data example in [this storage](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/xliu89_jh_edu/EQBBHA7nrzpEhwnzu9PHVrkBAw7JTS8skhzMi-NS044mVg?e=fPj31p) for data formats. 
Color images with the format of ```{:08d}.jpg``` are extracted from the video sequence where SfM is applied. ```camer_intrinsics_per_view``` stores the estimated camera intrinsic matrices for all registered views. 
In this example, since all images are from the same video sequence, we assume the intrinsic matrices are the same for all images. The first four rows in this file are focal length along x and y direction, and principal point along x and y direction of the camera for the first frame. 
```motion.yaml``` stores the estimated poses of the camera coordinate system w.r.t. the world coordinate system . ```selected_indexes``` stores all frame indexes of the video sequence. 
```structure.ply``` stores the estimated sparse 3D reconstruction from SfM. ```undistorted_mask.bmp``` is a binary mask used to mask out blank regions of the video frames. 
```view_indexes_per_point``` stores the indexes of the frames that each point in the sparse reconstruction gets triangulated with. 
The views per point are separated by -1 and the order of the points is the same as that in ```structure.ply```. 
```visible_view_indexes``` stores the original frame indexes of the registered views where valid camera poses are successfully estimated by SfM. 
3. We provide a python script, named ```colmap_model_converter.py```, to convert the [COLMAP](https://colmap.github.io/) format to the one used in this codebase. 
All relevant files described above can be generated from COLMAP results, which are ```cameras.bin```, ```points3D.bin```, and ```images.bin```.
One example for using ```colmap_model_converter.py``` is:
```
/path/to/python /path/to/colmap_model_converter.py --colmap_exe_path /path/to/COLMAP.bat --sequence_root /path/to/video/sequence
```
4. Run ```train.py``` with proper arguments for dense descriptor learning. One example is:
```
/path/to/python /path/to/train.py --adjacent_range 1 50 --image_downsampling 4.0 --network_downsampling 64 --input_size 256 320 --id_range 1 --batch_size 4 --num_workers 4 --num_pre_workers 4 --lr_range 1.0e-4 1.0e-3 --validation_interval 1 --display_interval 20 --rr_weight 1.0 --inlier_percentage 0.99 --training_patient_id 1 --testing_patient_id 1 --validation_patient_id 1 --num_epoch 100 --num_iter 3000 --display_architecture --load_intermediate_data --sampling_size 10 --log_root "/path/to/training/directory" --training_data_root "/path/to/training/data" --feature_length 256 --filter_growth_rate 10 --matching_scale 20.0 --matching_threshold 0.9 --cross_check_distance 5.0 --heatmap_sigma 5.0 --visibility_overlap 20 
```
5. Add additional arguments ```--load_trained_model --trained_model_path "/path/to/trained/model"``` to continue previous training. Run ```tensorboard``` to visualize training progress. One example is: ```tensorboard --logdir="/path/to/training/directory/"```.
6. Run ```test.py``` with proper arguments to evaluate the pair-wise feature matching performance of the learned dense descriptor model. One example is:
```
/path/to/python /path/to/test.py --adjacent_range 1 50 --image_downsampling 4.0 --network_downsampling 64 --input_size 256 320 --num_workers 4 --num_pre_workers 4 --inlier_percentage 0.99 --testing_patient_id 1 --load_intermediate_data --visibility_overlap 20
--display_architecture --trained_model_path "/path/to/trained/model" --testing_data_root "/path/to/testing/data" --log_root "/path/to/testing/directory" --feature_length 256 --filter_growth_rate 10 --keypoints_per_iter 3000 --gpu_id 0
```
7. ```dense_feature_matching.py``` can be used to generate pair-wise feature matches for a SfM algorithm to further process on. One usage example is:
```
/path/to/python /path/to/dense_feature_matching.py --image_downsampling 4.0 --network_downsampling 64 --input_size 256 320 --batch_size 1 --num_workers 1 --load_intermediate_data --data_root /path/to/video/sfm/data/ --sequence_root /path/to/video/sequence --trained_model_path /path/to/trained/descriptor/model --feature_length 256 --filter_growth_rate 10 --max_feature_detection 3000 --cross_check_distance 3.0 --id_range 1 --gpu_id 0 --temporal_range 30 --test_keypoint_num 200 --residual_threshold 5.0 --octave_layers 8 --contrast_threshold 5e-5 --edge_threshold 100 --sigma 1.1 --skip_interval 5 --min_inlier_ratio 0.2 --hysterisis_factor 0.7
```
8. Run ```colmap_database_creation.py``` to convert the generated feature matches in HDF5 format to SQLite format, named ```database.db```, that is compatible with COLMAP. One example is:
```
/path/to/python /path/to/colmap_database_creation.py --sequence_root /path/to/video/sequence
``` 
9. Run ```colmap_sparse_reconstruction.py``` to run ```mapper``` in COLMAP for bundle adjustment to generate sparse reconstruction and camera trajectory. One usage example is:
```
/path/to/python /path/to/colmap_sparse_reconstruction.py --colmap_exe_path /path/to/COLMAP.bat --sequence_root /path/to/video/sequence
```
10. Run ```colmap_model_converter.py``` again as described in step 2 if you want to generate point cloud-video overlays like the GIFs above. 
11. Run ```point_cloud_overlay_generation.py``` to generate a point cloud-video overlay video. One example is:
```
/path/to/python /path/to/point_cloud_overlay_generation.py --sequence_root /path/to/video/sequence --display_image --write_video
```

## Disclaimer

This codebase is only experimental and not ready for clinical applications.

Authors are not responsible for any accidents related to this repository.

This codebase is only allowed for non-commercial usage.

