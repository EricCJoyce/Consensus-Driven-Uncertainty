# [ZebraPose](https://github.com/suyz526/ZebraPose)
[ZebraPose (Su et al. 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Su_ZebraPose_Coarse_To_Fine_Surface_Encoding_for_6DoF_Object_Pose_CVPR_2022_paper.pdf) produces dense 2D-3D correspondences. One network must be trained for each object to be detected/estimated. Dense correspondences are passed to either RANSAC/PnP or to Progressive-X to solve pose.

## Pre-reqs

```
pip3 install pypng
pip3 install imgaug
pip3 install vispy
pip3 install pyopengl
sudo apt-get install libgflags-dev
sudo apt install libgoogle-glog-dev
```

### OpenCV

```
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install git libgtk-3-dev
sudo apt-get install libjpeg8-dev libtiff5-dev libdc1394-dev libeigen3-dev libtheora-dev libvorbis-dev
sudo apt-get install libtbb2 libtbb-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev sphinx-common yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libswresample-dev
sudo apt-get install libatlas-base-dev gfortran

sudo -s
cd /opt
wget -O opencv.zip https://github.com/Itseez/opencv/archive/4.8.0.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/4.8.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv /opt/opencv-4.8.0/ /opt/opencv/
mv /opt/opencv_contrib-4.8.0/ /opt/opencv_contrib/
rm *.zip
cd opencv
mkdir release
cd release
cmake -D WITH_IPP=ON -D INSTALL_CREATE_DISTRIB=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules /opt/opencv/
make
make install
ldconfig
exit
cd ~
pkg-config --cflags --libs opencv4
pkg-config --modversion opencv4
```

### Progressive-X
```
cd ~/Documents
mkdir Progressive-X
cd Progressive-X
git clone --recursive https://github.com/danini/progressive-x.git
cd progressive-x
```

Replace these files with their equivalents from this repository:
- `progressive-x/graph-cut-ransac/src/pygcransac/include/estimators/essential_estimator.h`
- `progressive-x/graph-cut-ransac/src/pygcransac/include/estimators/fundamental_estimator.h`
- `progressive-x/graph-cut-ransac/src/pygcransac/include/estimators/perspective_n_point_estimator.h`
- `progressive-x/graph-cut-ransac/src/pygcransac/include/GCRANSAC.h`
- `progressive-x/graph-cut-ransac/src/pygcransac/include/neighborhood/neighborhood_graph.h`

I've initialled all changes.

```
cd build
cmake ..
make
cd ..
pip3 install -e .
```

## Set up

```
git clone https://github.com/suyz526/ZebraPose.git --recurse-submodules ZebraPose
```

### Backbone & Datasets

Go to the [Deutsches Forschungszentrum für Künstliche Intelligenz cloud](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW). Download `pretrained_backbone` and any datasets on which you wish to run ZebraPose.

Unzip `pretrained_backbone.zip` and copy or move the contents

- `resnet18-5c106cde.pth`
- `resnet34-333f7ec4.pth`
- `resnet50-19c8e357.pth`

to `ZebraPose/zebrapose/pretrained_backbone/resnet` (which is initially empty.)

In order to use the EfficientNet backbone, create a directory and download its weights:

```
cd ZebraPose/zebrapose/pretrained_backbone
mkdir efficientnet
cd efficientnet
wget https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth
```
The datasets found on the DFKI cloud contain information ZebraPose will need just to run inference.

For **LMO**, copy `models_GT_color` and `test_GT` into the existing `lmo` directory.

For **YCBV**, copy `models_GT_color` and `test_GT` into the existing `ycbv` directory.

### Pre-trained weights
We will not re-train ZebraPose. We will use their trained weights to compute pose errors. Download the following files from this [second DFKI cloud](https://cloud.dfki.de/owncloud/index.php/s/EmQDWgd5ipbdw3E)
- `bop.zip` contains pre-trained pose-estimation weights, given the ResNet backbone.
- `bop_effnet.zip` contains pre-trained pose-estimation weights, given the EfficientNet backbone.

Create a directory `ZebraPose/checkpoints`. If you are using the ResNet backbone, then create a subfolder named `resnet` and unzip the contents of `bop.zip` into `ZebraPose/checkpoints/resnet`. Notice that there is a directory for each dataset on which we may run ZebraPose. If you are using the EfficientNet backbone, then create a subfolder named `effnetb4` and unzip the contents of `bop_effnet.zip` into `ZebraPose/checkpoints/effnetb4`.

## Test
Replace these files with their equivalents from this repository:
- `ZebraPose/bop_toolkit/bop_toolkit_lib/inout.py`
- `ZebraPose/zebrapose/test.py`
- `ZebraPose/zebrapose/test_vivo.py`

I've initialled all changes.

Notice that the out-of-the-box ZebraPose repository includes a directory `zebrapose/detection_results/`. Inside are several subdirectories named for BOP datasets. These are pre-computed object-detections, so that we do not need to train, load, or run object detection before running ZebraPose.

### LMO (ResNet version)
Make sure that line 3 of `ZebraPose/zebrapose/config/config_BOP/lmo/exp_lmo_BOP.txt` reads `bop_challange = True`

Change line 4 of `ZebraPose/zebrapose/config/config_BOP/lmo/exp_lmo_BOP.txt` to `bop_path = /media/eric/Hoboken/Projects/Uncertainty for 6DoF Pose Est/Datasets/BOP`

Change line 35 of `ZebraPose/zebrapose/config/config_BOP/lmo/exp_lmo_BOP.txt` to `check_point_path=/home/eric/Documents/ZebraPose/checkpoints/resnet/` (and make sure such a directory exists.)

Change line 36 of `ZebraPose/zebrapose/config/config_BOP/lmo/exp_lmo_BOP.txt` to `tensorboard_path=/home/eric/Documents/ZebraPose/tensorboard_logs/runs/` (and make sure such a directory exists.)

Following the steps above, you should have a directory named `ZebraPose/checkpoints/resnet/lmo` containing pre-trained weights for objects in this dataset from the DFKI cloud. Notice how these paths are specified in the script calls below.

Create the directory `ZebraPose/evaluation_output`. ZebraPose will write `*.csv` and `*.bbox` files here.

Now run ZebraPose for every object in LMO. Notice that we turn ICP refinement off.

```
cd ZebraPose/zebrapose
python3 test.py --cfg config/config_BOP/lmo/exp_lmo_BOP.txt --obj_name ape --ckpt_file ../checkpoints/resnet/lmo/ape --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/lmo/exp_lmo_BOP.txt --obj_name can --ckpt_file ../checkpoints/resnet/lmo/can --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/lmo/exp_lmo_BOP.txt --obj_name cat --ckpt_file ../checkpoints/resnet/lmo/cat --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/lmo/exp_lmo_BOP.txt --obj_name driller --ckpt_file ../checkpoints/resnet/lmo/driller --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/lmo/exp_lmo_BOP.txt --obj_name duck --ckpt_file ../checkpoints/resnet/lmo/duck --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/lmo/exp_lmo_BOP.txt --obj_name eggbox --ckpt_file ../checkpoints/resnet/lmo/eggbox --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/lmo/exp_lmo_BOP.txt --obj_name glue --ckpt_file ../checkpoints/resnet/lmo/glue --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/lmo/exp_lmo_BOP.txt --obj_name holepuncher --ckpt_file ../checkpoints/resnet/lmo/holepuncher --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
```

### YCBV (ResNet version)
Make sure that line 3 of `ZebraPose/zebrapose/config/config_BOP/ycbv/exp_ycbv_BOP.txt` reads `bop_challange = True`

Change line 4 of `ZebraPose/zebrapose/config/config_BOP/ycbv/exp_ycbv_BOP.txt` to `bop_path = /media/eric/Hoboken/Projects/Uncertainty for 6DoF Pose Est/Datasets/BOP`

Change line 35 of `ZebraPose/zebrapose/config/config_BOP/ycbv/exp_ycbv_BOP.txt` to `check_point_path=/home/eric/Documents/ZebraPose/checkpoints/resnet/` (and make sure such a directory exists.)

Change line 36 of `ZebraPose/zebrapose/config/config_BOP/ycbv/exp_ycbv_BOP.txt` to `tensorboard_path=/home/eric/Documents/ZebraPose/tensorboard_logs/runs/` (and make sure such a directory exists.)

Following the steps above, you should have a directory named `ZebraPose/checkpoints/resnet/ycbv` containing pre-trained weights for objects in this dataset from the DFKI cloud. Notice how these paths are specified in the script calls below.

Create the directory `ZebraPose/evaluation_output`. ZebraPose will write `*.csv` and `*.bbox` files here.

Now run ZebraPose for every object in YCBV. Notice that we turn ICP refinement off.

```
cd ZebraPose/zebrapose
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name master_chef_can --ckpt_file ../checkpoints/resnet/ycbv/master_chef_can --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name cracker_box --ckpt_file ../checkpoints/resnet/ycbv/cracker_box --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name sugar_box --ckpt_file ../checkpoints/resnet/ycbv/sugar_box --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name tomato_soup_can --ckpt_file ../checkpoints/resnet/ycbv/tomato_soup_can --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name mustard_bottle --ckpt_file ../checkpoints/resnet/ycbv/mustard_bottle --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name tuna_fish_can --ckpt_file ../checkpoints/resnet/ycbv/tuna_fish_can --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name pudding_box --ckpt_file ../checkpoints/resnet/ycbv/pudding_box --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name gelatin_box --ckpt_file ../checkpoints/resnet/ycbv/gelatin_box --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name potted_meat_can --ckpt_file ../checkpoints/resnet/ycbv/potted_meat_can --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name banana --ckpt_file ../checkpoints/resnet/ycbv/banana --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name pitcher_base --ckpt_file ../checkpoints/resnet/ycbv/pitcher_base --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name bleach_cleanser --ckpt_file ../checkpoints/resnet/ycbv/bleach_cleanser --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name bowl --ckpt_file ../checkpoints/resnet/ycbv/bowl --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name mug --ckpt_file ../checkpoints/resnet/ycbv/mug --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name power_drill --ckpt_file ../checkpoints/resnet/ycbv/power_drill --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name wood_block --ckpt_file ../checkpoints/resnet/ycbv/wood_block --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name scissors --ckpt_file ../checkpoints/resnet/ycbv/scissors --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name large_marker --ckpt_file ../checkpoints/resnet/ycbv/large_marker --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name large_clamp --ckpt_file ../checkpoints/resnet/ycbv/large_clamp --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name extra_large_clamp --ckpt_file ../checkpoints/resnet/ycbv/extra_large_clamp --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
python3 test.py --cfg config/config_BOP/ycbv/exp_ycbv_BOP.txt --obj_name foam_brick --ckpt_file ../checkpoints/resnet/ycbv/foam_brick --ignore_bit 0 --use_icp False --eval_output_path ../evaluation_output
```

## Render ZebraPose results

Change line 51 in `ZebraPose/bop_toolkit/scripts/vis_est_poses.py` to the explicit path of your results folder. In my case: `/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop`

Change line 15 in `ZebraPose/bop_toolkit/bop_toolkit_lib/config.py` to the explicit path of your dataset directory. In my case: `/media/eric/Hoboken/Projects/Uncertainty for 6DoF Pose Est/Datasets/BOP`

Change line 26 in `ZebraPose/bop_toolkit/bop_toolkit_lib/config.py` to the explicit path of your visualizations directory and be sure that this directory already exists. In my case: `/home/eric/Documents/ZebraPose/visualizations`

The BOP Toolkit insists that outputs be named according to its convention. It's also very rigid about how your dataset has been laid out. This is annoying and motivated a lot of crude edits to `ZebraPose/bop_toolkit/scripts/vis_est_poses.py` and `ZebraPose/bop_toolkit/bop_toolkit_lib/dataset_params.py`, which I have initialed. You'll also have to change the hard-coded `result_filenames` on line 50 in `ZebraPose/bop_toolkit/scripts/vis_est_poses.py` every time you want to visualize something else.

### Render results for LMO
Redefine the variable `result_filenames` in `ZebraPose/bop_toolkit/scripts/vis_est_poses.py` to target results from the LMO dataset:
```
  'result_filenames': [
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/lmo/lmo_ape.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/lmo/lmo_can.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/lmo/lmo_cat.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/lmo/lmo_driller.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/lmo/lmo_duck.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/lmo/lmo_eggbox.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/lmo/lmo_glue.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/lmo/lmo_holepuncher.csv'
  ],
```

Manually specify `method`, `dataset`, `split`, and `split_type` in `ZebraPose/bop_toolkit/scripts/vis_est_poses.py`:
```
  # Parse info about the method and the dataset from the filename.
  result_name = os.path.splitext(os.path.basename(result_fname))[0]
  result_info = result_name.split('_')
  method = 'ZebraPose'                                              #  EJ
  dataset = 'lmo'                                                   #  EJ
  split = 'test'                                                    #  EJ
  split_type = None                                                 #  EJ
  '''                                                               #  EJ
  method = result_info[0]
  dataset_info = result_info[1].split('-')
  dataset = dataset_info[0]
  split = dataset_info[1]
  split_type = dataset_info[2] if len(dataset_info) > 2 else None
  '''                                                               #  EJ
```

The generic call to the rendering script appears as follows:

```
cd ZebraPose/bop_toolkit
python3 scripts/vis_est_poses.py --renderer_type=vispy
```

### Render results for YCBV
Redefine the variable `result_filenames` in `ZebraPose/bop_toolkit/scripts/vis_est_poses.py` to target results from the YCBV dataset:
```
  'result_filenames': [
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_banana.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_bleach_cleanser.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_bowl.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_cracker_box.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_extra_large_clamp.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_foam_brick.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_gelatin_box.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_large_clamp.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_large_marker.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_master_chef_can.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_mug.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_mustard_bottle.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_pitcher_base.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_potted_meat_can.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_power_drill.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_pudding_box.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_scissors.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_sugar_box.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_tomato_soup_can.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_tuna_fish_can.csv',
    '/home/eric/Documents/ZebraPose/evaluation_output/pose_result_bop/ycbv/ycbv_wood_block.csv'
  ],
```

Manually specify `method`, `dataset`, `split`, and `split_type` in `ZebraPose/bop_toolkit/scripts/vis_est_poses.py`:
```
  # Parse info about the method and the dataset from the filename.
  result_name = os.path.splitext(os.path.basename(result_fname))[0]
  result_info = result_name.split('_')
  method = 'ZebraPose'                                              #  EJ
  dataset = 'ycbv'                                                  #  EJ
  split = 'test'                                                    #  EJ
  split_type = None                                                 #  EJ
  '''                                                               #  EJ
  method = result_info[0]
  dataset_info = result_info[1].split('-')
  dataset = dataset_info[0]
  split = dataset_info[1]
  split_type = dataset_info[2] if len(dataset_info) > 2 else None
  '''                                                               #  EJ
```

The generic call to the rendering script appears as follows:

```
cd ZebraPose/bop_toolkit
python3 scripts/vis_est_poses.py --renderer_type=vispy
```

## Render ground-truth poses
The BOP toolkit included in the ZebraPose repository lets you render ground-truth poses. Replace the file `ZebraPose/bop_toolkit/bop_toolkit_lib/visualization.py` with its equivalent in this repository. The single change has been initialed.

Identify the dataset you wish to render on line 21 of `ZebraPose/bop_toolkit/scripts/vis_gt_poses.py`. Identify the split on line 24.

Save changes and then make the following call:

```
cd ZebraPose/bop_toolkit
python3 scripts/vis_gt_poses.py --renderer_type=vispy
```
