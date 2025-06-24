# [GDRNPP](https://github.com/shanice-l/gdrnpp_bop2022) for [BOP2022](https://bop.felk.cvut.cz/method_info/275/)
[GDRNPP (Liu et al. 2022)](https://cmp.felk.cvut.cz/sixd/workshop_2022/slides/bop_challenge_2022_results.pdf) improves upon GDR-Net from 2021. The original paper advocates a combination of "indirect" pose-estimation methods (correspondence-based) and "direct" regression of 6DoF pose. Their innovation is that 2D-3D correspondences are generated internally as intermediate features before pose is directly regressed using a learned, patch-based PnP approximator. The authors credit the success of their method to a sound choice for representing rotation as a six-dimensional vector (following "On the Continuity of Rotation Representations in Neural Networks," by Zhou et al. 2019), translation as a Scale-Invariant representation for Translation Error (following "CDPN," Li et al. 2019), and a loss function that combines pose and geometry. CDPN and EPOS are their major influences.

## Docker container
We will create a Docker container that meets GDRNPP's specs and do our work in there.

### Install Docker
```
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Check the installation:
```
sudo systemctl status docker
```

Try it out:
```
sudo docker info                                #  Display basic info: version, number of containers, etc.
sudo docker images                              #  Lists images on your machine.
sudo docker image ls                            #  The same as above.
sudo docker ps -a                               #  List (all) containers on your system. (Containers are runnable instances of images.)
sudo docker pull hello-world                    #  Copy an instance of the hello-world image to your machine.
sudo docker image ls                            #  Image exists...
sudo docker ps -a                               #  Container does not.
sudo docker run hello-world                     #  Run the hello-world container.
sudo docker ps -a                               #  Now the hello-world container should appear in your list (as having run).
sudo docker rm ************                     #  Copy the container id and paste it into the remove command.
sudo docker ps -a                               #  Container is gone.
sudo docker image ls                            #  Image remains.
sudo docker image rm hello-world                #  Remove the image.
sudo docker image ls                            #  Image is gone.
```

Need to completely clear out all lingering Docker garbage? Do this:
```
sudo systemctl stop docker                      #  Halt Docker service.
sudo systemctl status docker                    #  Be sure you halted it.
sudo rm -R /var/lib/docker                      #  This kills everything so be sure there's nothing you'd like to keep.
sudo systemctl start docker                     #  Start Docker service again.
sudo systemctl status docker                    #  Be sure that you started it.
sudo ls -l /var/lib/docker                      #  See that file structure has been recreated--without all the stuff.
```

### Install NVIDIA Container Toolkit
The Toolkit allows Docker containers to use your GPU.
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check your installation of the NVIDIA Container Toolkit:
```
nvidia-ctk --version
```

Now make sure that the Container Toolkit can see your GPU. The command below pulls a container designed only to test the Container Toolkit. If all went well, you'll see the same output you'd see if you ran `nvidia-smi` from the command line (except without any processes because nothing's running in the container.)
```
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

My output:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.05              Driver Version: 535.86.05    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX TITAN X     Off | 00000000:01:00.0  On |                  N/A |
| 22%   40C    P8              18W / 250W |    312MiB / 12288MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

Notice that I used a base image for CUDA version 11.6.2, which is different than what is actually on my host machine: **12.2.0**. I can get away with this when simply running `nvidia-smi`, but this discrepancy will cause installation of [Detectron2](https://github.com/facebookresearch/detectron2) (required by GDRNPP) to fail. The `Dockerfile` in this repository is configured to work with my GPU and my CUDA version.

## Set up

### Collect pre-computed bounding boxes
Download pre-computed test-set bounding boxes from the authors' [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/Eq_2aCC0RfhNisW8ZezYtIoBGfJiRIZnFxbITuQrJ56DjA?e=hPbJz2) (password = `groupji`) or [BaiDu YunPan](https://pan.baidu.com/s/1FzTO4Emfu-DxYkNG40EDKw) (password = `vp58`). There is a single directory named `BOP_DATASETS`:
```
/BOP_DATASETS
  |
  +--hb
  |   \--test
  |       \--test_bboxes
  |           \--yolox_x_640_hb_pbr_hb_test_primesense_bop19.json
  +--icbin
  |   \--test
  |       \--test_bboxes
  |           \--yolox_x_640_icbin_pbr_icbin_bop_test.json
  +--itodd
  |   \--test
  |       \--test_bboxes
  |           \--yolox_x_640_itodd_pbr_itodd_bop_test.json
  +--lmo
  |   \--test
  |       \--test_bboxes
  |           \--yolox_x_640_lmo_pbr_lmo_bop_test.json
  +--tless
  |   \--test
  |       \--test_bboxes
  |           +--yolox_x_640_tless_pbr_tless_bop_test.json
  |           \--yolox_x_640_tless_real_pbr_tless_bop_test.json
  +--tudl
  |   \--test
  |       \--test_bboxes
  |           +--yolox_x_640_tudl_pbr_tudl_bop_test.json
  |           \--yolox_x_640_tudl_real_pbr_tudl_bop_test.json
  \--ycbv
      \--test
          \--test_bboxes
              +--yolox_x_640_ycbv_pbr_ycbv_bop_test.json
              \--yolox_x_640_ycbv_real_pbr_ycbv_bop_test.json
```

These pre-computed bounding boxes need to be integrated with your system's existing copy of the BOP dataset. In may case, the BOP dataset is at `/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP`. It must therefore become:
```
/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP
  |
  +--hb
  |   +--models
  |   +--models_eval
  |   +--test_kinect
  |   +--test_primesense
  |   |   \--test_bboxes
  |   |       \--yolox_x_640_hb_pbr_hb_test_primesense_bop19.json
  |   +--train_pbr
  |   +--val_kinect
  |   +--val_primesense
  |   +--camera_kinect.json
  |   +--camera_primesense.json
  |   +--dataset_info.md
  |   \--test_targets_bop19.json
  +--hope
  +--icbin
  |   +--models
  |   +--models_eval
  |   +--test
  |   |   \--test_bboxes
  |   |       \--yolox_x_640_icbin_pbr_icbin_bop_test.json
  |   +--train
  |   +--train_pbr
  |   +--camera.json
  |   +--dataset_info.md
  |   \--test_targets_bop19.json
  +--icmi
  +--itodd
  |   +--models
  |   +--models_eval
  |   +--test
  |   |   \--test_bboxes
  |   |       \--yolox_x_640_itodd_pbr_itodd_bop_test.json
  |   +--train_pbr
  |   +--val
  |   +--camera.json
  |   +--dataset_info.md
  |   \--test_targets_bop19.json
  +--lm
  +--lmo
  |   +--models
  |   +--models_eval
  |   +--test
  |   |   \--test_bboxes
  |   |       \--yolox_x_640_lmo_pbr_lmo_bop_test.json
  |   +--train
  |   +--camera.json
  |   +--dataset_info.md
  |   \--test_targets_bop19.json
  +--ruapc
  +--tless
  |   +--models
  |   +--models_cad
  |   +--models_eval
  |   +--models_reconst
  |   +--test_primesense_bop
  |   |   \--test_bboxes
  |   |       +--yolox_x_640_tless_pbr_tless_bop_test.json
  |   |       \--yolox_x_640_tless_real_pbr_tless_bop_test.json
  |   +--train_pbr
  |   +--train_primesense
  |   +--train_render_reconst
  |   +--camera_primesense.json
  |   +--dataset_info.md
  |   +--test_targets_bop18.json
  |   \--test_targets_bop19.json
  +--tudl
  |   +--models
  |   +--models_eval
  |   +--test
  |   |   \--test_bboxes
  |   |       +--yolox_x_640_tudl_pbr_tudl_bop_test.json
  |   |       \--yolox_x_640_tudl_real_pbr_tudl_bop_test.json
  |   +--train_pbr
  |   +--train_real
  |   +--train_render
  |   +--camera.json
  |   +--dataset_info.md
  |   \--test_targets_bop19.json
  +--tyol
  \--ycbv
      +--models
      +--models_eval
      +--models_fine
      +--test
      |   \--test_bboxes
      |       +--yolox_x_640_ycbv_pbr_ycbv_bop_test.json
      |       \--yolox_x_640_ycbv_real_pbr_ycbv_bop_test.json
      +--train_pbr
      +--train_real
      +--train_synt
      +--camera_cmu.json
      +--camera_uw.json
      +--dataset_info.md
      \--test_targets_bop19.json
```

### Collect pre-trained pose-estimation network
Download the pre-trained GDR-Net models in `gdrn.zip` from [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EgOQzGZn9A5DlaQhgpTtHBwB2Bwyx8qmvLauiHFcJbnGSw?e=EZ60La) (password = `groupji`) or [BaiDu YunPan](https://pan.baidu.com/s/1LhXblEic6pYf1i6hOm6Otw) (password = `10t3`).

```
gdrn
  |
  +--hb_pbr
  +--hbPbrSO
  +--icbin_pbr
  +--icbinPbrSO
  +--itodd_pbr
  +--itoddPbrSO
  +--lmo_pbr
  +--lmoPbrSO
  +--tless
  +--tlessPbrSO
  +--tlessSO
  +--tudl
  +--tudlPbrSO
  +--tudlSO
  +--ycbv
  +--ycbvPbrSO
  \--ycbvSO
```

GDRNPP uses separate networks for each object for each dataset. These checkpoints plus the BOP dataset are more than I want to pack into a container at once. Instead, we will mount directories for the data and the models in the container at runtime (see below).

The authors' cloud also contains a set of pre-trained detection networks in `yolox.zip`, but we are not interested in those here.

### Build Docker container for GDRNPP
Download `Dockerfile`, `inout.py`, `install_deps.sh`, and `RT_transform.py` into the same directory on your machine. All changes to `inout.py` and `RT_transform.py` have been initialed.

Download `cudnn-linux-x86_64-8.9.1.23_cuda12-archive.tar.xz` from NVIDIA.

Before building the Docker image, your project directory should look like this:
```
/YourProjectDirectory
  |
  +--split-tless
  |   +--configs
  |   +--core
  |   +--tless-1
  |   +--tless-2
  |   +--tless-3
  |   +--tless-4
  |   +--tless-5
  |   \--tless-6
  +--cudnn-linux-x86_64-8.9.1.23_cuda12-archive.tar.xz
  +--Dockerfile
  +--inout.py
  +--install_deps.sh
  \--RT_transform.py
```

The contents of split-tless were necessary work-around for the limitations of my GPU. See notes below.

Build a container named `gdrnpp`:
```
sudo docker build -t gdrnpp .
```

This image will be fairly large. `sudo docker image ls` reveals:
```
REPOSITORY   TAG       IMAGE ID       CREATED          SIZE
gdrnpp       latest    4690fe2e8647   45 seconds ago   37.9GB
```

## Test

Upon running the container, we will map the dataset directory `/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP` to the path `/home/gdrnpp_bop2022/datasets/BOP_DATASETS` inside the container so that GDRNPP can reach the BOP dataset. Similarly, we will map `/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP` to the path `/home/gdrnpp_bop2022/output/gdrn` inside the container so that GDRNPP has access to the various downloaded model weights. The test scripts in the GDRNPP repository will also write pose prediction results to the directory containing model checkpoints.

An annotated testing template:
```
sudo docker run --rm \                          #  Delete the container once it's stopped.
                --shm-size=10gb \               #  Give the container ample shared memory.
                --runtime=nvidia --gpus all \   #  Make host GPU visible to container.
                -dit --name gdrnpp-test \       #  Provide connections (mounts) to data and models.
                --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP,target=/home/gdrnpp_bop2022/datasets/BOP_DATASETS \
                --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP,target=/home/gdrnpp_bop2022/output/gdrn gdrnpp
                                    
sudo docker exec -it gdrnpp-test bash           #  Enter the container.

./core/gdrn_modeling/test_gdrn.sh \             #  Run a test: config, gpu-id, checkpoint-path(also output path)
  configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py \
  0 \
  output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_final_wo_optim.pth
```

### HB
```
sudo docker run --rm --shm-size=10gb --runtime=nvidia --gpus all -dit --name gdrnpp-test --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP,target=/home/gdrnpp_bop2022/datasets/BOP_DATASETS --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP,target=/home/gdrnpp_bop2022/output/gdrn gdrnpp
sudo docker exec -it gdrnpp-test bash

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hb_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_hb.py 0 output/gdrn/hb_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_hb/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/01_bear.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/01_bear/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/03_round_car.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/03_round_car/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/04_thin_cow.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/04_thin_cow/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/08_green_rabbit.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/08_green_rabbit/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/09_holepuncher.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/09_holepuncher/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/10.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/10/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/12.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/12/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/15.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/15/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/17.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/17/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/18_jaffa_cakes_box.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/18_jaffa_cakes_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/19_minions.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/19_minions/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/22_rhinoceros.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/22_rhinoceros/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/23_dog.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/23_dog/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/29_tea_box.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/29_tea_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/32_car.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/32_car/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/33_yellow_rabbit.py 0 output/gdrn/hbPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_hb/33_yellow_rabbit/model_final_wo_optim.pth
```

### ICBIN
```
sudo docker run --rm --shm-size=10gb --runtime=nvidia --gpus all -dit --name gdrnpp-test --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP,target=/home/gdrnpp_bop2022/datasets/BOP_DATASETS --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP,target=/home/gdrnpp_bop2022/output/gdrn gdrnpp
sudo docker exec -it gdrnpp-test bash

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/icbin_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_icbin.py 0 output/gdrn/icbin_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_icbin/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/icbinPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_icbin/01_coffee_cup.py 0 output/gdrn/icbinPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_icbin/coffee_cup/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/icbinPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_icbin/02_juice_carton.py 0 output/gdrn/icbinPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_icbin/juice_carton/model_final_wo_optim.pth
```

### ITODD
```
sudo docker run --rm --shm-size=10gb --runtime=nvidia --gpus all -dit --name gdrnpp-test --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP,target=/home/gdrnpp_bop2022/datasets/BOP_DATASETS --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP,target=/home/gdrnpp_bop2022/output/gdrn gdrnpp
sudo docker exec -it gdrnpp-test bash

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itodd_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_itodd.py 0 output/gdrn/itodd_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_itodd/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/1.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/1/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/2.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/2/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/3.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/3/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/4.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/4/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/5.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/5/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/6.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/6/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/7.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/7/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/8.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/8/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/9.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/9/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/10.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/10/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/11.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/11/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/12.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/12/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/13.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/13/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/14.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/14/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/15.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/15/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/16.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/16/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/17.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/17/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/18.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/18/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/19.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/19/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/20.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/20/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/21.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/21/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/22.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/22/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/23.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/23/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/24.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/24/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/25.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/25/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/26.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/26/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/27.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/27/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/28.py 0 output/gdrn/itoddPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_itodd/28/model_final_wo_optim.pth
```

### LMO
```
sudo docker run --rm --shm-size=10gb --runtime=nvidia --gpus all -dit --name gdrnpp-test --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP,target=/home/gdrnpp_bop2022/datasets/BOP_DATASETS --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP,target=/home/gdrnpp_bop2022/output/gdrn gdrnpp
sudo docker exec -it gdrnpp-test bash

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py 0 output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/ape.py 0 output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/ape/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/can.py 0 output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/cat.py 0 output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/cat/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/driller.py 0 output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/driller/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/duck.py 0 output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/duck/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/eggbox.py 0 output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/eggbox/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/glue.py 0 output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/glue/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/holepuncher.py 0 output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/holepuncher/model_final_wo_optim.pth
```

### TLESS
This dataset is too large to fit within the 10GB I can afford to allocate for the container.

The ideal solution would be to *hide* part of TLESS and run tests on the remaining part. Then restore and conceal other parts, testing whatever is visible in the `tless` directory. We would only need to protect each test's results from being over-written by the next.

But this repo is too over-engineered to allow that. We have to change the config files in the container, too. Very annoying.

```
sudo docker run --rm --shm-size=10gb --runtime=nvidia --gpus all -dit --name gdrnpp-test --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP,target=/home/gdrnpp_bop2022/datasets/BOP_DATASETS --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP,target=/home/gdrnpp_bop2022/output/gdrn gdrnpp
sudo docker exec -it gdrnpp-test bash

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tless/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tless.py 0 output/gdrn/tless/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tless/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/1.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/1/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/2.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/2/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/3.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/3/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/4.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/4/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/5.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/5/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/6.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/6/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/7.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/7/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/8.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/8/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/9.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/9/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/10.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/10/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/11.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/11/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/12.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/12/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/13.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/13/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/14.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/14/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/15.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/15/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/16.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/16/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/17.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/17/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/18.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/18/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/19.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/19/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/20.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/20/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/21.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/21/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/22.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/22/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/23.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/23/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/24.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/24/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/25.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/25/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/26.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/26/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/27.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/27/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/28.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/28/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/29.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/29/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/30.py 0 output/gdrn/tlessPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/30/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tlessSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/1.py 0 output/gdrn/tlessSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tless/1/model_final_wo_optim.pth
```

### TUDL
```
sudo docker run --rm --shm-size=10gb --runtime=nvidia --gpus all -dit --name gdrnpp-test --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP,target=/home/gdrnpp_bop2022/datasets/BOP_DATASETS --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP,target=/home/gdrnpp_bop2022/output/gdrn gdrnpp
sudo docker exec -it gdrnpp-test bash

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tudl/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl.py 0 output/gdrn/tudl/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tudlPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/01_dragon.py 0 output/gdrn/tudlPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/dragon/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tudlPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/02_frog.py 0 output/gdrn/tudlPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/frog/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tudlPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/03_can.py 0 output/gdrn/tudlPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/can/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tudlSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/01_dragon.py 0 output/gdrn/tudlSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/dragon/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tudlSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/02_frog.py 0 output/gdrn/tudlSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/frog/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tudlSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/03_can.py 0 output/gdrn/tudlSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/can/model_final_wo_optim.pth
```

### YCBV
```
sudo docker run --rm --shm-size=10gb --runtime=nvidia --gpus all -dit --name gdrnpp-test --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Datasets/BOP,target=/home/gdrnpp_bop2022/datasets/BOP_DATASETS --mount type=bind,source=/media/eric/Hoboken/Projects/Consensus_Driven_Uncertainty/Models/GDRNPP,target=/home/gdrnpp_bop2022/output/gdrn gdrnpp

sudo docker exec -it gdrnpp-test bash

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0 output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/002_master_chef_can.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/002_master_chef_can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/003_cracker_box.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/003_cracker_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/004_sugar_box.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/004_sugar_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/005_tomato_soup_can.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/005_tomato_soup_can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/006_mustard_bottle.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/006_mustard_bottle/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/007_tuna_fish_can.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/007_tuna_fish_can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/008_pudding_box.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/008_pudding_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/009_gelatin_box.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/009_gelatin_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/010_potted_meat_can.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/010_potted_meat_can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/011_banana.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/011_banana/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/019_pitcher_base.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/019_pitcher_base/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/021_bleach_cleanser.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/021_bleach_cleanser/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/024_bowl.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/024_bowl/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/025_mug.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/025_mug/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/035_power_drill.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/035_power_drill/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/036_wood_block.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/036_wood_block/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/037_scissors.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/037_scissors/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/040_large_marker.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/040_large_marker/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/051_large_clamp.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/051_large_clamp/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/052_extra_large_clamp.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/052_extra_large_clamp/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/061_foam_brick.py 0 output/gdrn/ycbvPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/061_foam_brick/model_final_wo_optim.pth

./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/002_master_chef_can.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/002_master_chef_can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/003_cracker_box.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/003_cracker_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/004_sugar_box.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/004_sugar_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/005_tomato_soup_can.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/005_tomato_soup_can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/006_mustard_bottle.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/006_mustard_bottle/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/007_tuna_fish_can.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/007_tuna_fish_can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/008_pudding_box.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/008_pudding_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/009_gelatin_box.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/009_gelatin_box/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/010_potted_meat_can.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/010_potted_meat_can/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/011_banana.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/011_banana/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/019_pitcher_base.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/019_pitcher_base/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/021_bleach_cleanser.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/021_bleach_cleanser/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/024_bowl.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/024_bowl/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/025_mug.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/025_mug/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/035_power_drill.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/035_power_drill/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/036_wood_block.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/036_wood_block/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/037_scissors.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/037_scissors/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/040_large_marker.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/040_large_marker/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/051_large_clamp.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/051_large_clamp/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/052_extra_large_clamp.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/052_extra_large_clamp/model_final_wo_optim.pth
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/061_foam_brick.py 0 output/gdrn/ycbvSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_ycbv/061_foam_brick/model_final_wo_optim.pth
```

## Render GDRNPP results
