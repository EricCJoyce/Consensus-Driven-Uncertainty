# [EPOS](https://github.com/thodan/epos)
[Estimating 6D Pose of Objects with Symmetries (Hodan et al. 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.pdf) proposes a novel way to improve the accuracy of 6D pose estimates for objects that are textureless or symmetrical. The approach requires defining 3D models of objects to be recognized as sets of fragments. The number of fragments must be fixed before training. The network then learns to predict probabilities for fragments to which a pixel might belong. This handles symmetry: consider that points on a textureless sphere would have uniform probability of belonging to fragments of the sphere. These predictions are converted to confidence scores and passed to a variant of PnP-RANSAC that can better handle the many-to-many relationship that texturelessness and symmetry impose on a problem that had been previously approached under a one-to-one assumption. EPOS outperforms all its competing RGB-only methods, though it is slower.

## Pre-req
Do NOT use the BOP-renderer installation routine suggested by the EPOS authors. Do this instead:
```
sudo apt install libosmesa6-dev
```

## Set up
Create and enter `~/Documents/EPOS`

### Virtual environment
Build a virtual environment for EPOS:
```
conda create --name epos python=3.6.10
conda activate epos

conda install numpy=1.16.6
conda install tensorflow-gpu=1.12.0
conda install pyyaml=5.3.1
conda install opencv=3.4.2
conda install pandas=1.0.5
conda install tabulate=0.8.3
conda install imageio=2.9.0
conda install pip
pip install pypng
conda install -c conda-forge igl
conda install glog=0.4.0
```

Call `conda list` to check that you have the following:
```
# packages in environment at /home/eric/miniconda3/envs/epos:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
_tflow_select             2.1.0                       gpu  
absl-py                   0.15.0             pyhd3eb1b0_0  
astor                     0.8.1            py36h06a4308_0  
blas                      1.0                    openblas  
bzip2                     1.0.8                h7b6447c_0  
c-ares                    1.19.1               h5eee18b_0  
ca-certificates           2023.08.22           h06a4308_0  
cairo                     1.16.0               hb05425b_5  
certifi                   2021.5.30        py36h06a4308_0  
coverage                  5.5              py36h27cfd23_2  
cudatoolkit               9.2                           0  
cudnn                     7.6.5                 cuda9.2_0  
cupti                     9.2.148                       0  
cython                    0.29.24          py36h295c915_0  
dataclasses               0.8                pyh4f3eec9_6  
expat                     2.5.0                h6a678d5_0  
ffmpeg                    4.0                  hcdf2ecd_0  
fontconfig                2.14.1               h4c34cd2_2  
freeglut                  3.0.0                hf484d3e_5  
freetype                  2.12.1               h4a9f257_0  
gast                      0.5.3              pyhd3eb1b0_0  
gflags                    2.2.2                he6710b0_0  
glib                      2.69.1               h4ff587b_1  
glog                      0.4.0                he6710b0_0  
graphite2                 1.3.14               h295c915_1  
grpcio                    1.36.1           py36h2157cd5_1  
h5py                      2.8.0            py36h989c5e5_3  
harfbuzz                  1.8.8                hffaf4a1_0  
hdf5                      1.10.2               hba1933b_1  
icu                       58.2                 he6710b0_3  
igl                       2.2.1            py36h0e5ac07_1    conda-forge
imageio                   2.9.0              pyhd3eb1b0_0  
importlib-metadata        4.8.1            py36h06a4308_0  
intel-openmp              2021.4.0          h06a4308_3561  
jasper                    2.0.14               hd8c5072_2  
jpeg                      9e                   h5eee18b_1  
keras-applications        1.0.8                      py_1  
keras-preprocessing       1.1.2              pyhd3eb1b0_0  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libblas                   3.9.0           13_linux64_openblas    conda-forge
libcblas                  3.9.0           13_linux64_openblas    conda-forge
libdeflate                1.17                 h5eee18b_0  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            7.5.0               ha8ba4b0_17  
libgfortran4              7.5.0               ha8ba4b0_17  
libglu                    9.0.0                hf484d3e_1  
libgomp                   11.2.0               h1234567_1  
libopenblas               0.3.18               hf726d26_0  
libopencv                 3.4.2                hb342d67_1  
libopus                   1.3.1                h7b6447c_0  
libpng                    1.6.39               h5eee18b_0  
libprotobuf               3.17.2               h4ff587b_1  
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.5.1                h6a678d5_0  
libuuid                   1.41.5               h5eee18b_0  
libvpx                    1.7.0                h439df22_0  
libwebp-base              1.2.4                h5eee18b_1  
libxcb                    1.15                 h7f8727e_0  
libxml2                   2.10.4               hcbfbd50_0  
lz4-c                     1.9.4                h6a678d5_0  
markdown                  3.3.4            py36h06a4308_0  
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0            py36h7f8727e_0  
ncurses                   6.4                  h6a678d5_0  
numpy                     1.16.6           py36h0a8e133_3  
numpy-base                1.16.6           py36h41b4c56_3  
olefile                   0.46                     py36_0  
opencv                    3.4.2            py36h6fd60c2_1  
openjpeg                  2.4.0                h3ad879b_0  
openssl                   1.1.1v               h7f8727e_0  
pandas                    1.0.5            py36h0573a6f_0  
pcre                      8.45                 h295c915_0  
pillow                    8.3.1            py36h2c7a002_0  
pip                       21.2.2           py36h06a4308_0  
pixman                    0.40.0               h7f8727e_1  
protobuf                  3.17.2           py36h295c915_0  
py-opencv                 3.4.2            py36hb342d67_1  
pypng                     0.20220715.0             pypi_0    pypi
python                    3.6.10               h7579374_2  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python_abi                3.6                     2_cp36m    conda-forge
pytz                      2021.3             pyhd3eb1b0_0  
pyyaml                    5.3.1            py36h7b6447c_1  
readline                  8.2                  h5eee18b_0  
scipy                     1.5.2            py36habc2bb6_0  
setuptools                58.0.4           py36h06a4308_0  
six                       1.16.0             pyhd3eb1b0_1  
sqlite                    3.41.2               h5eee18b_0  
tabulate                  0.8.3                    py36_0  
tensorboard               1.12.2           py36he6710b0_0  
tensorflow                1.12.0          gpu_py36he74679b_0  
tensorflow-base           1.12.0          gpu_py36had579c0_0  
tensorflow-gpu            1.12.0               h0d30ee6_0  
termcolor                 1.1.0            py36h06a4308_1  
tk                        8.6.12               h1ccaba5_0  
typing_extensions         4.1.1              pyh06a4308_0  
werkzeug                  2.0.3              pyhd3eb1b0_0  
wheel                     0.37.1             pyhd3eb1b0_0  
xz                        5.4.2                h5eee18b_0  
yaml                      0.2.5                h7b6447c_0  
zipp                      3.6.0              pyhd3eb1b0_0  
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.5                hc292b87_0  
```

### Environment variables
To set environment variables, create the file `/home/eric/miniconda3/etc/activate.d/env_vars.sh`.
```
mkdir /home/eric/miniconda3/etc/activate.d
touch /home/eric/miniconda3/etc/activate.d/env_vars.sh
chmod 764 /home/eric/miniconda3/etc/activate.d/env_vars.sh
```

Open it for editing `nano /home/eric/miniconda3/etc/activate.d/env_vars.sh` and paste the following content into it:
```
#!/bin/sh
                                                                                                  # Folder for the EPOS repository.
export REPO_PATH=/home/eric/Documents/Is\ Image-based\ Object\ Pose\ Estimation\ Ready\ to\ Support\ Grasping/V2/Models/EPOS/epos
                                                                                                  # Folder for TFRecord files and trained models (fine to leave this pointing at V1).
export STORE_PATH=/media/eric/Hoboken/Projects/Is\ Image-based\ Object\ Pose\ Estimation\ Ready\ to\ Support\ Grasping/V1/Models/EPOS
                                                                                                  # Folder for BOP datasets (bop.felk.cvut.cz/datasets) (fine to leave this pointing at V1).
export BOP_PATH=/media/eric/Hoboken/Projects/Is\ Image-based\ Object\ Pose\ Estimation\ Ready\ to\ Support\ Grasping/V1/Datasets/BOP

export TF_DATA_PATH=$STORE_PATH/tf_data                                                           # Folder with TFRecord files.
export TF_MODELS_PATH=$STORE_PATH/tf_models                                                       # Folder with trained EPOS models.

export PYTHONPATH=$REPO_PATH:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/bop_renderer/build:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/bop_toolkit:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/progressive-x/build:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/slim:$PYTHONPATH

export LD_LIBRARY_PATH=$REPO_PATH/external/llvm/lib:$LD_LIBRARY_PATH
```

Reactivate the environment:
```
conda activate epos
```

Run the script. (It should run when you reactivate the environment, but... meh.)
```
. /home/eric/miniconda3/etc/activate.d/env_vars.sh
```

Create directories:
- `/media/eric/Hoboken/Projects/Uncertainty for 6DoF Pose Est/Models/EPOS/tf_data`
- `/media/eric/Hoboken/Projects/Uncertainty for 6DoF Pose Est/Models/EPOS/tf_models`

### Clone the EPOS repository
`git clone --recurse-submodules https://github.com/thodan/epos.git`

### BOP Renderer

This clone will include a submodule of the BOP Renderer, which we are going to replace:
```
cd epos/external
sudo rm -R bop_renderer
git clone https://github.com/thodan/bop_renderer.git
cd bop_renderer
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Progressive-X
[Progressive-X](https://github.com/danini/progressive-x) is used to estimate the 6D object poses from 2D-3D correspondences.

Download `essential_estimator.h` from this repository and write it over `EPOS/epos/external/progressive-x/graph-cut-ransac/src/pygcransac/include/essential_estimator.h`.

Download `fundamental_estimator.h` from this repository and write it over `EPOS/epos/external/progressive-x/graph-cut-ransac/src/pygcransac/include/fundamental_estimator.h`.

Download `GCRANSAC.h` from this repository and write it over `EPOS/epos/external/progressive-x/graph-cut-ransac/src/pygcransac/include/GCRANSAC.h`.

Download `homography_estimator.h` from this repository and write it over `EPOS/epos/external/progressive-x/graph-cut-ransac/src/pygcransac/include/homography_estimator.h`.

Download `solver_p3p.h` from this repository and write it over `EPOS/epos/external/progressive-x/graph-cut-ransac/src/pygcransac/include/solver_p3p.h`.

I have initialed all changes.

Make sure your GCC version supports C++17 and compile Progressive-X by:
```
cd $REPO_PATH/external/progressive-x/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Finally, do this: `ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/eric/miniconda3/envs/epos/lib/libstdc++.so.6`.

## BOP dataset
This guide assumes you have already downloaded the BOP dataset.

## Pre-trained weights
Download pre-trained weights for the various BOP datasets:

- [LMO(64 fragments per object, CVPR)](https://bop.felk.cvut.cz/media/data/epos_store/lmo-cvpr20-xc65-f64.zip)
- [LMO(256 fragments per object, CVPR)](https://bop.felk.cvut.cz/media/data/epos_store/lmo-cvpr20-xc65-f256.zip)
- [LMO(64 fragments per object, BOP)](https://bop.felk.cvut.cz/media/data/epos_store/lmo-bop20-xc65-f64.zip)

Likewise for all the other datasets.

Unzip these into `/media/eric/Hoboken/Projects/Uncertainty for 6DoF Pose Est/Models/EPOS/tf_models
` such that:
```
/tf_models
  +--lmo-bop20-xc65-f64
  |    +--train
  |    |    +--checkpoint
  |    |    +--model.ckpt-2000001.data-00000-of-00001
  |    |    \--model.ckpt-2000001.index
  |    +--fragments.pkl
  |    \--params.yml
  +--lmo-cvpr20-xc65-f64
  |    +--train
  |    |    +--checkpoint
  |    |    +--model.ckpt-2000000.data-00000-of-00001
  |    |    \--model.ckpt-2000000.index
  |    +--fragments.pkl
  |    \--params.yml
  +--lmo-cvpr20-xc65-f256
  |    +--train
  |    |    +--checkpoint
  |    |    +--model.ckpt-2000000.data-00000-of-00001
  |    |    \--model.ckpt-2000000.index
  |    +--fragments.pkl
  |    \--params.yml
  :
  :
```

Likewise for all the other datasets.

## Test
All your work should be done from the directory `EPOS/epos/scripts/`.

### Build a TFRecord for every set you intend to test
Note: you will have to "hide" any directories that will confuse the collection script. It traverses the split directory on the assumption that everythin within has a numeric name.
```
python3 create_example_list.py --dataset=lmo --split=test
```

Now turn those examples into a TFRecord:
```
python3 create_tfrecord.py --dataset=lmo --split=test --examples_filename=lmo_test_examples.txt --add_gt=True --shuffle=True --rgb_format=png
```

```
export CUDA_VISIBLE_DEVICES=0
python3 infer.py --model=lmo-bop20-xc65-f64
```

## Clean up
Deactivate the `epos` environment: `conda deactivate`

Delete the environment: `conda remove --name epos --all`
