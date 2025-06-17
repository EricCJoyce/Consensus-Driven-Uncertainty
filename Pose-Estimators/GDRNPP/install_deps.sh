#!/usr/bin/env bash
# some other dependencies
set -x
install=${1:-"all"}

if test "$install" = "all"; then
echo "Installing apt dependencies"
apt-get install -y libjpeg-dev zlib1g-dev
apt-get install -y libopenexr-dev
apt-get install -y openexr
apt-get install -y python3-dev
apt-get install -y libglfw3-dev libglfw3
apt-get install -y libglew-dev
apt-get install -y libassimp-dev
apt-get install -y libnuma-dev  # for byteps
apt install -y clang
## for bop cpp renderer
apt install -y curl
apt install -y autoconf
apt-get install -y build-essential libtool

## for uncertainty pnp
apt-get install -y libeigen3-dev
apt-get install -y libgoogle-glog-dev
apt-get install -y libsuitesparse-dev
apt-get install -y libatlas-base-dev

## for nvdiffrast/egl
apt-get install -y --no-install-recommends \
    cmake curl pkg-config
apt-get install -y --no-install-recommends \
    libgles2 \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev
# (only available for Ubuntu >= 18.04)
apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libglvnd-dev

apt-get install -y libglew-dev
# for GLEW, add this into ~/.bashrc
# export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
fi

# conda install ipython

pip3 install -r requirements/requirements.txt

# pip install kornia

pip3 uninstall pillow
CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd

# install kaolin

# (optional) install the nvidia version which is cpp-accelerated
# git clone https://github.com/NVIDIA/cocoapi.git cocoapi_nvidia
# cd cocoapi_nvidia/PythonAPI
# make
# python setup.py build develop

# install detectron2
# git clone https://github.com/facebookresearch/detectron2.git
# cd detectron2 && pip install -e .

# install adet  # https://github.com/aim-uofa/adet.git
# git clone https://github.com/aim-uofa/adet.git
# cd adet
# python setup.py build develop
