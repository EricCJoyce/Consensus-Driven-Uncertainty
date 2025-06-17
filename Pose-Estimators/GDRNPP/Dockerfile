FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

#  Build essentials
#    gcc 9.4.0
#    g++ 9.4.0
#    nvcc 12.2.91
RUN apt update
RUN apt-get install software-properties-common -y
RUN apt install nvidia-cuda-toolkit -y
RUN apt install git wget -y

#  Install cuDNN
COPY cudnn-linux-x86_64-8.9.1.23_cuda12-archive.tar.xz /home/
WORKDIR /home
RUN tar -xvf cudnn-linux-x86_64-8.9.1.23_cuda12-archive.tar.xz
WORKDIR /home/cudnn-linux-x86_64-8.9.1.23_cuda12-archive
RUN cp include/cudnn.h /usr/lib/cuda/include
RUN cp lib/libcudnn* /usr/lib/cuda/lib64
RUN chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn*
RUN echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> /home/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> /home/.bashrc
RUN echo 'export PYTHONPATH="${PYTHONPATH}:$HOME/.local/bin"' >> /home/.bashrc
RUN sh /home/.bashrc
WORKDIR /home
RUN rm cudnn-linux-x86_64-8.9.1.23_cuda12-archive.tar.xz
RUN rm -R cudnn-linux-x86_64-8.9.1.23_cuda12-archive

#  Install Python3.8
#          pip 20.0.2
#          PyTorch 2.0.1
#          Torchvision 0.15.2
RUN apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev libbz2-dev -y
RUN apt install python3-pip -y
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu122
RUN pip3 uninstall --yes starlette
RUN pip3 install starlette==0.27
RUN pip3 install pytorch-lightning==1.7.2
RUN pip3 uninstall --yes torchmetrics
RUN pip3 install torchmetrics==0.11.4
RUN HOROVOD_WITH_PYTORCH=1 pip3 install horovod[pytorch]
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python

#  Install Detectron2
WORKDIR /home
RUN git clone https://github.com/facebookresearch/detectron2.git
WORKDIR /home/detectron2
RUN pip3 install ninja
RUN pip3 install -e .

#  Install BOP Renderer
WORKDIR /home
RUN git clone https://github.com/thodan/bop_renderer.git
WORKDIR /home/bop_renderer
RUN apt install libosmesa6-dev -y
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build

#  Download GDRNPP
WORKDIR /home
RUN git clone https://github.com/shanice-l/gdrnpp_bop2022.git
COPY inout.py /home/gdrnpp_bop2022/lib/pysixd
COPY RT_transform.py /home/gdrnpp_bop2022/lib/pysixd

#  Be able to split up TLESS.
COPY split-tless/configs /home/gdrnpp_bop2022/configs
COPY split-tless/core /home/gdrnpp_bop2022/core

#  Install dependecies and build extensions
COPY install_deps.sh /home/gdrnpp_bop2022/scripts
WORKDIR /home/gdrnpp_bop2022
RUN sh scripts/install_deps.sh
RUN sh scripts/compile_all.sh

#  Link BOP renderer
RUN rm /home/gdrnpp_bop2022/bop_renderer
RUN ln -s /home/bop_renderer /home/gdrnpp_bop2022/bop_renderer

#  Prepare GDRNPP
WORKDIR /home/gdrnpp_bop2022
RUN mkdir datasets
