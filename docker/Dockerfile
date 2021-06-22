FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install --no-install-recommends -y \
    python3.8 \
    python3-pip \
    python3.8-dev

# Install deps
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

#for bayes opt
RUN pip install botorch

#for mask generation
RUN pip install rasterio
RUN pip install geopandas

# for unet
RUN pip install tifffile
RUN pip install keras
RUN pip install unet

# for gpu
RUN pip install tensorflow-gpu

# for jupyter
RUN pip install jupyter notebook

WORKDIR /workspace