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

# more deps
RUN pip install matplotlib

# install gdal
RUN apt-get -y update && apt-get -y install \
    apt-utils \
    build-essential \
    software-properties-common \
    git
RUN add-apt-repository -y ppa:ubuntugis/ppa && \
    apt-get -y update && \
    apt-get -y install gdal-bin python3-gdal

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN pip install GDAL

# Clone UNet
RUN apt-get -y update && apt-get -y install curl unzip
RUN curl -Lo /tmp/unet.zip https://github.com/reachsumit/deep-unet-for-satellite-image-segmentation/archive/refs/heads/master.zip && \
    unzip /tmp/unet.zip -d /tmp && \
    mv /tmp/deep-unet-for-satellite-image-segmentation-master /usr/lib/unet && \
    rm /tmp/unet.zip

WORKDIR /workspace