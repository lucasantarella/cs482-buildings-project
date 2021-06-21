FROM pure/python:3.8-cuda10.2-base

ARG DEBIAN_FRONTEND=noninteractive

# Install deps
RUN pip install torchvision
RUN pip install torch
#for bayes opt
RUN pip install botorch
#for mask generation
RUN pip install rasterio
RUN pip install geopandas

# for unet
RUN pip install tifffile
RUN pip install keras
RUN pip install unet

WORKDIR /workspace