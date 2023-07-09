FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get -y update && apt-get -y upgrade && apt-get -y install \
    apt-utils \
    python3 \
    python3-pip

# Install deps
RUN pip install torch torchvision torchaudio

#for mask generation
RUN pip install rasterio
RUN pip install geopandas

# for unet
RUN pip install tifffile
RUN pip install keras
RUN pip install unet

# for gpu
RUN pip install tensorflow

# more deps
RUN pip install matplotlib

# USER root
# # RUN commands
# USER 1001

# install gdal
RUN apt-get -y update && apt-get -y install \
    apt-utils \
    build-essential \
    software-properties-common \
    git
RUN add-apt-repository -y ppa:ubuntugis/ppa && \
    apt-get -y update && \
    apt-get -y install gdal-bin libgdal-dev

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN pip install GDAL

# Add openvc
RUN apt-get -y update && \
    apt-get -y install ffmpeg libsm6 libxext6
RUN pip install opencv-python cython

# for jupyter
RUN pip install jupyter notebook jupyterhub jupyterlab

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions
COPY jupyter_notebook_config.py /etc/jupyter/

#User information
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Create NB_USER with name jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN useradd -l -m -s /bin/bash -N -u "${NB_UID}" "${NB_USER}" && \
    chmod g+w /etc/passwd && \
    mkdir -p "${HOME}" &&\
    chown "${NB_USER}:${NB_GID}" "${HOME}" && \
    fix-permissions "${HOME}"


USER ${NB_UID}


