
FROM python:3.12-slim
LABEL maintainer="Suhas Katari Chaluva Kumar"

LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6 \
    # For installing pip \
    curl \
    ca-certificates \
    # For versioning \
    git \
    # for convenience \
    wget \
    # Needed for building \
    build-essential \
    # can speed up large_image caching \
    memcached \
    # used to reduce docker image size \
    rdfind \
    && \
    # Clean up to reduce docker size \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# copy HistomicsTK files
ENV itseg_path=/itseg
RUN mkdir -p itseg_path

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    # Install large_image memcached and sources extras \
    # Install girder-client \
    pip install --no-cache-dir girder-client && \
    pip install --no-cache-dir scipy && \
    # Install some other dependencies here to save time in the histomicstk \
    # install step \
    pip install --no-cache-dir importlib opencv-python pandas scikit-learn matplotlib && \
    # clean up \
    rm -rf /root/.cache/pip/* && \
    # Make duplicate files not take extra space in the docker image \
    rdfind -minsize 32768 -makehardlinks true -makeresultsfile false /usr/local

COPY . $itseg_path/
WORKDIR $itseg_path

# Install HistomicsTK and its dependencies
RUN pip install --no-cache-dir virtualenv scikit-build && \
    pip install --no-cache-dir . && \
    rm -rf /root/.cache/pip/* && \
    # Make duplicate files not take extra space in the docker image \
    rdfind -minsize 32768 -makehardlinks true -makeresultsfile false /usr/local

# Show what was installed
RUN pip freeze

# pregenerate font cache
RUN python -c "from matplotlib import pylab"
ENV NUMBA_DISABLE_JIT=1
ENV NUMBA_CACHE_DIR=$itseg_path/
# define entrypoint through which all CLIs can be run
WORKDIR $itseg_path/itseg/cli
LABEL entry_path=$itseg_path/itseg/cli



RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint IntemaSegmentation --help
# Debug import time
RUN python -X importtime IntemaSegmentation/IntemaSegmentation.py --help

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]