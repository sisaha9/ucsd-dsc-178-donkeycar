ARG BASE_IMAGE
ARG PROC

FROM $BASE_IMAGE as base
LABEL maintainer="Siddharth Saha <sisahawork@gmail.com>"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

## Install apt packages
RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y install --no-install-recommends \
  ca-certificates \
  && apt clean \
  && rm -rf /var/lib/apt/lists/*

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y install --no-install-recommends \
  software-properties-common \
  build-essential \
  sudo \
  git \
  udev \
  python3-pip \
  python3-setuptools \
  vim \
  nano \
  net-tools \
  rsync \
  zip \
  htop \
  curl \
  wget \
  iputils-ping \
  ffmpeg \
  libsm6 \
  libxext6 \
  && apt clean \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN DEBIAN_FRONTEND=noninteractive pip3 uninstall --no-cache tensorflow && \
    DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache install seaborn

FROM base as base_cpu

RUN DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache install tensorflow==2.8.0 && \
    DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

FROM base as base_gpu

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y install --no-install-recommends \
    libcudnn8=8.4.0.27-1+cuda11.6  \
    libcudnn8-dev=8.4.0.27-1+cuda11.6 \
    libnvinfer8=8.4.3-1+cuda11.6 \
    libnvinfer-dev=8.4.3-1+cuda11.6 \
    libnvinfer-plugin8=8.4.3-1+cuda11.6 \
    libnvinfer-plugin-dev=8.4.3-1+cuda11.6 \
    libnvonnxparsers8=8.4.3-1+cuda11.6 \
    libnvonnxparsers-dev=8.4.3-1+cuda11.6 \
    libnvparsers8=8.4.3-1+cuda11.6 \
    libnvparsers-dev=8.4.3-1+cuda11.6 \
    python3-libnvinfer=8.4.3-1+cuda11.6 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo export CUDA_HOME="/usr/local/cuda-11.7" >> /etc/bash.bashrc && \
    echo export PATH="/usr/local/cuda-11.7/bin:${PATH}" >> /etc/bash.bashrc && \
    echo export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH}" >> /etc/bash.bashrc

RUN DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache install cupy-cuda117 tensorflow-gpu==2.8.0 && \
    DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

FROM base_${PROC} as final

RUN pip3 install --no-cache git+https://github.com/sisaha9/PyVESC.git@sid-devel

RUN git clone https://github.com/luxonis/depthai.git && \
    git clone https://github.com/luxonis/depthai-python.git && \
    cd depthai && \
    curl -fL https://docs.luxonis.com/install_dependencies.sh | bash && \
    python3 install_requirements.py && \
    cd ../depthai-python/examples && \
    python3 install_requirements.py 

RUN git clone https://github.com/sisaha9/donkeycar.git -b sid_devel && \
    cd donkeycar && \
    DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache install -e .[pc]

RUN git clone https://github.com/sisaha9/gym-donkeycar -b sid_devel && \
    cd gym-donkeycar && \
    DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache install -e .[gym-donkeycar]

CMD ["/bin/bash"]


