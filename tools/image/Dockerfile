ARG BASE_IMAGE
ARG PROC

FROM $BASE_IMAGE as base_cpu
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
  python3-pip \
  vim \
  nano \
  net-tools \
  rsync \
  zip \
  htop \
  curl \
  wget \
  iputils-ping \
  && apt clean \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN git clone https://github.com/UCSD-DSC178/donkeycar.git && \
    cd donkeycar && \
    DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache pip install -e .[pc]

RUN git clone https://github.com/tawnkramer/gym-donkeycar && \
    cd gym-donkeycar && \
    DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache pip install -e .[gym-donkeycar]

FROM base_cpu as base_gpu

RUN apt update && DEBIAN_FRONTEND=noninteractive apt -y install --no-install-recommends \
    libcudnn8=8.4.1.50-1+cuda11.6 \
    libcudnn8-dev=8.4.1.50-1+cuda11.6 \
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

RUN DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache pip install seaborn cupy-cuda116 tensorflow-gpu==2.2.0 && \
    DEBIAN_FRONTEND=noninteractive pip3 install -U --no-cache pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

FROM base_${PROC} as final

CMD ["/bin/bash"]

