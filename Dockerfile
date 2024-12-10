FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

WORKDIR /root/work
ENV DEBIAN_FRONTEND=noninteractive

# Python3.8
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  software-properties-common \
  curl \
  libgl1-mesa-dev && \
  curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
  apt-get install -y --no-install-recommends nodejs && \
  apt-get autoremove -y && apt-get clean && \
  rm -rf /usr/local/src/* && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y python3.8 python3.8-distutils python3.8-dev pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 30

# SemanticStyleGANはmaster ( https://github.com/seasonSH/SemanticStyleGAN#pretrained-models )
COPY SemanticStyleGAN/requirements.txt .

RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install -U pip && \
  pip install \
  jupyterlab \
  autopep8 \
  jupyterlab_code_formatter && \
  pip install --no-cache-dir -r  requirements.txt && \
  rm -rf ~/.cache/pip