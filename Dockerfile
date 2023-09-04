FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu18.04

WORKDIR /root/work

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  python3 \
  curl \
  python3-dev \
  libgl1-mesa-dev \
  python3-pip \
  nodejs \
  npm && \
  apt-get autoremove -y && apt-get clean && \
  rm -rf /usr/local/src/*

RUN npm update -g npm && \
  npm install -g n && \
  n 16

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 30 && \
  update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 30

# SemanticStyleGANはmaster ( https://github.com/seasonSH/SemanticStyleGAN#pretrained-models )
COPY SemanticStyleGAN/requirements.txt .

RUN pip install --upgrade pip && \
  pip install --no-cache-dir  \
  opencv-python


RUN  pip install --no-cache-dir -r  requirements.txt && \
  pip install --no-cache-dir  jupyterlab \ 
  autopep8 \
  jupyterlab_code_formatter  && \
  rm -rf ~/.cache/pip

# @lckr/jupyterlab_variableinspector@3.0.7 : 変数や行列の中身を確認，@lckr/jupyterlab_variableinspector@3.0.7 : 自動整形
RUN jupyter labextension install @lckr/jupyterlab_variableinspector@3.0.7
