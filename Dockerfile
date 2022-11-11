FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3 \
  python3-pip \
  curl

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 30 && \
  update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 30

# SemanticStyleGANはv1.0.0 ( https://github.com/seasonSH/SemanticStyleGAN#pretrained-models )
COPY SemanticStyleGAN/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install jupyterlab

# 依存関係ファイルの削除
RUN apt-get autoremove -y && apt-get clean && \
  rm -rf /usr/local/src/*

# nodejs v16 をインストール．この際にnpmもインストールされる．
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs


# 変数や行列の中身を確認
RUN jupyter labextension install @lckr/jupyterlab_variableinspector@3.0.7

# 自動整形
RUN pip install autopep8 \
  && pip install jupyterlab_code_formatter \
  && jupyter labextension install @ryantam626/jupyterlab_code_formatter \
  && jupyter serverextension enable --py jupyterlab_code_formatter