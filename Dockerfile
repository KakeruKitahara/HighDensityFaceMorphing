FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y git \
  python3 \
  python3-pip \
  curl

WORKDIR /work

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