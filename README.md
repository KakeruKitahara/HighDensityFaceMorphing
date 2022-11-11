# GAN による表情の高品質高密度モーフィング画像生成とそれによる表情弁別閾値楕円の評価
SemanticStyleGANを用いて[StyleGANで実装された高品質な表情モーフィング生成](https://github.com/KakeruKitahara/GANFaceMorphing)の改良版．
## 概論
- 作成中
## dockerセットアップ
Dockerを用いて仮想コンテナ上で開発環境を実装する．ファインチューニングにgpuを使うので事前に自分のグラボのドライバを入れておくこと．並列計算処理などをするCUDA，cuDNNなどをダウンロードする必要はない．

### Windows
[Docker Desktop for Windows](https://docs.docker.jp/desktop/install/windows-install.html)を導入する．詳細は https://qiita.com/gahoh/items/7b21377b5c9e3ffddf4a を参照する．それにともなってWSL2も入れる必要がある．

### Debian (Ubuntu)
[Docker for Linux](https://docs.docker.jp/linux/index.html)を導入する．詳細は https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04-ja を参照する．

## コンテナ起動

### StyleGAN_LatentEditors
`docker-compose up -d --build -d` をしてイメージ作成，コンテナ作成をする．コンテナ名は`highmorph`となる． \
セットアップで作成したコンテナを起動しているときは[localhost:8081](http://localhost:8080)においてjupyter notebookにアクセスできますので，そちらから参照する．


## 使用方法

### StyleGAN_LatentEditor
- 作成中

### StyleGAN2
- 作成中

## Docker環境
Dockerで実装した環境をここに記す．
自分で調べたい場合は各自以下のコマンドを入力すること．
プログラムに合わせて使用しなければならないバージョンが存在して，それらを合わせないとエラーをはくので変更する際は十分注意をする．
- `nvcc --version` : CUDAのバージョン
- `dpkg -l | grep 'cudnn'` : cuDNNのバージョン
- `pip list | grep 'torch'` : pytorch, torchvisionのバージョン
- `pip list | grep 'tensorflow'` : tensorflow, tensorflow-gpuのバージョン

### StyleGAN_LatentEditor　
```
OS : ubuntu18.04
CUDA : 10.0
cuDNN : 7.0
pytorch : 1.10.1
torchvision : 0.11.2
tensorflow, tensorflow-gpu : 1.14.0
```

### StyleGAN2
```
OS : ubuntu18.04
CUDA : 10.1
cuDNN : 7.6.5.32
pytorch : 1.10.2
torchvision : 0.9.1
tensorflow, tensorflow-gpu : 2.3.0
```

## クレジット
- 作成中

## 免責
- 作成中

## Docker環境
- 作成中
