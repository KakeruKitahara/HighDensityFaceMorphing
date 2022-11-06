# GAN による表情の高品質高密度モーフィング画像生成とそれによる表情弁別閾値楕円の評価

## 概論
- 作成中

## セットアップ

### Windows

#### docker
[Docker Desktop for Windows](https://docs.docker.jp/desktop/install/windows-install.html)を導入する．詳細は https://qiita.com/gahoh/items/7b21377b5c9e3ffddf4a を参照する．

#### GPU
1. [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) をセッティングする．CUDA Toolkit 11.8.0 を現在は使用している．
2. [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) をセッティングする．cuDNN のバージョンは CUDA のバージョンに対応させること．

詳細は https://rupic.hatenablog.com/entry/2020/03/24/021455 を参照する．

### Linunx (Debian)
- 作成中
### コンテナ作成
`docker-compose up -d` をしてイメージ作成，コンテナ作成をする．コンテナ名は`highmorph`となる．

## 使用方法
セットアップで作成したコンテナを起動しているときは[localhost:8888](http://localhost:8888)においてjupyter notebookにアクセスできますので，そちらから参照．
