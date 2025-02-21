# GANによる潜在空間を利用した表情モーフィングの生成と心理物理表情空間における意味次元の測定
自身の研究である"GANによる表情モーフィング画像生成"のプログラムである．

## 概論
現在，表情認識に次元説が知られており、心理空間における意味次元の検討がなされている．一方、心理空間の問題点を解決するために，炭矢らは表情の心理物理空間を提案した．本研究では，SemanticStyleGANの潜在空間を用いることでモーフィングを作成する手法を提案し，高次元の表情弁別閾値楕円の測定を可能にした．また，潜在空間と表情空間の間に局所的な単体補間法を用いることで弁別楕円の軸方向のモーフィングを作成し，それに対して心理物理表情空間における意味次元の測定を試みた．
> 2024年度の修士論文から一部引用．

## 関連発表
- 飯野 匠, 小林 洋明, 趙 晋輝, “GANによる表情のモーフィング作成と高次元表情弁別閾値楕円面の推定”, FIT2023（第22回情報科学技術フォーラム）, 2023年9月.
- 飯野 匠, 小林 洋明, 趙 晋輝, “GANの潜在空間を利用した表情モーフィング画像生成と高次元表情弁別閾値楕円の測定”, HCGシンポジウム2023, 2023年12月.
- 飯野 匠,浜崎 昂多,趙 晋輝, “GANによる潜在空間を利用した表情モーフィングの生成と心理物理表情空間における意味次元の測定”, 電子情報通信学会ヒューマンコミュニケーション基礎 (HCS) 研究会, 2025年3月.

## dockerセットアップ
Dockerを用いて仮想コンテナ上で開発環境を実装する．ファインチューニングにgpuを使うので事前に自分のグラボのドライバを入れておくこと．並列計算処理などをするCUDA，cuDNNなどをダウンロードする必要はない．

### Windows
[Docker Desktop for Windows](https://docs.docker.jp/desktop/install/windows-install.html)を導入する．詳細は https://qiita.com/gahoh/items/7b21377b5c9e3ffddf4a を参照する．それに伴いWSL2も入れる必要がある．

### Debian (Ubuntu)
[Docker for Linux](https://docs.docker.jp/linux/index.html)を導入する．詳細は https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04-ja を参照する．

## コンテナ起動
### SemanticStyleGAN
`docker-compose up -d --build -d` をしてイメージ作成，コンテナ作成をする．コンテナ名は`highmorph`となる． \
セットアップで作成したコンテナを起動しているときは[localhost:8081](http://localhost:8080)においてjupyter notebookにアクセスできますので，そちらから参照する．


## 使用方法
  docker起動し，`main.ipynb`をjupyter nootbookなどで参照すること．

## Docker環境
Dockerで実装した環境をここに記す．
自分で調べたい場合は各自以下のコマンドを入力すること．
プログラムに合わせて使用しなければならないバージョンが存在して，それらを合わせないとエラーをはくので変更する際は十分注意をする．
- `nvcc --version` : CUDAのバージョン
- `dpkg -l | grep 'cudnn'` : cuDNNのバージョン
- `pip list | grep 'torch'` : pytorch, torchvisionのバージョン
- `pip list | grep 'tensorflow'` : tensorflow, tensorflow-gpuのバージョン
　
```
OS : ubuntu20.04
CUDA : 11.2
cuDNN : 8.0
pytorch : 1.10.1+cu111
torchvision : 0.11.2+cu111
```

## 免責
生じた如何なる損害や修正や更新も責任を負いません． 使用する際は自己責任でお願いします．

## クレジット
- GANFaceMorphing（先行研究） : https://github.com/KakeruKitahara/GANFaceMorphing
- Semantic StyleGAN : https://github.com/seasonSH/SemanticStyleGAN#pretrained-models

## ■ 更新情報

```
v1.0.0
・実装完了
・公開

v1.1.0
・首や髪の推論の除外

v1.2.0
・PTIによるファインチューニングの実装
・単体補間法によるモーフィング補間の実装
・更新一時終了
```