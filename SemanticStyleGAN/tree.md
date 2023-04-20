主要なプログラムのファイル構成をここに示す．

|-- LICENSES/ : ライセンス関連
|-- criteria/ : 基準となる距離や評価の定義
|-- images/ : モーフィングに用いる入力画像の保管
|-- models/ : SemanticStyleGANのモデル
|-- pretrained/ : 学習済みモデルを保管
|-- results/ : 出力ファイル
|   |-- components/ : パーツ分けをした出力画像・フーリエスペクトル
|   |-- interpolation/ : モーフィング画像・動画
|   `-- inversion/ : 潜在変数の保管
|-- visualize/ : プログラム本体
|   |-- generate_components.py* : 作成したモーフィングをパーツ分け（スペクトル化）
|   |-- generate_morph.py* : 潜在変数からモーフィング作成
|   |-- intermediate_expression.py* : 潜在変数のモーフィング 
|   `-- invert.py : 画像から潜在変数へ変換する
|-- components_display.py* : -> generate_components.py
|-- main.ipynb* : 本研究のnote
|-- morphing.py* : -> generate_morph.py
|-- pair_inverting.py* : -> invert.py
|-- prepare_image_data.py*
|-- prepare_inception.py*
|-- prepare_mask_data.py* 
|-- requirements.txt : パッケージリストを記載しているファイル
|-- resize512.py : 画像を512x512へリサイズ
|-- train.py : モデルの学習
|-- train_adaptation.py : モデルのドメイン適応
`-- tree.md : ファイル構成
