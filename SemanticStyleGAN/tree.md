主要なプログラムのファイル構成をここに示す．

```
|-- LICENSES/ : ライセンス関連
|-- criteria/ : 基準となる距離や評価の定義
|-- images/ : モーフィングに用いる入力画像の保管
|-- mat/ : 単体補間法でもちいるmatlabファイル等（HDF5ベースで保存すること．）
|-- models/ : SemanticStyleGANのモデル
|-- pretrained/ : 学習済みモデルを保管
|-- results/ : 出力ファイル
|   |-- components/ : パーツ分けをした出力画像・フーリエスペクトル
|   |-- interpolation/ : モーフィング画像・動画
|   `-- inversion/ : 潜在変数の保管
|-- visualize/ : プログラム本体
|   |-- generate_components.py* : 作成したモーフィングをパーツ分け（スペクトル化）
|   |-- generate_morph.py* : 潜在変数から線形補間でモーフィング作成
|   |-- intermediate_expression.py* : 潜在変数のモーフィング
|   |-- invert_pti.py* : 画像から潜在変数へ変換する
|   |-- simplex.py* : matファイルと潜在変数から単体補間法でモーフィング作成
|   `-- utils.py* : visualizeによるutils
|-- components_display.py* : -> generate_components.py
|-- main.ipynb* : 本研究のnote
|-- morphing_line.py* : -> simplex.py
|-- morphing_line.py* : -> generate_morph.py
|-- prepare_image_data.py*
|-- prepare_inception.py*
|-- prepare_mask_data.py*
|-- requirements.txt : パッケージリストを記載しているファイル
|-- resize512.py : 画像を512x512へリサイズしてチャンネルを1つにする
|-- train.py : モデルの学習
|-- train_adaptation.py : モデルのドメイン適応
`-- tree.md : ファイル構成
```