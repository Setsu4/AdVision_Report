# Object Removal and Inpainting

このプロジェクトは、Hugging Faceの`facebook/detr-resnet-50`を使用した物体検出と、`runwayml/stable-diffusion-inpainting`を使用した画像補完（インペインティング）を実現するパイプラインを提供します。ユーザーは画像内のオブジェクトを特定して削除し、削除された領域を周囲とシームレスに補完することができます。

## 機能
- 事前学習済みの物体検出モデルを使用した画像内のオブジェクト検出。
- 選択したオブジェクトの削除。
- 削除した領域を周囲の背景に基づいてシームレスに補完。

## 実行環境
- Python 3.9.21 (動作確認済み)
- CUDA 対応の NVIDIA GPU

## インストール

### ステップ 1: 仮想環境のセットアップ
`conda`を使用して以下のように仮想環境を作成してください（Python 3.9.21で動作確認済み）：

```bash
conda create -n object_removal_env python=3.9 -y
conda activate object_removal_env
```

### ステップ 2: 必要なライブラリのインストール
以下のコマンドで必要なライブラリをインストールします：

```bash
pip install transformers diffusers pillow opencv-python-headless numpy

# PyTorchのインストール
# 実行環境に適したバージョンを以下のリンクから確認してインストールしてください。
# https://pytorch.org/get-started/locally/
# 例: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers diffusers pillow opencv-python-headless numpy
```

> **注意**: CUDAバージョンはお使いのシステム構成に応じて調整してください。

## 使用方法

このプログラムは、入力画像としてデフォルトで`001.jpg`を使用するように設定されています。

1. **リポジトリをクローンする**

   このリポジトリをローカル環境にクローンします：

   ```bash
   git clone https://github.com/your-username/object-removal-inpainting.git
   cd object-removal-inpainting
   ```

2. **入力画像を配置する**

   処理したい画像を`001.jpg`という名前でスクリプトと同じディレクトリに保存するか、スクリプト内の以下の部分を書き換えて任意のファイル名を指定してください：

```python
image_path = "001.jpg"
```

3. **プログラムを実行する**

   以下のコマンドでプログラムを実行します：

   ```bash
   python Object_Removal_Inpainting.py
   ```

4. **プログラムと対話する**

   プログラムを実行すると、以下のように物体検出結果が表示されます。

   ```
   モデルを読み込んでいます...
   画像を処理しています...
   検出結果を描画しています...
   物体認識結果を 'annotated_image.jpg' に保存しました。画像ビューアで確認してください。
   削除したい物体の番号をカンマ区切りで入力してください（例: 0,2,3）: 
   ```

   1. **物体の選択**:
      - 検出された物体には番号が振られています。
      - 番号をカンマ区切りで入力することで、削除したい物体を選択します（例: `1,3`）。

   2. **処理結果**:
      - プログラムは選択された物体を削除し、以下の3つの画像を生成します：
        - `annotated_image.jpg`: 検出された物体とバウンディングボックスが表示された画像。
        - `removed_image.jpg`: 選択した物体が削除された画像。
        - `result_image.jpg`: 削除された領域が補完された最終的な画像。

   3. **出力メッセージ**:
      - 処理が完了すると以下のようなメッセージが表示されます：

      ```
      選択した物体を削除した画像を 'removed_image.jpg' に保存しました。画像ビューアで確認してください。
      補完結果を 'result_image.jpg' に保存しました。画像ビューアで確認してください。
      ```バウンディングボックスと番号を表示します。
   - 削除したい物体に対応する番号を入力してください（例: `0,2,3`）。
   - 以下の3つの出力が生成されます：
     - `annotated_image.jpg`: 検出された物体とバウンディングボックスが表示された画像。
     - `removed_image.jpg`: 選択した物体が削除された画像。
     - `result_image.jpg`: 削除された領域が補完された最終的な画像。

## 出力例
- **入力画像**: ユーザーが提供するオリジナルの画像（`003.jpg`）。
- **注釈付き画像**: 検出された物体とバウンディングボックスが描画された画像（`annotated_image.jpg`）。
- **削除後の画像**: 選択した物体が削除された画像（`removed_image.jpg`）。
- **補完結果画像**: 削除された領域が補完された画像（`result_image.jpg`）。

## 注意事項
- 使用するStable Diffusionモデルには高いGPUメモリが必要です。
- 高解像度の画像の場合、処理時間はハードウェアに依存します。

## 謝辞
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)

