# Object Removal and Inpainting

このプロジェクトは、Hugging Faceの`facebook/detr-resnet-50`を使用した物体検出と、`runwayml/stable-diffusion-inpainting`を使用した画像補完（インペインティング）を実現するためのJupyter Notebookを提供します。ユーザーは画像内の不要なオブジェクトを特定して削除し、削除された領域を周囲とシームレスに補完することができます。

## 機能
- **物体検出**: 事前学習済みの物体検出モデルを使用して画像内のオブジェクトを特定。
- **オブジェクト削除**: 選択したオブジェクトを削除。
- **画像補完**: 削除された領域を周囲の背景に基づいて補完。

## 実行環境
Google Colab

## 使用方法

1. **リポジトリをクローンする**
   このプロジェクトのリポジトリをクローンして、Jupyter Notebookファイルをダウンロードしてください：

   ```
   git clone https://github.com/your-username/object-removal-inpainting.git
   ```

2. **Google Colabでファイルを開く**
   - `Object_Removal_Inpainting.ipynb`をGoogle Colabにアップロードして開きます。

3. **ランタイムの設定**
   - [ランタイム] > [ランタイムのタイプを変更] から、「ハードウェアアクセラレータ」を「GPU」に設定してください。

4. **必要なライブラリのインストール**
   - Notebook内の以下のセルを実行して必要なライブラリをインストールしてください。

     ```bash
     !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     !pip install transformers diffusers pillow opencv-python-headless numpy
     ```

5. **画像の準備**
   - 処理したい画像を`001.jpg`としてアップロードしてください。別の名前を使用する場合はNotebook内でファイル名を変更してください。

6. **Notebookの実行**
   - 各セルを上から順に実行してください。
   - 検出されたオブジェクトに番号が振られるので、削除したいオブジェクトの番号を入力してください。

## 出力例
以下のような画像ファイルが生成されます：
- **`annotated_image.jpg`**: 検出された物体と番号が描画された画像。
- **`removed_image.jpg`**: 選択した物体が削除された画像。
- **`result_image.jpg`**: 削除された領域が補完された最終的な画像。

## 注意事項
- 高解像度の画像の場合、処理時間が長くなる可能性があります。

## 謝辞
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
