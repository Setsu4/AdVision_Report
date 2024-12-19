import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from diffusers import StableDiffusionInpaintPipeline

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def detect_objects(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    return results

def draw_bounding_boxes(image, results):
    image_np = np.array(image, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # 大きめのフォントサイズ
    thickness = 3  # 太めの線

    for idx, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
        box = [int(coord) for coord in box.tolist()]
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text = f"{idx}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = box[0]
        text_y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 10
        cv2.putText(image_np, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    return image_np

def save_image(image, file_path):
    # Ensure image is in RGB and uint8 format before saving
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(file_path)

def apply_gamma_correction(image, gamma=2.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def inpaint_image(image, mask, inpaint_pipeline):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask).convert("L")
    result = inpaint_pipeline(prompt="background continuation, seamless blend", image=image, mask_image=mask).images[0]
    return np.array(result)

def generate_surrounding_mask_color(image, mask):
    image_np = np.array(image)
    coords = np.column_stack(np.where(mask == 255))
    if len(coords) == 0:
        return image_np

    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)

    surrounding_pixels = image_np[max(0, min_row - 1):max_row + 2, max(0, min_col - 1):max_col + 2]
    average_color = surrounding_pixels.mean(axis=(0, 1)).astype(np.uint8)

    return average_color

def resize_to_original(image, original_size):
    return cv2.resize(image, (original_size[0], original_size[1]), interpolation=cv2.INTER_LANCZOS4)

def blend_images(original_image, inpainted_image, mask):
    blended_image = np.array(original_image).copy()
    blended_image[mask == 255] = inpainted_image[mask == 255]
    return blended_image

def main():
    image_path = "001.jpg"

    if not os.path.exists(image_path):
        print("画像ファイルが見つかりません。")
        return

    print("モデルを読み込んでいます...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to("cuda")

    print("画像を処理しています...")
    image = load_image(image_path)
    original_size = image.size
    image_np = np.array(image)

    # Apply gamma correction to ensure consistent color
    image_np = apply_gamma_correction(image_np)

    results = detect_objects(image, model, processor)

    print("検出結果を描画しています...")
    annotated_image = draw_bounding_boxes(image_np, results)
    output_annotated_path = "annotated_image.jpg"
    save_image(annotated_image, output_annotated_path)
    print(f"物体認識結果を '{output_annotated_path}' に保存しました。画像ビューアで確認してください。")

    selected_indices = input("削除したい物体の番号をカンマ区切りで入力してください（例: 0,2,3）: ")
    selected_indices = [int(idx.strip()) for idx in selected_indices.split(",") if idx.strip().isdigit()]

    print("マスクを生成しています...")
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for selected_idx in selected_indices:
        if selected_idx < 0 or selected_idx >= len(results["boxes"]):
            print(f"無効な番号: {selected_idx}")
            continue

        box = [int(coord) for coord in results["boxes"][selected_idx].tolist()]
        mask[box[1]:box[3], box[0]:box[2]] = 255

    print("マスク部分の色を周辺に合わせています...")
    removed_image = np.array(image_np)
    average_color = generate_surrounding_mask_color(image, mask)
    removed_image[mask == 255] = average_color

    output_removed_path = "removed_image.jpg"
    save_image(removed_image, output_removed_path)
    print(f"選択した物体を削除した画像を '{output_removed_path}' に保存しました。画像ビューアで確認してください。")

    print("画像を補完しています...")
    # 制限付き範囲でのみ補完を実行
    inpainted_image = inpaint_pipeline(
        prompt="background continuation, seamless blend",
        image=Image.fromarray(removed_image),
        mask_image=Image.fromarray(mask).convert("L")
    ).images[0]

    # リサイズして元の画像と一致させる
    inpainted_image_resized = resize_to_original(np.array(inpainted_image), removed_image.shape[:2][::-1])

    # マスク範囲を元に戻しつつ補完
    result_image = blend_images(removed_image, inpainted_image_resized, mask)

    output_result_path = "result_image.jpg"
    save_image(result_image, output_result_path)
    print(f"補完結果を '{output_result_path}' に保存しました。画像ビューアで確認してください。")

if __name__ == "__main__":
    main()
