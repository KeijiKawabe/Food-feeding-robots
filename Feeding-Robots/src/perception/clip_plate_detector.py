# src/perception/clip_plate_detector.py

import torch
import clip
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1. CLIP の初期化
# =========================
def init_clip_for_plate():
    """
    皿検出用に CLIP モデルとテキスト特徴を準備する。

    Returns:
        model        : CLIP の画像エンコーダ
        preprocess   : CLIP 用の前処理関数
        text_features: 正規化済みテキスト特徴 (num_prompts, D)
    """
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    plate_prompts = [
    "a top-down photo of a round, flat white ceramic plate on a table",
    "a top-down view of an empty round white dinner plate",
    "a round flat white plate seen from above on a dining table",
    
    # 斜め・横からの見え方もカバー
    "a round, flat white ceramic dinner plate on a table, not a bowl",
    "a side view of a round flat white plate used for serving food",
    "not checlboard",
    ]

    with torch.no_grad():
        text_tokens = clip.tokenize(plate_prompts).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, text_features


# =========================
# 2. マスク毎にスコア計算
# =========================
def find_plate_mask(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    model,
    preprocess,
    text_features,
    min_pixels: int = 2000,
) -> Tuple[
    int,
    float,
    Optional[Tuple[int, int, int, int]],
    List[Tuple[int, float]],
    List[Optional[Tuple[int, int, int, int]]],
]:
    """
    画像中の複数マスクから「皿っぽいマスク」を CLIP で 1つ選ぶ。

    Args:
        image_rgb : (H, W, 3) の RGB 画像 (np.uint8)
        masks     : 各マスクは (H, W) の 0/1 or bool の配列のリスト
        model, preprocess, text_features : init_clip_for_plate() の戻り値
        min_pixels : 小さすぎるマスクを無視する閾値（画素数）

    Returns:
        best_idx   : 一番皿スコアが高いマスクのインデックス
        best_score : そのスコア（cos 類似度）
        best_bbox  : (x0, y0, x1, y1) のタプル（なければ None）
        all_scores : 各マスクの (idx, score) のリスト
        all_bboxes : 各マスクの BBox (x0, y0, x1, y1) or None のリスト
    """
    H, W, _ = image_rgb.shape
    pil_img = Image.fromarray(image_rgb)

    all_scores: List[Tuple[int, float]] = []
    all_bboxes: List[Optional[Tuple[int, int, int, int]]] = []

    for i, mask in enumerate(masks):
        # マスクが bool でない場合をケア
        mask_bool = mask.astype(bool)

        # 小さすぎる領域はスキップ
        if mask_bool.sum() < min_pixels:
            all_scores.append((i, -1e9))
            all_bboxes.append(None)
            continue

        # マスクのバウンディングボックスを計算
        ys, xs = np.where(mask_bool)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        bbox = (int(x0), int(y0), int(x1), int(y1))
        all_bboxes.append(bbox)

        # 画像をクロップ（皿が中心になるように）
        crop = pil_img.crop((x0, y0, x1 + 1, y1 + 1))

        # CLIP の前処理
        img_input = preprocess(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(img_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # テキスト特徴との類似度（cos）
            sims = image_features @ text_features.T  # (1, N_prompts)
            score = sims.max().item()  # 複数プロンプトの最大値

        all_scores.append((i, score))

    # 一番スコアが高いマスクを選ぶ
    best_idx, best_score = max(all_scores, key=lambda x: x[1])
    best_bbox = all_bboxes[best_idx]

    return best_idx, best_score, best_bbox, all_scores, all_bboxes


# （任意）単体テストしたいとき用の簡易 main
if __name__ == "__main__":
    # ここは「単体で動かして確認したいとき専用」
    # 実際のパイプラインでは import して使う
    import pathlib

    img_path = "example.jpg"
    if not pathlib.Path(img_path).exists():
        print(f"テスト画像 {img_path} がありません。")
        exit(0)

    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ダミー例として全画素を 1 つのマスクにする（本番では SAM2 のマスクを渡す）
    masks = [np.ones(image_rgb.shape[:2], dtype=np.uint8)]

    model, preprocess, text_feats = init_clip_for_plate()
    best_idx, best_score, best_bbox, all_scores, all_bboxes = find_plate_mask(
        image_rgb, masks, model, preprocess, text_feats
    )

    print("best_idx:", best_idx)
    print("best_score:", best_score)
    print("best_bbox:", best_bbox)
