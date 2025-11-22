# test_scripts/test_pipeline_on_image.py

import os
import sys
import cv2
import numpy as np

# プロジェクトの src を import できるようにパス追加
ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(PROJECT_ROOT)

from src.pipeline import PerceptionPipeline
from src.utils.misc import draw_mask_on_image


def main():
    # --- テスト用画像パス ---
    # 例: feeding-robots/data/test_image.jpg
    IMG = os.path.join(ROOT, "..", "data", "test_image.jpg")

    # --- SAM2 の設定ファイル/重みのパス ---
    CFG = os.path.join(
        ROOT,
        "..",
        "..",
        "sam2",
        "sam2",
        "configs",
        "sam2.1",
        "sam2.1_hiera_b+.yaml",
    )
    CKPT = os.path.join(
        ROOT,
        "..",
        "..",
        "sam2",
        "checkpoints",
        "sam2.1_hiera_base_plus.pt",
    )

    # パス確認
    if not os.path.exists(IMG):
        print("❌ テスト画像が見つかりません:", IMG)
        print("   例として feeding-robots/data/test_image.jpg を置いてください。")
        return
    if not os.path.exists(CFG):
        print("❌ SAM2 config が見つかりません:", CFG)
        return
    if not os.path.exists(CKPT):
        print("❌ SAM2 checkpoint が見つかりません:", CKPT)
        return

    # --- PerceptionPipeline の初期化 ---
    pipe = PerceptionPipeline(
        sam2_cfg=CFG,
        sam2_ckpt=CKPT,
        device="cuda",          # GPU なしなら "cpu" でもOK（少し遅くなる）
        maskgen_interval=1,     # 静止画1枚なので 1 でOK
        min_area=1000,
        max_area_frac=0.5,
        clip_model="ViT-B/32",
        clip_prompts=None,      # ClipScorer 側のデフォルト or 後で update_clip_prompts() で上書き
        enable_depth=False,     # 画像テストなので depth は使わない
    )

    print("\n=== Testing PerceptionPipeline on Single Image ===")
    print("Image:", IMG)

    # --- 画像読み込み ---
    bgr = cv2.imread(IMG)
    if bgr is None:
        print("❌ 画像を読み込めませんでした:", IMG)
        return

    # --- パイプライン実行（depth_frame は None） ---
    out = pipe.process_frame(bgr, depth_frame=None)

    mask = out["mask"]
    bbox = out["bbox"]
    label = out["label"]
    score = out["score"]
    center_px = out["center_px"]
    depth_m = out["depth_m"]   # enable_depth=False なので None のはず
    fps = out["fps"]

    # コンソール出力
    print("\n=== Pipeline Output ===")
    print("label    :", label)
    print("score    :", score)
    print("bbox     :", bbox)
    print("center_px:", center_px)
    print("depth_m  :", depth_m)
    print("fps(EMA) :", fps)

    # --- 可視化 ---
    vis = bgr.copy()

    # マスクをオーバーレイ
    if mask is not None:
        vis = draw_mask_on_image(vis, mask)

    # BBox を描画
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 中心ピクセルの表示
    if center_px is not None:
        cx, cy = center_px
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

    # テキスト情報
    text_lines = []
    if label is not None:
        text_lines.append(f"Label: {label}")
    if score is not None:
        text_lines.append(f"CLIP score: {score:.2f}")
    if fps is not None:
        text_lines.append(f"FPS(EMA): {fps:.1f}")

    y0 = 20
    for i, txt in enumerate(text_lines):
        y = y0 + i * 18
        cv2.putText(
            vis,
            txt,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ウィンドウ表示
    cv2.imshow("PerceptionPipeline - Single Image", vis)
    print("\nウィンドウに結果を表示しました。何かキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("OK: pipeline single-image test finished.")


if __name__ == "__main__":
    main()
