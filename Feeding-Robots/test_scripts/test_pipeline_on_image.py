# test_scripts/test_pipeline_on_image.py

import os
import sys
import cv2
import numpy as np

# プロジェクトの src をパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import PerceptionPipeline
from src.utils.misc import draw_mask_on_image


def main():

    # --- Thermal用 API KEY ---
    API_KEY = os.getenv("OPENAI_API_KEY")

    # --- SAM2 設定ファイル ---
    SAM2_CFG = "../../sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    SAM2_CKPT = "../../sam2/checkpoints/sam2.1_hiera_base_plus.pt"

    # --- Pipeline 初期化 ---
    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        enable_thermal=True,
        openai_api_key=API_KEY,
        device="cuda",
        maskgen_interval=1,
        min_area=1000,
        max_area_frac=0.5,
        clip_model="ViT-B/32"
    )

    print("\n=== Testing Perception Pipeline on Single Image ===")

    # --- 読み込み画像 ---
    img_path = "test_image.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ 画像を読み込めません: {img_path}")
        return

    # --- パイプライン実行 ---
    out = pipe.process_frame(img)

    print("\n=== Pipeline Output ===")
    for k, v in out.items():
        if k == "mask" and v is not None:
            print(f"mask: shape={v.shape}")
        else:
            print(f"{k}: {v}")

    # -------------------------------
    # 可視化処理（元コードと同じ動作）
    # -------------------------------
    vis = draw_mask_on_image(img.copy(), out["mask"])

    if out["bbox"] is not None:
        x1, y1, x2, y2 = out["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(vis, f"{out['label']} ({out['score']:.1f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 180, 0),
                    2)

    # --- 画像表示 ---
    cv2.imshow("pipeline-on-image", vis)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    print("OK: pipeline single-frame works.")


if __name__ == "__main__":
    main()
