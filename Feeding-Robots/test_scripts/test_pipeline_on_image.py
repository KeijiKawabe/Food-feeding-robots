import os
import sys
import cv2

# プロジェクトのルートディレクトリを Python パスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import PerceptionPipeline
from src.utils.misc import draw_mask_on_image

ROOT = os.path.dirname(__file__)
IMG  = os.path.join(ROOT, "..", "data", "test_image.jpg")
CFG  = os.path.join(ROOT, "..", "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml")
CKPT = os.path.join(ROOT, "..", "..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")

def main():
    if not os.path.exists(IMG):  raise SystemExit("test_image not found")
    if not os.path.exists(CFG) or not os.path.exists(CKPT):
        raise SystemExit("SAM2 config/ckpt not found")

    pipe = PerceptionPipeline(
        sam2_cfg=CFG, sam2_ckpt=CKPT, device="cuda",
        maskgen_interval=1,  # 画像1枚なので1でOK
        min_area=1000, max_area_frac=0.5,
        clip_model="ViT-B/32"
    )

    bgr = cv2.imread(IMG)
    out = pipe.process_frame(bgr)
    print("[PIPELINE] out:", {k: (v if k!='mask' else f"mask-{v.shape}" if v is not None else None) for k,v in out.items()})

    vis = draw_mask_on_image(bgr.copy(), out["mask"])
    if out["bbox"] is not None:
        x1,y1,x2,y2 = out["bbox"]
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,180,0), 2)
    cv2.imshow("pipeline-on-image", vis)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    print("OK: pipeline single-frame works.")

if __name__ == "__main__":
    main()
