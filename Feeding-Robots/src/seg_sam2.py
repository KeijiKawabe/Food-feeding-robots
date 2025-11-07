# feeding_robot/src/seg_sam2.py
import os, sys, torch
import numpy as np

# SAM2を上の階層からimportできるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), "../../sam2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# === SAM2 設定ファイルと重み ===
# 現在の src から2階層上に sam2 があるので、ここで "../.." を使う
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../sam2/checkpoints/sam2.1_hiera_large.pt")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

# === SAM2モデル初期化 ===
sam2_model = build_sam2(config_file=CONFIG_PATH, checkpoint=CHECKPOINT_PATH).to(device)
predictor = SAM2ImagePredictor(sam2_model)
# Automatic mask generator (for prompt-free mask generation)
amg = SAM2AutomaticMaskGenerator(
    sam2_model,
    points_per_side=256,  # Increase the number of points sampled
    pred_iou_thresh=0.1,  # Minimize IoU threshold for mask filtering
    min_mask_region_area=0  # Allow all mask regions regardless of size
)

# Debug input image properties
def debug_input_image_properties(bgr_img):
    print("[DEBUG] Input image shape:", bgr_img.shape)
    print("[DEBUG] Input image dtype:", bgr_img.dtype)

# Debug annotations
def debug_annotations(anns):
    print("[DEBUG] Annotations:", anns)

# === 画像からマスク生成 ===
# feeding-robots/src/seg_sam2.py

def get_masks(bgr_img):
    """入力画像からマスク候補を生成（3D→2D対応版）"""
    # The automatic mask generator provides prompt-free mask proposals for the whole image.
    # Convert BGR (cv2) -> RGB as SAM2 expects RGB numpy images
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # generate() returns a list of dicts with keys like 'segmentation', 'bbox', 'area', 'predicted_iou'
    anns = amg.generate(rgb)

    debug_input_image_properties(bgr_img)
    debug_annotations(anns)

    print("[DEBUG] Number of annotations generated:", len(anns))
    for i, ann in enumerate(anns):
        print(f"[DEBUG] Annotation {i}: Area={ann.get('area', 0)}, Predicted IoU={ann.get('predicted_iou', 0.0)}")

    out = []
    for i, ann in enumerate(anns):
        seg = ann.get("segmentation")
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()
        seg = np.asarray(seg)
        seg = np.squeeze(seg)
        if seg.size == 0:
            continue
        if seg.ndim != 2:
            print(f"[WARN] Skipping mask {i}: unexpected shape {seg.shape}")
            continue
        seg = seg != 0
        area = int(ann.get("area", seg.sum()))
        if area < 500:
            continue
        ys, xs = np.nonzero(seg)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
        out.append({
            "id": i,
            "mask": seg,
            "bbox": (x, y, w, h),
            "area": area,
            "score": float(ann.get("predicted_iou", 0.0)),
        })
    print(f"[INFO] SAM2 generated {len(out)} masks.")
    return out