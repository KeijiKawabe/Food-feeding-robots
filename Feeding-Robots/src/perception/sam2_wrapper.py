import os, sys, numpy as np
from typing import List, Tuple

# Correct the path to the `sam2` folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "sam2"))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2Engine:
    def __init__(self, cfg_path: str, ckpt_path: str, device: str="cuda",
                 points_per_side=8, min_mask_region_area=1000,
                 pred_iou_thresh=0.75, stability_score_thresh=0.9,
                 box_nms_thresh=0.7, crop_n_layers=0):
        self.sam2 = build_sam2(cfg_path, ckpt_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2)
        self.maskgen = SAM2AutomaticMaskGenerator(
            self.sam2,
            points_per_side=points_per_side,
            points_per_batch=128,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_mask_region_area,
        )

    def set_image(self, rgb_np):
        self.predictor.set_image(rgb_np)

    def generate_masks(self, rgb_np):
        return self.maskgen.generate(rgb_np)

    def predict_by_bbox(self, bbox_xyxy) -> np.ndarray:
        arr = np.array(bbox_xyxy, dtype=np.float32)[None, :]
        mask, _, _ = self.predictor.predict(box=arr)  # (1,H,W)
        return mask[0].astype(np.uint8)
