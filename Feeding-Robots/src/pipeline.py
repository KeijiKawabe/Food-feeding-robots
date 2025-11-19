# src/pipeline.py

import time
import numpy as np
from typing import Dict, Any

from .utils.misc import (
    to_rgb, draw_mask_on_image,
    filter_masks_by_area,
    masks_to_crops_and_bboxes
)

from .perception.sam2_wrapper import SAM2Engine
from .perception.clip_scorer import ClipScorer
from .perception.depth_anything_engine import DepthAnythingEngine


class PerceptionPipeline:
    def __init__(self, sam2_cfg, sam2_ckpt,
                 device="cuda",
                 maskgen_interval=5,
                 min_area=1000,
                 max_area_frac=0.5,
                 clip_model="ViT-B/32",
                 enable_depth=False,
                 depth_encoder="vitl",
                 depth_ckpt="checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
                 max_depth=20.0):

        self.device = device

        # SAM2
        self.sam = SAM2Engine(
            sam2_cfg,
            sam2_ckpt,
            device=device,
            points_per_side=8,
            min_mask_region_area=min_area
        )

        # CLIP
        self.clip = ClipScorer(
            device=device,
            model_name=clip_model
        )

        # DepthAnything（任意）
        self.enable_depth = enable_depth
        if self.enable_depth:
            self.depth_engine = DepthAnythingEngine(
                encoder=depth_encoder,
                ckpt_path=depth_ckpt,
                device=device,
                max_depth=max_depth
            )

        self.maskgen_interval = maskgen_interval
        self.min_area = min_area
        self.max_area_frac = max_area_frac

        self.frame_count = 0
        self.last = {
            "mask": None, "bbox": None,
            "label": None, "score": None,
            "depth": None
        }
        self.ema_fps = None

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr) -> Dict[str, Any]:
        t0 = time.time()
        rgb = to_rgb(frame_bgr)
        H, W = rgb.shape[:2]

        need = (self.frame_count % self.maskgen_interval == 0) or (self.last["mask"] is None)

        if need:
            # 1. SAM2 で mask 推定
            self.sam.set_image(rgb)
            masks = filter_masks_by_area(
                self.sam.generate_masks(rgb),
                H, W,
                self.min_area,
                self.max_area_frac
            )

            # 2. CLIP でカテゴリ選択
            crops, bboxes = masks_to_crops_and_bboxes(rgb, masks)
            pick = self.clip.pick_best(crops) if crops else None

            if pick:
                idx = pick["index"]
                bbox = bboxes[idx]
                refined = self.sam.predict_by_bbox(bbox)

                # Depth 取得（任意）
                depth_value = None
                if self.enable_depth:
                    depth_map = self.depth_engine.infer_depth(frame_bgr)
                    depth_value = self.depth_engine.depth_of_mask(depth_map, refined)

                self.last.update(
                    mask=refined,
                    bbox=bbox,
                    label=pick["cls"],
                    score=float(pick["score"]),
                    depth=depth_value
                )
            else:
                self.last.update(mask=None, bbox=None, label=None, score=None, depth=None)

        self.frame_count += 1

        fps = 1.0 / max(time.time() - t0, 1e-6)
        self.ema_fps = fps if self.ema_fps is None else \
            0.9 * self.ema_fps + 0.1 * fps

        return {**self.last, "fps": self.ema_fps}
