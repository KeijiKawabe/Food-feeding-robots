import time
import numpy as np
from typing import Dict, Any

from .utils.misc import (
    to_rgb, draw_mask_on_image, 
    filter_masks_by_area, masks_to_crops_and_bboxes
)
from .perception.sam2_wrapper import SAM2Engine
from .perception.clip_scorer import ClipScorer

# Thermal（あれば読み込む）
try:
    from .thermal.thermal_gpt_system import ThermalGPTSystem
    THERMAL_OK = True
except:
    THERMAL_OK = False


class PerceptionPipeline:
    def __init__(
        self,
        sam2_cfg: str,
        sam2_ckpt: str,
        device="cuda",
        maskgen_interval=10,
        min_area=1000,
        max_area_frac=0.5,
        clip_model="ViT-B/32",
        clip_prompts=None,
        enable_thermal=False,
        openai_api_key=None
    ):
        # SAM2
        self.sam = SAM2Engine(sam2_cfg, sam2_ckpt, device=device,
                              points_per_side=8, min_mask_region_area=min_area)

        # CLIP
        self.clip = ClipScorer(device=device, model_name=clip_model, prompts=clip_prompts)

        # mask settings
        self.maskgen_interval = maskgen_interval
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.frame_count = 0

        # last result
        self.last = {"mask":None, "bbox":None, "label":None, "score":None}
        self.ema_fps = None

        # Thermal
        self.thermal = None
        if enable_thermal and THERMAL_OK:
            self.thermal = ThermalGPTSystem(openai_api_key=openai_api_key)

    def process_frame(self, frame_bgr) -> Dict[str, Any]:
        t0 = time.time()
        rgb = to_rgb(frame_bgr); H,W = rgb.shape[:2]

        need = (self.frame_count % self.maskgen_interval == 0) or (self.last["mask"] is None)

        # --- SAM2 + CLIP (元処理)
        if need:
            self.sam.set_image(rgb)
            masks = filter_masks_by_area(self.sam.generate_masks(rgb), H, W,
                                         self.min_area, self.max_area_frac)
            crops, bboxes = masks_to_crops_and_bboxes(rgb, masks)

            pick = self.clip.pick_best(crops) if crops else None

            if pick:
                bbox = bboxes[pick["index"]]
                refined = self.sam.predict_by_bbox(bbox)
                self.last.update(mask=refined, bbox=bbox,
                                 label=pick["cls"], score=float(pick["score"]))
            else:
                self.last.update(mask=None, bbox=None, label=None, score=None)

        self.frame_count += 1

        # FPS
        fps = 1.0 / max(time.time()-t0, 1e-6)
        self.ema_fps = fps if self.ema_fps is None else 0.9*self.ema_fps + 0.1*fps

        out = {**self.last, "fps": self.ema_fps}

        # --- Thermal 判定（追加部分）
        if self.thermal:
            thermal_data, thermal_image = self.thermal.capture_thermal_image()
            if thermal_data is not None:
                stats = self.thermal.get_temp_stats(thermal_data)
                analysis = self.thermal.analyze_with_gpt(
                    thermal_image, stats, thermal_data
                )
                out["thermal"] = {
                    "stats": stats,
                    "analysis": analysis
                }
            else:
                out["thermal"] = {"stats":None, "analysis":"error"}

        return out
