import time, numpy as np
from typing import Dict, Any, Optional
from .utils.misc import to_rgb, draw_mask_on_image, filter_masks_by_area, masks_to_crops_and_bboxes
from .perception.sam2_wrapper import SAM2Engine
from .perception.clip_scorer import ClipScorer

class PerceptionPipeline:
    def __init__(self, sam2_cfg:str, sam2_ckpt:str, device="cuda",
                 maskgen_interval=10, min_area=1000, max_area_frac=0.5,
                 clip_model="ViT-B/32", clip_prompts=None):
        self.sam = SAM2Engine(sam2_cfg, sam2_ckpt, device=device,
                              points_per_side=8, min_mask_region_area=min_area)
        self.clip = ClipScorer(device=device, model_name=clip_model, prompts=clip_prompts)
        self.maskgen_interval = maskgen_interval
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.frame_count = 0
        self.last = {"mask":None, "bbox":None, "label":None, "score":None}
        self.ema_fps = None

    def update_clip_prompts(self, prompts): self.clip.update_prompts(prompts)

    def process_frame(self, frame_bgr) -> Dict[str, Any]:
        t0 = time.time()
        rgb = to_rgb(frame_bgr); H,W = rgb.shape[:2]
        need = (self.frame_count % self.maskgen_interval == 0) or (self.last["mask"] is None)

        if need:
            self.sam.set_image(rgb)                          # 高コスト
            masks = filter_masks_by_area(self.sam.generate_masks(rgb), H, W,
                                         self.min_area, self.max_area_frac)
            crops, bboxes = masks_to_crops_and_bboxes(rgb, masks)
            pick = self.clip.pick_best(crops, thresholds={"rice": 23.0}) if crops else None
            if pick:
                bbox = bboxes[pick["index"]]
                refined = self.sam.predict_by_bbox(bbox)
                self.last.update(mask=refined, bbox=bbox, label=pick["cls"], score=float(pick["score"]))
            else:
                self.last.update(mask=None, bbox=None, label=None, score=None)

        self.frame_count += 1
        fps = 1.0 / max(time.time()-t0, 1e-6)
        self.ema_fps = fps if self.ema_fps is None else 0.9*self.ema_fps + 0.1*fps
        return {**self.last, "fps": self.ema_fps}