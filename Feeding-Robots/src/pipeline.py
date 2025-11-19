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
# Thermal system (optional)
try:
    from .thermal.thermal_gpt_system import ThermalGPTSystem
except Exception:
    ThermalGPTSystem = None


class PerceptionPipeline:
    def __init__(self, sam2_cfg, sam2_ckpt,
                 device="cuda",
                 maskgen_interval=5,
                 min_area=1000,
                 max_area_frac=0.5,
                 clip_model="ViT-B/32",
                 enable_depth=False,
                 depth_encoder="vitb",
                 depth_ckpt="checkpoints/depth_anything_v2_metric_hypersim_vitb.pth",
                 max_depth=20.0,
                 # --- thermal options ---
                 enable_thermal: bool = False,
                 thermal_api_key: str | None = None,
                 thermal_target_temp: float = 65.0,
                 thermal_save_image: bool = False):

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

        # Thermal（任意） - ThermalGPTSystem を利用して温度取得・分析を行う
        self.enable_thermal = enable_thermal
        self.thermal_target_temp = thermal_target_temp
        self.thermal_save_image = thermal_save_image
        if self.enable_thermal:
            if ThermalGPTSystem is None:
                print("[WARN] ThermalGPTSystem not available; thermal support disabled")
                self.thermal = None
            else:
                try:
                    # thermal_api_key may be None; ThermalGPTSystem can be initialized without it
                    self.thermal = ThermalGPTSystem(openai_api_key=thermal_api_key)
                except Exception as e:
                    print(f"[ERR] Thermal system init failed: {e}")
                    self.thermal = None

        self.maskgen_interval = maskgen_interval
        self.min_area = min_area
        self.max_area_frac = max_area_frac

        self.frame_count = 0
        self.last = {
            "mask": None, "bbox": None,
            "label": None, "score": None,
            "depth": None,
            # thermal: will contain {'stats':{min,max,mean}, 'analysis': str} when run
            "thermal": None
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

    # Note: thermal capture/analysis is not tied to each RGB frame by default.
    # Call `run_thermal_analysis()` to perform one-shot thermal capture + analysis

        self.frame_count += 1

        fps = 1.0 / max(time.time() - t0, 1e-6)
        self.ema_fps = fps if self.ema_fps is None else \
            0.9 * self.ema_fps + 0.1 * fps

        return {**self.last, "fps": self.ema_fps}

    # ------------------------------------------------------------------
    def run_thermal_analysis(self) -> Any:
        """
        Perform one-shot thermal capture and (optional) GPT analysis.
        Stores result in self.last['thermal'] and returns it.
        Returns None if thermal system is not available or capture failed.
        """
        if not self.enable_thermal or getattr(self, 'thermal', None) is None:
            print("[WARN] Thermal support not enabled or not initialized")
            return None

        # Capture thermal data and palette image
        thermal_data, palette_image = self.thermal.capture_thermal_image()
        if thermal_data is None:
            print("[ERR] Thermal capture failed")
            return None

        stats = self.thermal.get_temp_stats(thermal_data)

        # Run GPT analysis if thermal system has analyze_with_gpt
        analysis = None
        try:
            analysis = self.thermal.analyze_with_gpt(palette_image, stats, thermal_data, target_temp=self.thermal_target_temp)
        except Exception as e:
            analysis = f"GPT analysis failed: {e}"

        result = {"stats": stats, "analysis": analysis}
        self.last['thermal'] = result

        # Optionally save palette image
        if self.thermal_save_image and palette_image is not None:
            try:
                import time
                fname = f"thermal_{int(time.time())}.jpg"
                import cv2
                cv2.imwrite(fname, palette_image)
                print(f"[OK] Saved thermal image: {fname}")
            except Exception as e:
                print(f"[WARN] Failed to save thermal image: {e}")

        return result
