# src/perception/depth_anything_engine.py
import torch
import cv2
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        
FEEDING_ROOT = os.path.join(BASE_DIR, "..", "..")            
PROJECT_ROOT = os.path.join(FEEDING_ROOT, "..")              
DEPTH_ROOT = os.path.join(PROJECT_ROOT, "Depth-Anything-V2")

if DEPTH_ROOT not in sys.path:
    sys.path.append(DEPTH_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingEngine:
    def __init__(self,
                 encoder="vitl",
                 ckpt_path="metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
                 device="cuda",
                 max_depth=20.0):

        self.device = device
        self.max_depth = max_depth

        print("Initializing DepthAnythingV2...")

        self.model = DepthAnythingV2(
            encoder=encoder,
            features=128,
            out_channels=[96, 192, 384, 768]
        )

        # 正しいパス
        ckpt_full = os.path.join(DEPTH_ROOT, ckpt_path)

        if not os.path.exists(ckpt_full):
            raise FileNotFoundError(f"Depth checkpoint not found: {ckpt_full}")

        state_dict = torch.load(ckpt_full, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print("✓ DepthAnythingV2 loaded.")

    @torch.no_grad()
    def infer_depth(self, bgr_img):
        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        img = img.unsqueeze(0).to(self.device)

        pred = self.model.infer(img)[0]
        depth = pred.cpu().numpy().astype(np.float32)

        return np.clip(depth, 0, self.max_depth)

    def depth_of_mask(self, depth_map, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return float(depth_map[int(ys.mean()), int(xs.mean())])
