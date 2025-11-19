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
                 ckpt_path="metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth",
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
    def infer_depth(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        bgr_img: OpenCV の BGR 画像 (H, W, 3)
        戻り値: depth マップ (H, W), float32
        """
        # ★ run.py と同じく infer_image を使う
        #   run.py では raw_image = cv2.imread(...); depth_anything.infer_image(raw_image, input_size)
        self.input_size = 518  # run.py のデフォルト値に合わせる
        depth = self.model.infer_image(bgr_img, self.input_size)  # (H, W) の numpy 配列のはず
        depth = depth.astype(np.float32)

        # max_depth を設定している場合のみクリップ
        if self.max_depth is not None:
            depth = np.clip(depth, 0, self.max_depth)

        return depth
        
    def depth_of_mask(self, depth_map, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return float(depth_map[int(ys.mean()), int(xs.mean())])
