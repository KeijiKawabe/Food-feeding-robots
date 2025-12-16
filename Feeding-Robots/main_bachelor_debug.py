# main_bachelor_v1.4.py
"""
Meal-Assistance Robot Main Script (v1.4 + DEBUG XYZ)
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import pyrealsense2 as rs
from typing import Dict, List, Any
from openai import OpenAI
from xarm.wrapper import XArmAPI

# -------- Local Modules --------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(THIS_DIR)
sys.path.append(PROJECT_ROOT)

from src.pipeline import PerceptionPipeline
from src.thermal.thermal_gpt_system import ThermalGPTSystem
from src.planner.task_planner import TaskPlanner
from src.robot.robot_controller import Robotcontroller


# ==============================
# 設定
# ==============================

XARM_IP = "192.168.1.199"

PLATE_CENTERS = {
    "Plate1": (367, 310.5),
    "Plate2": (238, 329),
    "Plate3": (321, 384),
}

THERMAL_ZONES = {
    "Plate1": (20, 50, 110, 140),
    "Plate2": (0, 30, 50, 90),
    "Plate3": (40, 70, 70, 110),
}

SAFE_TEMP_MAX = 65.0

SAM2_CFG = os.path.join(PROJECT_ROOT, "..", "sam2", "sam2", "configs",
                         "sam2.1", "sam2.1_hiera_b+.yaml")
SAM2_CKPT = os.path.join(PROJECT_ROOT, "..", "sam2", "checkpoints",
                          "sam2.1_hiera_base_plus.pt")

TRAJ_DIR = os.path.join(PROJECT_ROOT, "traj")
TRAJ_MAP = {
    "Plate1": os.path.join(TRAJ_DIR, "natural_yog1.traj"),
    "Plate2": os.path.join(TRAJ_DIR, "natural-yog2.traj"),
    "Plate3": os.path.join(TRAJ_DIR, "natural-yog3.traj"),
}

# ==============================
# === DEBUG ADD: Calibration ===
# ==============================

K_COLOR = np.array([
    [608.54150390625, 0.0, 309.4483947753906],
    [0.0, 607.1893920898438, 264.0105285644531],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

T_REALSENSE_BASE = np.array([
    [-0.38002488, -0.66363039,  0.64434104, 0.01422921],
    [-0.91330134,  0.15888117, -0.37501707, 0.45528391],
    [ 0.14649943, -0.73099346, -0.66647392, 0.74867888],
    [ 0.0,         0.0,         0.0,        1.0]
], dtype=np.float32)


# ==============================
# Utility Functions
# ==============================

def build_manual_clip_prompts():
    return {
        "Strawberry Yogurt": [
            "a bowl of yogurt with strawberry jam",
            "creamy yogurt with red fruit jam",
            "white yogurt mixed with strawberry jam",
        ],
        "curry source": [
            "thick brown curry sauce",
            "Japanese curry roux sauce",
            "brown curry gravy",
            "curry sauce without rice",
        ],
        "potato salad": [
            "potato salad with potatoes and mayonnaise",
            "potato salad with carrots",
        ]
    }


def init_xarm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)
    return arm


def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline


def assign_plate(center_px):
    cx, cy = center_px
    best, best_dist = None, 1e9
    for pid, (px, py) in PLATE_CENTERS.items():
        d = np.hypot(cx - px, cy - py)
        if d < best_dist:
            best = pid
            best_dist = d
    return best, best_dist


def extract_thermal_region(tdata, plate_id):
    y1, y2, x1, x2 = THERMAL_ZONES[plate_id]
    region = tdata[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return {
        "min": float(region.min()),
        "max": float(region.max()),
        "mean": float(region.mean()),
    }


# ==============================
# === DEBUG ADD: Geometry ===
# ==============================

def pixel_depth_to_cam_xyz(u, v, depth_m, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float32)


def cam_xyz_to_base_xyz(P_cam, T_cam_base):
    P_h = np.array([P_cam[0], P_cam[1], P_cam[2], 1.0], dtype=np.float32)
    P_base = T_cam_base @ P_h
    return P_base[:3]


def get_score(item: Dict[str, Any]) -> float:
    return float(item.get("score", item.get("prob", item.get("clip_score", 0.0))))


# ==============================
# Main
# ==============================

def main():
    print("=== Meal Assistance Robot v1.4 + DEBUG XYZ ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        return

    client = OpenAI(api_key=api_key)

    arm = init_xarm(XARM_IP)
    rs_pipe = init_realsense()
    align = rs.align(rs.stream.color)
    thermal = ThermalGPTSystem(api_key)

    clip_prompts = build_manual_clip_prompts()
    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        device="cuda",
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=False,
    )

    eat_history: List[str] = []

    while True:
        cmd = input("\nEnter → next bite / q → exit: ").strip()
        if cmd == "q":
            break

        frames = rs_pipe.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            print("⚠ Missing frame")
            continue

        color = np.asanyarray(color_frame.get_data())

        # ===== RGB Recognition =====
        rgb_results = pipe.process_frame_multi(color)

        # ===== DEBUG: Top CLIP → XYZ =====
        TARGET_LABELS = ["Strawberry Yogurt", "curry source"]
        best = {lab: None for lab in TARGET_LABELS}

        for item in rgb_results:
            lab = item.get("label")
            if lab not in best:
                continue
            if best[lab] is None or get_score(item) > get_score(best[lab]):
                best[lab] = item

        print("\n--- DEBUG: Top CLIP candidates ---")
        for lab in TARGET_LABELS:
            item = best[lab]
            if item is None:
                print(f"{lab}: not detected")
                continue

            u, v = map(int, item["center_px"])
            depth_m = depth_frame.get_distance(u, v)
            if depth_m <= 0:
                print(f"{lab}: invalid depth")
                continue

            P_cam = pixel_depth_to_cam_xyz(u, v, depth_m, K_COLOR)
            P_base = cam_xyz_to_base_xyz(P_cam, T_REALSENSE_BASE)

            print(f"{lab}: score={get_score(item):.4f}")
            print(f"  Cam  XYZ [m]: {P_cam}")
            print(f"  Base XYZ [m]: {P_base}")

        # ===== 以降は既存の Plate → Thermal → LLM → Robot =====
        # （元コードそのままなので省略）

    thermal.cleanup()
    rs_pipe.stop()
    arm.disconnect()
    print("✓ Exit")


if __name__ == "__main__":
    main()
