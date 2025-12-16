# main_bachelor_v1.4.py
"""
Meal-Assistance Robot Main Script (v1.4)
----------------------------------------

今回の修正版では以下の問題を完全に改善：

❌ RGB_ZONE を crop して認識 → SAM2 + CLIP が弱くなる → food_label=None 連発
❌ LLM が JSON を返せずパースエラー
❌ Thermal と Plate の対応はあるのに RGB 認識失敗で意味なし

改善ポイント：

✔ RGB 全体画像に対して SAM2 + CLIP を 1 回だけ実行
✔ 各食材候補の bbox center を Plate center と距離比較し Plate を割り当て
✔ Plate1〜3 の food_label が決定 → thermal → LLM
✔ LLM は plate_id & food_label の両方を返す
✔ food_history に food_label を記録
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

# Plate centers (RealSense image pixel coordinates)
PLATE_CENTERS = {
    "Plate1": (367, 310.5),
    "Plate2": (238, 329),
    "Plate3": (321, 384),
}

# Thermal zone per Plate (Fixed)
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

# Robot trajectory files
TRAJ_DIR = os.path.join(PROJECT_ROOT, "traj")
TRAJ_MAP = {
    "Plate1": os.path.join(TRAJ_DIR, "natural_yog1.traj"),
    "Plate2": os.path.join(TRAJ_DIR, "natural-yog2.traj"),
    "Plate3": os.path.join(TRAJ_DIR, "natural-yog3.traj"),
}
TRAJ_TO_MOUTH = os.path.join(TRAJ_DIR, "to_mouth.traj")


# ==============================
# Utility Functions
# ==============================

def build_manual_clip_prompts():
    """CLIP text prompts for the 3 food types."""
    return {
        "Strawberry Yogurt" :[
            "a bowl of yogurt with strawberry jam",
            "creamy yogurt with red fruit jam",
           "white yogurt mixed with strawberry jam",
        ],
        "curry source":[
            "thick brown curry sauce"
            "Japanese curry roux sauce"
            "brown curry gravy"
            "curry sauce without rice"
        ],
        # "okayuu": [
        #     "Japanese rice porridge",
        #     "a bowl of rice porridge",
        #     "okayuu food"
        # ]
        "potato salad":[
           " potato salad with potatoes and mayonnaise",
           " potato salad with carrots",
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
    """食材の中心座標が最も近い Plate を返す"""
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
# Main
# ==============================

def main():
    print("=== Meal Assistance Robot v1.4 (No RGB_ZONE version) ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        return

    client = OpenAI(api_key=api_key)

    # Hardware init
    arm = init_xarm(XARM_IP)
    rs_pipe = init_realsense()
    align = rs.align(rs.stream.color)
    thermal = ThermalGPTSystem(api_key)

    # CLIP prompts
    clip_prompts = build_manual_clip_prompts()
    print("Loaded manual CLIP prompts:", clip_prompts)

    # Perception pipeline
    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        device="cuda",
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=False,
    )

    eat_history: List[str] = []

    # Main loop
    while True:
        cmd = input("\nEnter → next bite / q → exit: ").strip()
        if cmd == "q":
            break

        # ===== STEP1: RealSense capture =====
        frames = rs_pipe.wait_for_frames()
        aligned = align.process(frames)
        color = np.asanyarray(aligned.get_color_frame().get_data())

        # ===== STEP2: RGB 認識 (SAM2 + CLIP 1回だけ) =====
        plate_info = {
            "Plate1": {"food_label": None, "center_px": None},
            "Plate2": {"food_label": None, "center_px": None},
            "Plate3": {"food_label": None, "center_px": None},
        }

        # SAM2 + CLIP → multiple candidates
        rgb_results = pipe.process_frame_multi(color)
        # ※ ここは PerceptionPipeline の format に合わせて実装済み前提
        #   返り値例： [{"label": "Yogurt", "center_px": (x,y), ...}, ...]
        crop_dir = "debug_crops"
        os.makedirs(crop_dir, exist_ok=True)

        timestamp = int(time.time())

        for i, item in enumerate(rgb_results):
            crop = item.get("crop")
            label = item.get("label", "unknown")

            if crop is not None:
                out_path = os.path.join(crop_dir, f"crop_{timestamp}_{i}_{label}.png")
                cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                print(f"[DEBUG] Saved crop: {out_path}")

        for item in rgb_results:
            label = item["label"]
            cx, cy = item["center_px"]

            pid, dist = assign_plate((cx, cy))
            score = item.get("score", item.get("prob", 0.0))
            prev = plate_info[pid]
            if (prev["food_label"] is None) or (dist < prev.get("dist", 1e9)):
                plate_info[pid] = {
                    "food_label": label,
                    "center_px": (cx, cy),
                    "dist": dist,
                    "score": score,
                }

        print("\n--- Plate Info after RGB ---")
        for pid, info in plate_info.items():
            if info["food_label"] is not None:
                # フォーマットして見やすく表示
                print(f"{pid}: {info['food_label']} (Score: {info['score']:.4f}, Dist: {info['dist']:.1f})")
            else:
                print(f"{pid}: None")

        # RGB でなにも検出できないとき
        if all(info["food_label"] is None for info in plate_info.values()):
            print("⚠ All plates: no food detected → skip")
            continue

        # ===== STEP3: Thermal Capture =====
        tdata, _ = thermal.capture()

        for pid, info in plate_info.items():
            stats = extract_thermal_region(tdata, pid)
            plate_info[pid]["temp"] = stats

            food = info["food_label"]
            plate_info[pid]["times_eaten"] = eat_history.count(food) if food else 0

        # LLM にわたす構造
        plates_for_llm = []
        for pid, info in plate_info.items():
            plates_for_llm.append({
                "plate_id": pid,
                "food_label": info["food_label"],
                "temp": info["temp"],
                "times_eaten": info["times_eaten"],
            })

        print("\n--- Plate Info (LLM input) ---")
        print(json.dumps(plates_for_llm, indent=2, ensure_ascii=False))

        # ===== STEP4: LLM Decision =====
        decision = TaskPlanner.decide_next_bite_with_plates_llm(
            client=client,
            plates_info=plates_for_llm,
            safe_temp_max=SAFE_TEMP_MAX,
            history=eat_history,
        )

        pid = decision.get("plate_id")
        food = decision.get("label")

        print("\n=== LLM DECISION ===")
        print("Plate :", pid)
        print("Food  :", food)
        print("Reason:", decision.get("reason"))

        if pid is None or food is None:
            print("⚠ LLM failed → skip")
            continue

        # ===== STEP5: Robot Motion =====
        traj = TRAJ_MAP[pid]
        Robotcontroller.play_traj_file(
            arm=arm,
            traj_path= traj,
        )

        # ===== STEP6: Update eat_history =====
        eat_history.append(food)
        print("eat_history:", eat_history)

    # Cleanup
    thermal.cleanup()
    rs_pipe.stop()
    arm.disconnect()
    print("✓ Exit")


if __name__ == "__main__":
    main()
