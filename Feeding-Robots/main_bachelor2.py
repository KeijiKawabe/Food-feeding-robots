# main_feeding_robot_v1.1.py
"""
Meal-Assistance Robot Main Script (v1.1)
----------------------------------------

今回の仕様変更に基づき、以下の流れで動作する。

【最終フロー】
1) ループ外：GPT で CLIP プロンプトを生成（1回のみ）
2) Plate1〜3 の RGB 領域それぞれに対し SAM2 + CLIP で食材ラベルを推定
3) Thermal の決め打ちゾーンを Plate1〜3 に対応づけて温度取得
4) LLM に Plateごとの「食材・温度・食べた回数(food_history)」を入力
     → LLM は "plate_id" と "food_label" の両方を返す
5) 対応する Plate の traj を再生して robot feeding
6) eat_history.append(food_label)
"""

import os
import sys
import cv2
import time
import json
import numpy as np
import pyrealsense2 as rs
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
from xarm.wrapper import XArmAPI

# ---- Local Modules ----
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(THIS_DIR)
sys.path.append(PROJECT_ROOT)

from src.pipeline import PerceptionPipeline
from src.thermal.thermal_gpt_system import ThermalGPTSystem
from src.utils.misc import draw_mask_on_image
from src.planner.task_planner import TaskPlanner
from src.robot.robot_controller import Robotcontroller


# ==============================
# 設定部分
# ==============================

XARM_IP = "192.168.1.199"

plate1_center = (421, 309)
plate2_center = (293, 321)
plate3_center = (378.5, 378)

# Thermal の決め打ちゾーン
THERMAL_ZONES = {
    "Plate1": (20, 50, 110, 140),
    "Plate2": (0, 30, 50, 90),
    "Plate3": (40, 70, 70, 110),
}

SAFE_TEMP_MAX = 65.0

SAM2_CFG = os.path.join(PROJECT_ROOT, "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml")
SAM2_CKPT = os.path.join(PROJECT_ROOT, "..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")

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

def build_manual_clip_prompts() -> Dict[str, Any]:
    """
    CLIP 用のラベル＆プロンプトを手動定義。
    必要に応じてここを編集すればよい。
    """
    return {
    "Yogurt": [
        "a bowl of white yogurt",
        "plain yogurt on a white plate",
        "creamy white yogurt food"
    ],
    "curry": [
        "Japanese brown curry on a plate",
        "a dish of rice with curry sauce",
        "brown curry food"
    ],
    "okayuu": [
        "Japanese rice porridge",
        "a bowl of white rice porridge",
        "okayuu rice porridge food"
    ]
        # 他の食材があれば追加
    }


def build_clip_prompts_with_gpt(client: OpenAI, color_image: np.ndarray) -> Dict[str, Any]:
    """
    （オプション）RGB 画像を GPT-4o に見せて、
    CLIP 用のラベルとプロンプト辞書を生成してもらう。
    うまくいかなくてもシステムが止まらないよう、失敗時は空 dict を返す。
    """
    import base64

    ok, buf = cv2.imencode(".jpg", color_image)
    if not ok:
        print("⚠ 画像エンコード失敗。手動プロンプトを使ってください。")
        return {}

    b64 = base64.b64encode(buf).decode("ascii")
    image_url = f"data:image/jpeg;base64,{b64}"

    prompt = """
You are a vision assistant for a meal-assistance robot.
Look at the plates and list each distinct food type (e.g., "Yogurt", "curry", okayuu")
For each food type, output 2-3 short English phrases that can be used as CLIP text prompts.

Output strictly in JSON:
{
  "labels": ["Yogurt", "Fruit"],
  "clip_prompts": {
    "Yogurt": ["a bowl of yogurt", "natural yogurt"],
    "Fruit": ["a bowl of fruit", "mixed fruit"]
  }
}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=400,
            temperature=0,
        )
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        clip_prompts = data.get("clip_prompts", {})
        print("[CLIP PROMPTS from GPT]", clip_prompts)
        return clip_prompts
    except Exception as e:
        print("⚠ GPT による CLIP プロンプト生成に失敗しました:", e)
        return {}

def init_xarm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)
    return arm


def init_realsense() -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def extract_thermal_region(thermal_data: np.ndarray, plate_id: str):
    y1, y2, x1, x2 = THERMAL_ZONES[plate_id]
    region = thermal_data[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return {
        "min": float(np.min(region)),
        "max": float(np.max(region)),
        "mean": float(np.mean(region)),
    }

def assign_plate_id(center_px):
    cx_food, cy_food = center_px
    min_dist = 1e9
    best_plate_id = None
    plate_centers = [
        {"id": "Plate1", "center": plate1_center},
        {"id": "Plate2", "center": plate2_center},
        {"id": "Plate3", "center": plate3_center},
    ]

    for plate in plate_centers:
        cx_p, cy_p = plate["center"]
        dist = np.linalg.norm(np.array([cx_food - cx_p, cy_food - cy_p]))

        if dist < min_dist:
            min_dist = dist
            best_plate_id = plate["id"]

    return best_plate_id, float(min_dist)



# ==============================
# Main
# ==============================

def main():
    print("=== Meal Assistance Robot v1.1 (food-history version) ===")

    # OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY is not set")
        return
    client = OpenAI(api_key=api_key)

    # Initialize hardware
    arm = init_xarm(XARM_IP)
    rs_pipeline = init_realsense()
    align = rs.align(rs.stream.color)
    thermal_cam = ThermalGPTSystem(api_key)

    # ====== STEP 1: GPT で CLIP プロンプト生成（1回） ======
    # frames = rs_pipeline.wait_for_frames()
    # aligned = align.process(frames)
    # init_color = np.asanyarray(aligned.get_color_frame().get_data())

    # print("\n→ Generating CLIP prompts via GPT...")
    # clip_prompts = build_clip_prompts_with_gpt(client, init_color)
    # print("CLIP PROMPTS:", clip_prompts)
    clip_prompts = build_manual_clip_prompts()
    print("CLIP PROMPTS:", clip_prompts)
    # PerceptionPipeline
    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        device="cuda",
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=False,
    )

    # 食べ物履歴（Plate ではなく food_label）
    eat_history: List[str] = []

    while True:
        cmd = input("\nEnter → 次の一口 / q → 終了: ").strip()
        if cmd == "q":
            break

        # ====== STEP 2: RGB Capture ======
        # ====== STEP 2: RGB 認識 (SAM2 + CLIP) ======

        plate_info = {
            "Plate1": {"food_label": None, "center_px": None},
            "Plate2": {"food_label": None, "center_px": None},
            "Plate3": {"food_label": None, "center_px": None},
        }

        for plate_id, (y1, y2, x1, x2) in PLATE_RGB_ZONES.items():
            crop = color[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"[RGB] {plate_id}: crop empty")
                continue

            rgb_out = pipe.process_frame(crop)

            label = rgb_out.get("label")
            center_local = rgb_out.get("center_px")  # crop 内の座標 (cx, cy)

            if label is None or center_local is None:
                print(f"[RGB] {plate_id}: no food detected")
                continue

            # --- crop 内の中心 → 画像全体の座標に変換 ---
            cx_global = center_local[0] + x1
            cy_global = center_local[1] + y1
            center_global = (cx_global, cy_global)

            # --- 皿中心との距離で nearest plate を決定 ---
            assigned_plate, dist = assign_plate_id(center_global)

            print(f"[RGB] base {plate_id}: detected {label} at {center_global} → nearest={assigned_plate}, dist={dist:.1f}")

            # すでに何か入っている plate に別の結果が被った場合は、
            # 「距離が短い方を採用する」というロジックにしておくと少し頑丈
            current = plate_info.get(assigned_plate, {})
            if current.get("food_label") is None or dist < current.get("dist", 1e9):
                plate_info[assigned_plate] = {
                    "food_label": label,
                    "center_px": center_global,
                    "dist": dist,
                }

        print("\n[Plate Info after RGB]")
        for pid, info in plate_info.items():
            print(f"  {pid}: {info}")


        # ====== STEP 3: Plate1〜3 を SAM2+CLIP で認識 ======
        # --- RGB 認識（全体画像に対して1回だけ） ---
        plate_info = {}

        for plate_id, (y1, y2, x1, x2) in PLATE_RGB_ZONES.items():

            # --- crop ----
            crop = color[y1:y2, x1:x2]
            rgb_out = pipe.process_frame(crop)

            label = rgb_out.get("label")
            center_local = rgb_out.get("center_px")  # crop 内の座標

            if label is None or center_local is None:
                print(f"[RGB] {plate_id}: no food detected")
                plate_info[plate_id] = {
                    "food_label": None,
                    "center_px": None,
                }
                continue

            # --- crop 内中心を global 座標に変換 ---
            cx_global = center_local[0] + x1
            cy_global = center_local[1] + y1
            center_global = (cx_global, cy_global)

            # --- Plate 距離ベースで Plate を判定 ---
            assigned_plate, dist = assign_plate_id(center_global)

            print(f"[RGB] {plate_id}: detected {label} (center={center_global}) → nearest={assigned_plate}, dist={dist:.1f}")

            # --- 保管 ---
            plate_info[plate_id] = {
                "food_label": label,
                "center_px": center_global,
                "assigned_plate": assigned_plate,
                "dist": dist
            }


        # ====== STEP 4: Thermal Capture ======
        tdata, _ = thermal_cam.capture()

        # plate_info は Step3 で plate_info["Plate1"], ["Plate2"], ["Plate3"] を作ってある前提
        for plate_id, info in plate_info.items():

            # --- Thermal region extraction ---
            stats = extract_thermal_region(tdata, plate_id)
            plate_info[plate_id]["temp"] = stats

            # --- 食べた回数は food_label ベースでカウント ---
            food = plate_info[plate_id]["food_label"]
            if food is not None:
                plate_info[plate_id]["times_eaten"] = eat_history.count(food)
            else:
                plate_info[plate_id]["times_eaten"] = 0


        # ====== LLM Input 構造 ======
        plates_for_llm = []
        for pid, info in plate_info.items():
            plates_for_llm.append({
                "plate_id": pid,
                "food_label": info["food_label"],
                "temp": info["temp"],
                "times_eaten": info["times_eaten"],
            })

        # ====== STEP 5: LLM で「どの皿の何を食べるか」決定 ======
        decision = TaskPlanner.decide_next_bite_with_plates_llm(
            client=client,
            plates_info=plates_for_llm,
            safe_temp_max=SAFE_TEMP_MAX,
            history=eat_history,     # ← ここは food_label の履歴
        )

        chosen_plate = decision.get("plate_id")
        chosen_food = decision.get("label")
        reason = decision.get("reason")

        print("\n=== LLM DECISION ===")
        print("Next Plate :", chosen_plate)
        print("Food Label :", chosen_food)
        print("Reason     :", reason)

        if chosen_plate is None or chosen_food is None:
            print("⚠ LLM が決定できませんでした → スキップ")
            continue

        # ====== STEP 6: Robot 動作 ======
        traj = TRAJ_MAP[chosen_plate]
        Robotcontroller.play_traj_for_plate(
            arm=arm,
            traj_scoop_path=traj,
            traj_to_mouth_path=TRAJ_TO_MOUTH,
        )

        # ====== STEP 7: 食べ物履歴に food_label を追加 ======
        eat_history.append(chosen_food)
        print("Updated eat_history:", eat_history)

    # Cleanup
    thermal_cam.cleanup()
    rs_pipeline.stop()
    arm.disconnect()
    print("✓ 終了しました")


if __name__ == "__main__":
    main()
