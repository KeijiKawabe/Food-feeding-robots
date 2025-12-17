# main_feeding_robot_no_thermal.py
"""
食事介助ロボット・メインスクリプト（v1 / 条件B: Thermalなし）

構成（Thermal以外は v1 と同じ）:
- RealSense から RGB-D を取得
- SAM2 + CLIP で食材のマスク / BBox / ラベルを推定
- GPT に「食材ラベル & 食事履歴」を渡して「次に食べるか」を判断してもらう
- OK なら xArm を動かす関数を呼ぶ（中身は TODO/現状のまま）
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from openai import OpenAI

# --- プロジェクト内モジュール ---
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(THIS_DIR)
sys.path.append(PROJECT_ROOT)

from src.pipeline import PerceptionPipeline
from src.utils.misc import draw_mask_on_image
# TaskPlanner を使うなら import（今回は直接 LLM 判定にする）
# from src.planner.task_planner import TaskPlanner


# ==============================
# 設定
# ==============================

XARM_IP = "192.168.1.199"

CALIB_PATH = os.path.join(PROJECT_ROOT, "calibrations", "calib_config.json")

SAM2_CFG = os.path.join(
    PROJECT_ROOT, "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml"
)
SAM2_CKPT = os.path.join(
    PROJECT_ROOT, "..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt"
)

PROMPT_MODE = "manual"


# ==============================
# ユーティリティ
# ==============================

def load_calibration(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "K_color" in data:
        data["K_color"] = np.asarray(data["K_color"], dtype=np.float32)
    if "T_realsense_base" in data:
        data["T_realsense_base"] = np.asarray(data["T_realsense_base"], dtype=np.float32)
    if "T_realsense_thermal" in data:
        data["T_realsense_thermal"] = np.asarray(data["T_realsense_thermal"], dtype=np.float32)

    return data


def CheckIfNewPositionInWorkspace(x,y,z):
    if x > 500  or x < 300:
        return False
    if y < -200 or y > 300:
        return False
    if z < 94 or z > 400:
        return False
    return True


def init_xarm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip)
    print(f"[xArm] 接続中... IP={ip}")

    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)

    err, warn = arm.get_err_warn_code()
    print(f"[xArm] err={err}, warn={warn}")
    if err != 0:
        print("⚠ xArm にエラーが残っています。GUI で一度クリアしておくと安心です。")
    return arm


def init_realsense() -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("✓ RealSense スタート")
    return pipeline


def build_manual_clip_prompts() -> Dict[str, Any]:
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
        "Cone": [
            "a plate of yellow sweet corn kernels",
            "a pile of glossy yellow corn kernels",
            "yellow corn grains on a white plate",
        ],
    }


def build_clip_prompts_with_gpt(client: OpenAI, color_image: np.ndarray) -> Dict[str, Any]:
    import base64

    ok, buf = cv2.imencode(".jpg", color_image)
    if not ok:
        print("⚠ 画像エンコード失敗。手動プロンプトを使ってください。")
        return {}

    b64 = base64.b64encode(buf).decode("ascii")
    image_url = f"data:image/jpeg;base64,{b64}"

    prompt = """
You are a vision assistant for a meal-assistance robot.
Look at the plate and list each distinct food type.
For each food type, output 2-3 short English phrases that can be used as CLIP text prompts.

Output strictly in JSON:
{
  "clip_prompts": {
    "Yogurt": ["a bowl of yogurt", "natural yogurt"],
    "Curry": ["Japanese curry sauce", "brown curry roux"]
  }
}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt},
                     {"type": "image_url", "image_url": {"url": image_url}},
                 ]}],
            max_tokens=300,
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


def decide_without_thermal_llm(
    client: OpenAI,
    food_label: str,
    history: list,
) -> Dict[str, Any]:
    """
    条件B：温度情報なしで、物体認識ラベルと履歴だけから「食べるか」を決める。
    戻り値: {"allow": bool, "reason": str}
    """
    # 履歴の集計（LLMに “状態” として渡すと安定する）
    from collections import Counter
    counts = dict(Counter(history))
    last = history[-1] if len(history) > 0 else None

    prompt = f"""
You are a task planner for a meal-assistance robot.

The RGB system recognized the current candidate food item label:
"{food_label}"

Eating history (labels): {history}
Times eaten per label: {counts}
Last eaten label: {last}

Decide whether the robot should feed this item now.
Consider variety (avoid repeating the same label too many times in a row),
and avoid feeding if the decision seems unreliable or repetitive.

Output strict JSON only:
{{
  "allow": true/false,
  "reason": "<short English reason>"
}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        return data
    except Exception as e:
        print("⚠ LLM 判定に失敗:", e)
        return {"allow": False, "reason": "LLM_error"}


# ---- ここから下は v1 と同じ（Thermal部分だけ削除） ----

def compute_bbox_center_depth_to_base(center_px, depth_m: float, calib: Dict[str, Any]) -> Optional[np.ndarray]:
    if center_px is None or depth_m is None or depth_m <= 0:
        return None

    K = calib.get("K_color", None)
    T_rb = calib.get("T_realsense_base", None)  # Base ← RealSense

    if K is None or T_rb is None:
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = center_px
    Z = depth_m
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    P_cam = np.array([X, Y, Z, 1.0], dtype=np.float32)

    P_base = T_rb @ P_cam
    return P_base[:3]


def move_robot_to_food(arm: XArmAPI, center_px, depth_m: float, calib: Dict[str, Any], label: str):
    print("")
    print("========== [ROBOT ACTION] ==========")
    print(f"  target food  : {label}")
    print(f"  image center : {center_px}")
    print(f"  depth (m)    : {depth_m}")

    P_base = compute_bbox_center_depth_to_base(center_px, depth_m, calib)
    if P_base is not None:
        x_b, y_b, z_b = P_base
        print(f"  base coord   : x={x_b:.3f}, y={y_b:.3f}, z={z_b:.3f} [m]")
    else:
        print("  base coord   : (未計算 / キャリブ未設定)")

    print("====================================")
    # TODO: arm.set_position() などは現状のまま続けてOK
    if P_base is not None:
        # mm 単位に変換
        x_mm, y_mm, z_mm = x_b * 1000 - 240, y_b * 1000, 220
        CheckIfNewPositionInWorkspace(x_mm, y_mm, z_mm + 50)
        # アプローチ姿勢
        arm.set_position(
            x_mm, y_mm, z_mm + 50,   # 5cm 上から
            roll=-135, pitch=0, yaw=-90,
            speed=50, mvacc=1000,
            wait=True
        )
        CheckIfNewPositionInWorkspace(x_mm, y_mm, z_mm)
        arm.set_position(
            x_mm, y_mm, z_mm,   # 5cm 上から
            roll=-135, pitch=0, yaw=-90,
            speed=50, mvacc=1000,
            wait=True
        )
        CheckIfNewPositionInWorkspace(x_mm + 80, y_mm, z_mm)
        arm.set_position(
            x_mm + 80, y_mm, z_mm,   # 5cm 上から
            roll=-135, pitch=0, yaw=-90,
            speed=50, mvacc=1000,
            wait=True
        )
        CheckIfNewPositionInWorkspace(x_mm + 80, y_mm, z_mm)
        arm.set_position(
            x_mm + 80, y_mm, z_mm,   # 5cm 上から
            roll=-90, pitch=0, yaw=-90,
            speed=50, mvacc=1000,
            wait=True
        )


def move_food_to_mouth(arm: XArmAPI):
    arm.set_position(430, 20, 300, -90, 0, -90)
    return


def main():
    print("=== Meal-Assistance Robot Main (Condition B: No Thermal) ===")
    print("Project root:", PROJECT_ROOT)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY が環境変数として設定されていません。")
        return
    client = OpenAI(api_key=api_key)

    try:
        calib = load_calibration(CALIB_PATH)
        print("✓ Calibration loaded from:", CALIB_PATH)
    except FileNotFoundError as e:
        print("⚠ キャリブファイルが見つかりません:", e)
        calib = {}

    arm = init_xarm(XARM_IP)

    rs_pipeline = init_realsense()
    align = rs.align(rs.stream.color)

    if not os.path.exists(SAM2_CFG) or not os.path.exists(SAM2_CKPT):
        print("❌ SAM2 cfg/ckpt が見つかりません。")
        return

    clip_prompts = build_manual_clip_prompts()
    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        device="cuda",
        maskgen_interval=1,
        min_area=1000,
        max_area_frac=0.5,
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=True,
    )

    eat_history = []

    try:
        while True:
            cmd = input("\n>>> 次の一口を開始するには Enter、終了するには q + Enter: ").strip().lower()
            if cmd == "q":
                print("▶ ユーザー入力により終了します。")
                break

            frames = rs_pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                print("⚠ フレーム取得に失敗しました。スキップします。")
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if PROMPT_MODE == "llm":
                auto_prompts = build_clip_prompts_with_gpt(client, color_image)
                if auto_prompts:
                    pipe.update_clip_prompts(auto_prompts)

            rgb_out = pipe.process_frame(color_image, depth_frame=depth_image)
            instances = rgb_out.get("instances", {})
            fps = rgb_out.get("fps")

            if not instances:
                print("⚠ 食材が認識できませんでした。次のループへ。")
                continue

            label, best = max(instances.items(), key=lambda kv: kv[1]["score"])
            bbox = best["bbox"]
            center_px = best["center_px"]
            depth_m = best["depth_m"]

            print("\n--- RGB Perception ---")
            print("label    :", label)
            print("bbox     :", bbox)
            print("center_px:", center_px)
            print("depth_m  :", depth_m)
            print("fps      :", fps)

            # ★条件B：Thermalなし判定
            decision = decide_without_thermal_llm(
                client=client,
                food_label=label,
                history=eat_history,
            )
            allow = decision.get("allow", False)
            reason = decision.get("reason", "")

            print("\n--- LLM Decision (No Thermal) ---")
            print("allow :", allow)
            print("reason:", reason)

            if not allow:
                print("⚠ LLM 判定により、この一口はスキップします。")
                continue

            move_robot_to_food(
                arm=arm,
                center_px=center_px,
                depth_m=depth_m,
                calib=calib,
                label=label,
            )

            eat_history.append(label)

            # 可視化（元コード踏襲）
            vis = color_image.copy()
            for lbl, inst in instances.items():
                if inst.get("mask") is not None:
                    vis = draw_mask_on_image(vis, inst["mask"])
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Feeding Perception (RGB + SAM2 + CLIP)", vis)
            print("  → ウィンドウに RGB 認識結果を表示しました。何かキーを押すと閉じます。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            move_food_to_mouth(arm=arm)

    finally:
        try:
            rs_pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            arm.disconnect()
            print("✓ xArm 切断")
        except Exception:
            pass
        print("✓ 全てクリーンアップしました。")


if __name__ == "__main__":
    main()
