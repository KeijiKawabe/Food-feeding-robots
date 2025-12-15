# main_feeding_robot.py
"""
食事介助ロボット・メインスクリプト（v0.1：Plate×Food 対応版）

【このバージョンの方針】

- Hand-eye Calibration（xArm Base ← RealSense）は使わない
- RealSense ⇔ Thermal カメラの幾何キャリブも使わない
- 皿と Thermal / RealSense カメラは「毎回まったく同じ位置・向き」に固定して置く
- RGB (SAM2 + CLIP) で「どの食材か」をラベルとして出す（例: "Yogurt", "Fruit"）
- 食材の中心画素から「どの Plate 上か (Plate1, Plate2, Plate3...)」を判定する
- Thermal は (plate_id, label) に対応したゾーンの温度統計（min/max/mean）を算出する
- LLM には「plate_id, label, temp統計, times_eaten, history」を渡して、
    - 『どの皿から次の一口を取るか』
    - 『今回は feed しないか』
  を決めてもらう
- ロボットは「plate_id に対応する traj（TRAJ_MAP[plate_id]）」を再生して給餌する

将来:
- hand-eye が安定したら move_robot_to_plate() を座標ベース制御に差し替え
- PerceptionPipeline を「複数マスク対応」に拡張すれば、1フレームで複数皿を扱える
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional, List, Tuple

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
from src.thermal.thermal_gpt_system import ThermalGPTSystem
from src.utils.misc import draw_mask_on_image
from src.planner.task_planner import TaskPlanner
from src.robot.robot_controller import Robotcontroller


# ==============================
# 設定項目（環境に合わせて変更）
# ==============================

# xArm の IP
XARM_IP = "192.168.1.199"

# キャリブレーション JSON ファイルのパス
# v0 では hand-eye は使わないが、将来の拡張を見据えて枠だけ残す
CALIB_PATH = os.path.join(PROJECT_ROOT, "calibration", "calib_config.json")

# SAM2 の設定ファイル / 重み
SAM2_CFG = os.path.join(
    PROJECT_ROOT, "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml"
)
SAM2_CKPT = os.path.join(
    PROJECT_ROOT, "..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt"
)

# CLIP プロンプト設定モード
# "manual" : 辞書を手書き指定
# "llm"    : ループ最初に GPT に画像を見せて CLIP 用ラベル/プロンプトを生成（オプション）
PROMPT_MODE = "manual"

# Thermal 側の安全温度しきい値（例：65℃）
SAFE_TEMP_MAX = 65.0


# ==============================
# Plate & Thermal & Traj 定義
# ==============================

# --- RGB 画像上での「各 Plate の位置」（必ず実機で調整する） ---
# 形式: (y1, y2, x1, x2) で color_image[y1:y2, x1:x2] がその皿の領域
# PLATE_RGB_ZONES: Dict[str, Tuple[int, int, int, int]] = {
#     # ↓ ダミー値。実際の画像で (H, W) を見ながら調整すること。
#     "Plate1": (100, 300,  50, 250),
#     "Plate2": (100, 300, 270, 470),
#     "Plate3": (100, 300, 490, 690),
# }
#皿の中心座標をある程度定義する
plate1_center = (421, 309)
plate2_center = (293, 321)
plate3_center = (378.5, 378)
# --- Thermal ゾーン定義（(plate_id, label) ごと） ---
# 皿と Thermal カメラが完全固定されている前提で、
# 各組み合わせ（plate_id, food_label）がどの Thermal 領域に対応しているかを決め打ちする。
#
# 例: ("Plate1", "Yogurt") → ヨーグルトが Plate1 のときの Thermal 矩形
# 実際の PI160 の解像度や皿位置を見て、必ず調整してください。
#
# thermal_data[y1:y2, x1:x2] がその食材の温度領域。
THERMAL_ZONES = {
    # 例: ヨーグルトが Plate1〜3 にある場合のゾーン（ダミー）
    "Plate1": (20, 50,  110,  140),
    "Plate2": (0, 30, 50, 90),
    "Plate3": (40, 70, 70, 110),
    # ("Plate1", "Fruit"): ... など、必要に応じて追加
}

# --- traj ファイル定義（皿ごと） ---
TRAJ_DIR = os.path.join(PROJECT_ROOT, "traj")

# 各皿に対応する「すくい」動作の traj
TRAJ_MAP: Dict[str, str] = {
    "Plate1": os.path.join(TRAJ_DIR, "natural_yog1.traj"),
    "Plate2": os.path.join(TRAJ_DIR, "natural-yog2.traj"),
    "Plate3": os.path.join(TRAJ_DIR, "natural-yog3.traj"),
}

# 口元に運ぶ動作（どの皿でも共通）
TRAJ_TO_MOUTH = os.path.join(TRAJ_DIR, "to_mouth.traj")


# ==============================
# ユーティリティ関数
# ==============================

def load_calibration(path: str) -> Dict[str, Any]:
    """
    キャリブレーション結果を JSON から読み込む。
    v0 では hand-eye を使わないが、将来のために枠として残しておく。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "K_color" in data:
        data["K_color"] = np.asarray(data["K_color"], dtype=np.float32)
    if "T_base_realsense" in data:
        data["T_base_realsense"] = np.asarray(data["T_base_realsense"], dtype=np.float32)
    if "T_realsense_thermal" in data:
        data["T_realsense_thermal"] = np.asarray(data["T_realsense_thermal"], dtype=np.float32)

    return data


def init_xarm(ip: str) -> XArmAPI:
    """
    xArm 本体と接続して「動ける状態」にする初期化関数。

    - motion_enable(True)
    - set_mode(0)  : ポジションモード
    - set_state(0) : Ready 状態
    """
    arm = XArmAPI(ip)
    print(f"[xArm] 接続中... IP={ip}")

    arm.motion_enable(True)
    arm.set_mode(0)   # position mode
    arm.set_state(0)  # ready
    time.sleep(1.0)

    err, warn = arm.get_err_warn_code()
    print(f"[xArm] err={err}, warn={warn}")
    if err != 0:
        print("⚠ xArm にエラーが残っています。GUI で一度クリアしておくと安心です。")

    return arm


def init_realsense() -> rs.pipeline:
    """
    RealSense のパイプラインを開始する。
    color と depth を 640x480@30fps で有効化し、depth を color にアラインする。
    """
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    _ = profile  # suppress unused warning

    print("✓ RealSense スタート")
    return pipeline


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
Look at the plates and list each distinct food type (e.g., "Yogurt", "Fruit").
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


def assign_plate_id(center_px: Optional[Tuple[int, int]]):
    """
    CLIP+SAM2 で得た中心画素 center_px = (x, y) が、
    RGB画像上のどの Plate 矩形内にあるかを判定して plate_id を返す。
    """
    # if center_px is None:
    #     return None

    # cx, cy = center_px
    # for plate_id, (y1, y2, x1, x2) in plate_rgb_zones.items():
    #     if x1 <= cx < x2 and y1 <= cy < y2:
    #         return plate_id
    # return None


    """
    food_center: (cx_food, cy_food)
    plate_centers: List of dict
        [
            {"id": 0, "center": (cx0, cy0)},
            {"id": 1, "center": (cx1, cy1)},
            ...
        ]

    return:
        nearest_plate_id, min_distance
    """

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


def get_thermal_region_for_food(
    plate_id: str,
    food_label: str,
    thermal_data: np.ndarray,
) -> Optional[np.ndarray]:
    """
    (plate_id, food_label) に対応する Thermal ゾーンを返す。
    幾何キャリブは使わず、「皿もカメラも固定」の前提で、
    あらかじめ決めた矩形を切り出すだけ。
    """
    if plate_id not in THERMAL_ZONES:
        print(f"⚠ Thermal ゾーンが定義されていません: plate={plate_id}, label={food_label}")
        return None

    y1, y2, x1, x2 = THERMAL_ZONES[plate_id]

    # 画像サイズにクリップ
    h, w = thermal_data.shape[:2]
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h,     y2))
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w,     x2))

    if x2 <= x1 or y2 <= y1:
        return None

    region = thermal_data[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return region


# ==============================
# メイン処理ループ
# ==============================

def main():
    print("=== Meal-Assistance Robot Main (v0.1: Plate×Food) ===")
    print("Project root:", PROJECT_ROOT)

    # --- OpenAI API キー ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY が環境変数として設定されていません。")
        return
    client = OpenAI(api_key=api_key)

    # --- キャリブ読み込み（今は hand-eye は使わないが、枠として） ---
    try:
        calib = load_calibration(CALIB_PATH)
        print("✓ Calibration loaded from:", CALIB_PATH)
    except FileNotFoundError as e:
        print("⚠ キャリブファイルが見つかりません:", e)
        print("   v0 では hand-eye を使わないので致命的ではありません。")
        calib = {}

    # --- xArm 初期化 ---
    arm = init_xarm(XARM_IP)

    # --- RealSense 初期化 ---
    rs_pipeline = init_realsense()
    align_to = rs.stream.color
    align = rs.align(align_to)

    # --- Thermal GPT: decide next food BEFORE object detection ---
    # 食事履歴（ここでは plate_id ベースで記録）
    eat_history: List[str] = []
    thermal_gpt_system = ThermalGPTSystem(api_key)
    thermal_decision = thermal_gpt_system.decide_next_food(history=eat_history)
    next_food = thermal_decision.get("next_food")
    print("LLM says next_food =", next_food)

    if next_food is None:
        print("⚠ LLM が次の食材を決められませんでした → スキップ")
        return


    # --- PerceptionPipeline (SAM2 + CLIP) 初期化 ---
    if not os.path.exists(SAM2_CFG):
        print("❌ SAM2 config が見つかりません:", SAM2_CFG)
        return
    if not os.path.exists(SAM2_CKPT):
        print("❌ SAM2 checkpoint が見つかりません:", SAM2_CKPT)
        return

    # CLIP プロンプト
    clip_prompts = build_manual_clip_prompts()
    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        device="cuda",
        maskgen_interval=1,
        min_area=0,
        max_area_frac=0.5,
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=True,
    )
    print(clip_prompts)



    try:
        while True:
            # --- ユーザーの Enter 待ち ---
            cmd = input("\n>>> 次の一口を開始するには Enter、終了するには q + Enter: ").strip().lower()
            if cmd == "q":
                print("▶ ユーザー入力により終了します。")
                break

            # --- RealSense から 1フレーム取得 ---
            frames = rs_pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                print("⚠ フレーム取得に失敗しました。スキップします。")
                continue

            depth_image = np.asanyarray(depth_frame.get_data())   # (H, W), uint16, mm
            color_image = np.asanyarray(color_frame.get_data())   # (H, W, 3), BGR
            cv2.imwrite("debug_raw_color.png", color_image)
            print("✓ Saved raw RGB image before SAM2/CLIP: debug_raw_color.png")

            # --- オプション: GPT から CLIP プロンプト生成 ---
            if PROMPT_MODE == "llm":
                auto_prompts = build_clip_prompts_with_gpt(client, color_image)
                if auto_prompts:
                    pipe.update_clip_prompts(auto_prompts)

            # --- RGB パイプライン (SAM2 + CLIP + Depth) ---
            rgb_out = pipe.process_frame(
                color_image,
                depth_frame=depth_image,
                target_label=next_food,   # ← 追加
            )
            # label = rgb_out.get("label")          # 例: "Yogurt"
            bbox = rgb_out.get("bbox")
            center_px = rgb_out.get("center_px")  # (x, y)
            depth_m = rgb_out.get("depth_m")
            fps = rgb_out.get("fps")


            print("\n--- RGB Perception ---")
            print("label :", next_food)
            print("bbox     :", bbox)
            print("center_px:", center_px)
            print("depth_m  :", depth_m)
            print("fps      :", fps)

            # =======================
            # デバッグ：SAM2 マスク
            # =======================
            mask = rgb_out.get("mask")
            if mask is not None:
                vis_mask = draw_mask_on_image(color_image.copy(), mask)
                cv2.imwrite("debug_sam2_mask.png", vis_mask)
                print("✓ SAM2 mask saved: debug_sam2_mask.png")
            else:
                print("⚠ SAM2 mask is None")

            # =======================
            # デバッグ：CLIP に渡された crop
            # =======================
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = color_image[y1:y2, x1:x2]
                cv2.imwrite("debug_clip_crop.png", crop)
                print("✓ CLIP crop saved: debug_clip_crop.png")
            else:
                print("⚠ bbox is None → CLIP crop not saved")

            if label is None or center_px is None:
                            print("⚠ 食材が認識できませんでした。次のループへ。")
                            continue

            # --- 中心位置からどの Plate 上か判定 ---
            plate_id = assign_plate_id(center_px)
            if plate_id is None:
                print("⚠ 中心位置がどの Plate 領域にも属しません。次のループへ。")
                continue

            print(f"Detected food '{next_food} on {plate_id}")

            # --- Thermal から温度取得 ---
            thermal = thermal_gpt_system.capture()
            if thermal is None:
                print("⚠ Thermal 画像取得に失敗。次のループへ。")
                continue

            thermal_data, thermal_img = thermal  # thermal_data: (H, W) 温度 [°C] を想定
            _ = thermal_img  # 今は未使用

            # --- (plate_id, label) に対応する Thermal ゾーンを取得 ---
            region = get_thermal_region_for_food(plate_id, thermal_data)

            if region is None:
                print("⚠ Thermal 側の対応ゾーンが取得できませんでした。次のループへ。")
                continue

            temp_stats = {
                "min": float(np.min(region)),
                "max": float(np.max(region)),
                "mean": float(np.mean(region)),
            }

            print(f"\n--- Thermal stats for {label} on {plate_id} ---")
            print(f"min : {temp_stats['min']:.1f} °C")
            print(f"max : {temp_stats['max']:.1f} °C")
            print(f"mean: {temp_stats['mean']:.1f} °C")

            # --- LLM 用の plates_info を作成（今回は1つだけだが plate_id ベース） ---
            plate_info = {
                "plate_id": plate_id,
                "food_label": next_food,
                "temp": temp_stats,
                "times_eaten": eat_history.count(plate_id),
            }
            plates_info = [plate_info]

            # --- LLM に「どの皿から一口いくか」判定させる ---
            decision = TaskPlanner.decide_next_bite_with_plates_llm(
                client=client,
                plates_info=plates_info,
                safe_temp_max=SAFE_TEMP_MAX,
                history=eat_history,
            )

            choose = decision.get("choose", False)
            chosen_plate_id = decision.get("plate_id", None)
            chosen_label = decision.get("label", None)
            reason = decision.get("reason", "")

            print("\n--- LLM Decision (plate-based) ---")
            print("choose        :", choose)
            print("plate_id      :", chosen_plate_id)
            print("label         :", chosen_label)
            print("reason        :", reason)

            if not choose:
                print("⚠ LLM 判定により、この一口はスキップします。")
                continue

            # LLM 側でも plate_id を返してくるが、今回 plates_info は1つなので、
            # 念のため plate_id が None ならローカル推定を使う
            if chosen_plate_id is None:
                chosen_plate_id = plate_id
            if chosen_label is None:
                chosen_label = next_food

            # --- OK: ロボットを動かす（plate_id ベースで traj 再生） ---
            traj_path = TRAJ_MAP.get(chosen_plate_id)
            if traj_path is None:
                print(f"⚠ plate_id={chosen_plate_id} に対応する traj が TRAJ_MAP にありません。")
                continue

            print(f"▶ Move robot: plate_id={chosen_plate_id}, label={chosen_label}, traj={traj_path}")

            # Robotcontroller 側で traj 再生を行うメソッドを用意しておく想定
            Robotcontroller.play_traj_for_plate(
                arm=arm,
                traj_scoop_path=traj_path,
                traj_to_mouth_path=TRAJ_TO_MOUTH,
            )

            # 食事履歴更新（plate_id ベース）
            eat_history.append(chosen_plate_id)

            # --- デバッグ用に RGB+マスク+BBox を表示 ---
            vis = color_image.copy()
            if rgb_out.get("mask") is not None:
                vis = draw_mask_on_image(vis, rgb_out["mask"])
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Feeding Perception (RGB + SAM2 + CLIP)", vis)
            print("  → ウィンドウに RGB 認識結果を表示しました。何かキーを押すと閉じます。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\n⏹ キーボード割り込みにより終了します。")

    finally:
        # RealSense 停止
        try:
            rs_pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

        # Thermal 停止
        try:
            thermal_system.cleanup()
        except Exception:
            pass

        # xArm 切断
        try:
            arm.disconnect()
            print("✓ xArm 切断")
        except Exception:
            pass

        print("✓ 全てクリーンアップしました。")


if __name__ == "__main__":
    main()
