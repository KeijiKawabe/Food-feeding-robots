# main_feeding_robot.py
"""
食事介助ロボット・メインスクリプト（v1）

構成:
- RealSense から RGB-D を取得
- SAM2 + CLIP で食材のマスク / BBox / ラベルを推定
- Thermal カメラから熱画像を取得
- RGB BBox に対応する Thermal 領域の温度を計算
- GPT に温度 & 食事履歴を渡して「次に食べるか」を判断してもらう
- OK なら xArm を動かす関数を呼ぶ（中身はまだ TODO）

前提:
- src/pipeline.py              : PerceptionPipeline
- src/thermal/thermal_gpt_system.py : ThermalGPTSystem（カメララッパとして使用）
- src/planner/task_planner.py  : TaskPlanner（使ってもいいし、今回は直接 LLM 判定でもよい）
- キャリブ結果を JSON で保存しておき、ここで読み込む
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
from src.thermal.thermal_gpt_system import ThermalGPTSystem
from src.utils.misc import draw_mask_on_image
# TaskPlanner を使うなら import
# from src.planner.task_planner import TaskPlanner


# ==============================
# 設定項目（ここを環境に合わせて変更）
# ==============================

# xArm の IP
XARM_IP = "192.168.1.199"

# キャリブレーション JSON ファイルのパス
CALIB_PATH = os.path.join(PROJECT_ROOT, "calibration", "calib_config.json")
# 例として以下のキーを期待:
# {
#   "K_color": [[fx, 0, cx],[0, fy, cy],[0,0,1]],
#   "T_base_realsense": [[...4x4...]],
#   "T_realsense_thermal": [[...4x4...]]
# }

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
# ユーティリティ関数
# ==============================

def load_calibration(path: str) -> Dict[str, Any]:
    """
    キャリブレーション結果を JSON から読み込む。
    形式は好きに決めて良いが、ここでは例として:
        - "K_color"           : RealSense カラーの内部パラメータ (3x3)
        - "T_base_realsense"  : Base←RealSense の 4x4 行列
        - "T_realsense_thermal": Thermal←RealSense の 4x4 行列
    などを想定。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # numpy 配列に変換しておく
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

    # align オブジェクトは main 内で作る（align_to = rs.stream.color）
    print("✓ RealSense スタート")
    return pipeline


def build_manual_clip_prompts() -> Dict[str, Any]:
    """
    CLIP 用のラベル＆プロンプトを手動定義。
    必要に応じてここを編集すればよい。
    """
    return {
        "rice": ["a plate of white rice", "cooked white rice"],
        "curry": ["a plate of curry", "a bowl of curry"],
        "salad": ["a bowl of salad", "vegetable salad"],
        "soup": ["a bowl of soup"],
    }


def build_clip_prompts_with_gpt(client: OpenAI, color_image: np.ndarray) -> Dict[str, Any]:
    """
    （オプション）RGB 画像を GPT-4o に見せて、
    CLIP 用のラベルとプロンプト辞書を生成してもらう。
    うまくいかなくてもシステムが止まらないよう、失敗時は空 dict を返す。
    """
    import base64

    # 画像を JPEG にエンコードして base64 化
    ok, buf = cv2.imencode(".jpg", color_image)
    if not ok:
        print("⚠ 画像エンコード失敗。手動プロンプトを使ってください。")
        return {}

    b64 = base64.b64encode(buf).decode("ascii")
    image_url = f"data:image/jpeg;base64,{b64}"

    prompt = """
You are a vision assistant for a meal-assistance robot.
Look at the plate and list each distinct food type (e.g., "rice", "curry", "salad").
For each food type, output 2-3 short English phrases that can be used as CLIP text prompts.

Output strictly in JSON:
{
  "labels": ["rice", "curry", "salad"],
  "clip_prompts": {
    "rice": ["a plate of rice", "cooked rice"],
    "curry": ["a plate of curry", "curry rice"],
    ...
  }
}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
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


def compute_bbox_center_depth_to_base(
    center_px,
    depth_m: float,
    calib: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    RGB 画像上の中心画素 + 深度[m] から、
    ロボット base 座標系の 3D 点 (x,y,z) を計算する。

    前提:
        K_color: 3x3 の内パラ
        T_base_realsense: 4x4, Base ← RealSense

    戻り値:
        np.array([x, y, z])  (単位: m)
    """
    if center_px is None or depth_m is None or depth_m <= 0:
        return None

    K = calib.get("K_color", None)
    T_br = calib.get("T_base_realsense", None)  # Base ← RealSense

    if K is None or T_br is None:
        print("⚠ calib に K_color / T_base_realsense がありません。")
        return None

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u, v = center_px
    Z = depth_m  # [m]

    # カメラ座標系 (RealSense color) での 3D 点
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    P_cam = np.array([X, Y, Z, 1.0], dtype=np.float32)

    # Base 座標系へ
    P_base = T_br @ P_cam
    return P_base[:3]


def crop_thermal_region_by_color_bbox(
    bbox_color,
    thermal_data: np.ndarray,
    calib: Dict[str, Any],
) -> np.ndarray:
    """
    RGB 側の BBox に対応する Thermal 領域を切り出す関数の「枠」。
    ここはキャリブの結果に強く依存するので、
    とりあえず簡易版として「画素座標がだいたい揃っている前提」で
    同じ座標範囲をそのまま使う（※実機では要修正）。

    つまり: color の bbox [x1,y1,x2,y2] を Thermal 側でも同じインデックスで切るだけ。
    RealSense ⇔ PI160 のアライメントをちゃんとやったら、ここを書き換える。
    """
    if bbox_color is None:
        return None

    h_t, w_t = thermal_data.shape[:2]
    x1, y1, x2, y2 = bbox_color

    # 範囲を Thermal のサイズにクリップ
    x1_t = max(0, min(w_t - 1, x1))
    x2_t = max(0, min(w_t - 1, x2))
    y1_t = max(0, min(h_t - 1, y1))
    y2_t = max(0, min(h_t - 1, y2))

    if x2_t <= x1_t or y2_t <= y1_t:
        return None

    region = thermal_data[y1_t:y2_t, x1_t:x2_t]
    if region.size == 0:
        return None
    return region


def decide_with_thermal_llm(
    client: OpenAI,
    food_label: str,
    temp_stats: Dict[str, float],
    history: list,
    safe_temp_max: float,
) -> Dict[str, Any]:
    """
    Thermal の温度と履歴から、
    GPT に「この food を食べてよいか」を判定させる。

    temp_stats: {"min": , "max": , "mean": } の dict
    戻り値: {"allow": bool, "reason": str}
    """
    prompt = f"""
You are a task planner for a meal-assistance robot.

It has recognized a food item with label: "{food_label}"
from the RGB camera.

From the thermal camera, we estimated the temperature of this region:

- min temperature: {temp_stats["min"]:.1f} °C
- max temperature: {temp_stats["max"]:.1f} °C
- mean temperature: {temp_stats["mean"]:.1f} °C

The safety upper bound for eating is {safe_temp_max:.1f} °C.
Eating history so far: {history}

Decide whether the robot should feed this food item now.
Output strict JSON only in this format:
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
        print("⚠ Thermal LLM 判定に失敗:", e)
        return {"allow": False, "reason": "LLM_error"}


def move_robot_to_food(
    arm: XArmAPI,
    center_px,
    depth_m: float,
    calib: Dict[str, Any],
    label: str,
):
    """
    LLM + CLIP で決定した「次の一口」の座標をもとに、
    xArm を動かすための入口関数。

    今はまだ「座標を print するだけ」＋「TODO コメント」でOK。
    あとでここに set_position() などを書き足していく。
    """
    print("")
    print("========== [ROBOT ACTION] ==========")
    print(f"  target food  : {label}")
    print(f"  image center : {center_px}")
    print(f"  depth (m)    : {depth_m}")

    # Base 座標を計算（キャリブがあれば）
    P_base = compute_bbox_center_depth_to_base(center_px, depth_m, calib)
    if P_base is not None:
        x_b, y_b, z_b = P_base
        print(f"  base coord   : x={x_b:.3f}, y={y_b:.3f}, z={z_b:.3f} [m]")
    else:
        print("  base coord   : (未計算 / キャリブ未設定)")

    print("  ※ ここで pixel + depth → Base 座標に変換して xArm を動かす")
    print("====================================")

    # --- TODO: 実際のロボット動作をここに実装する ---
    # 例:
    # if P_base is not None:
    #     # mm 単位に変換
    #     x_mm, y_mm, z_mm = x_b * 1000, y_b * 1000, z_b * 1000
    #
    #     # アプローチ姿勢
    #     arm.set_position(
    #         x_mm, y_mm, z_mm + 50,   # 5cm 上から
    #         roll=0, pitch=0, yaw=0,
    #         speed=100, mvacc=1000,
    #         wait=True
    #     )
    #
    #     # 掬い動作など…
    #
    # -------------------------------------------------


# ==============================
# メイン処理ループ
# ==============================

def main():
    print("=== Meal-Assistance Robot Main ===")
    print("Project root:", PROJECT_ROOT)

    # --- OpenAI API キー ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY が環境変数として設定されていません。")
        return
    client = OpenAI(api_key=api_key)

    # --- キャリブ読み込み ---
    try:
        calib = load_calibration(CALIB_PATH)
        print("✓ Calibration loaded from:", CALIB_PATH)
    except FileNotFoundError as e:
        print("⚠ キャリブファイルが見つかりません:", e)
        print("   Base 座標への変換はスキップされます。")
        calib = {}

    # --- xArm 初期化 ---
    arm = init_xarm(XARM_IP)

    # --- RealSense 初期化 ---
    rs_pipeline = init_realsense()
    align_to = rs.stream.color
    align = rs.align(align_to)

    # --- Thermal GPT System（カメララッパとして利用） ---
    thermal_system = ThermalGPTSystem(openai_api_key=api_key, target_temp=SAFE_TEMP_MAX)

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
        min_area=1000,
        max_area_frac=0.5,
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=True,
    )

    eat_history = []  # 例: ["rice", "curry", ...]

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

            # --- オプション: GPT から CLIP プロンプト生成 ---
            if PROMPT_MODE == "llm":
                auto_prompts = build_clip_prompts_with_gpt(client, color_image)
                if auto_prompts:
                    pipe.update_clip_prompts(auto_prompts)

            # --- RGB パイプライン (SAM2 + CLIP + Depth) ---
            rgb_out = pipe.process_frame(color_image, depth_frame=depth_image)
            label = rgb_out.get("label")
            bbox = rgb_out.get("bbox")
            center_px = rgb_out.get("center_px")
            depth_m = rgb_out.get("depth_m")
            fps = rgb_out.get("fps")

            print("\n--- RGB Perception ---")
            print("label    :", label)
            print("bbox     :", bbox)
            print("center_px:", center_px)
            print("depth_m  :", depth_m)
            print("fps      :", fps)

            if label is None or bbox is None:
                print("⚠ 食材が認識できませんでした。次のループへ。")
                continue

            # --- Thermal から温度取得 ---
            thermal = thermal_system.capture()
            if thermal is None:
                print("⚠ Thermal 画像取得に失敗。次のループへ。")
                continue

            thermal_data, thermal_img = thermal  # thermal_data: (H, W) 温度 [°C] を想定

            # --- BBox に対応する Thermal 領域を取得 ---
            region = crop_thermal_region_by_color_bbox(bbox, thermal_data, calib)
            if region is None:
                print("⚠ Thermal 側の対応領域が取得できませんでした。次のループへ。")
                continue

            temp_stats = {
                "min": float(np.min(region)),
                "max": float(np.max(region)),
                "mean": float(np.mean(region)),
            }

            print("\n--- Thermal stats for this food ---")
            print(f"min : {temp_stats['min']:.1f} °C")
            print(f"max : {temp_stats['max']:.1f} °C")
            print(f"mean: {temp_stats['mean']:.1f} °C")

            # --- LLM に「この food を食べてよいか」判定させる ---
            decision = decide_with_thermal_llm(
                client=client,
                food_label=label,
                temp_stats=temp_stats,
                history=eat_history,
                safe_temp_max=SAFE_TEMP_MAX,
            )
            allow = decision.get("allow", False)
            reason = decision.get("reason", "")

            print("\n--- LLM Decision ---")
            print("allow :", allow)
            print("reason:", reason)

            if not allow:
                print("⚠ LLM 判定により、この一口はスキップします。")
                continue

            # --- OK: ロボットを動かす ---
            move_robot_to_food(
                arm=arm,
                center_px=center_px,
                depth_m=depth_m,
                calib=calib,
                label=label,
            )

            # 食事履歴更新
            eat_history.append(label)

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
