# main_feeding_robot_no_thermal.py
"""
食事介助ロボット・メインスクリプト（v1 / 条件B: Thermalなし / ランダムで食材ラベル選択）

構成:
- RealSense から RGB-D を取得
- SAM2 + CLIP で食材のマスク / BBox / ラベルを推定
- 各食材ラベルごとに「最もスコアが高い候補（=ベスト）」を保持
- （変更点）LLMではなくランダム関数で「次に食べるラベル」を決定
- 決定ラベルの座標（center/depth）を引っ張って xArm を制御する
"""

import os
import sys
import time
import json
import random
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

# PROMPT_MODE == "llm" を使う場合のみ OpenAI を使う（キー無し実験を可能にする）
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# --- プロジェクト内モジュール ---
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(THIS_DIR)
sys.path.append(PROJECT_ROOT)

from src.pipeline import PerceptionPipeline
from src.utils.misc import draw_mask_on_image


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

PROMPT_MODE = "manual"   # "manual" or "llm"

DEBUG_CROP_DIR = os.path.join(PROJECT_ROOT, "debug_crops")
os.makedirs(DEBUG_CROP_DIR, exist_ok=True)

# --- ランダム選択の実験用設定 ---
RANDOM_SEED = None          # 再現性が要るなら 0 など固定 / 毎回変えたいなら None
AVOID_IMMEDIATE_REPEAT = False   # 連続同一ラベルを避ける（候補が2つ以上あるときのみ）
MIN_SCORE_TO_ACCEPT = None      # 例: 15.0 など。Noneならスコアで弾かず必ず候補から選ぶ


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

    return data


def CheckIfNewPositionInWorkspace(x, y, z) -> bool:
    if x > 500 or x < 150:
        return False
    if y < -200 or y > 200:
        return False
    if z < 94 or z > 400:
        return False
    return True


def init_xarm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip)
    print(f"[xArm] 接続中... IP={ip}")

    arm.motion_enable(True)
    # sys.argv[0] はスクリプト名、[1] 以降が引数
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        print(f"受け取った引数: {arg}")
        if arg == "debug":
            arm.set_mode(2)
        else: 
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
            "white"
            "a bowl of yogurt with strawberry jam",
            "creamy yogurt with red fruit jam",
            "white yogurt mixed with strawberry jam",
        ],
        "curry source": [
            "thick brown curry sauce",
            "Japanese curry roux sauce",
            "brown curry gravy",
            "curry sauce without rice",
            "brown curry source in the plastic container",
        ],
        "Cone": [
            "sweet kernel corn in the plastic container",
            "a pile of yellow corn kernels",
            "close-up yellow corn kernels",
            "grainy one in the plastic container",
        ],
    }


def build_clip_prompts_with_gpt(client, color_image: np.ndarray) -> Dict[str, Any]:
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
            model="gpt-4o",
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


def get_depth_m_at_center(depth_frame: rs.depth_frame, center_px: Tuple[int, int]) -> Optional[float]:
    if center_px is None:
        return None
    u, v = int(center_px[0]), int(center_px[1])
    try:
        d = float(depth_frame.get_distance(u, v))  # meters
        if d <= 0:
            return None
        return d
    except Exception:
        return None


def save_crop_and_masked(
    out_dir: str,
    label: str,
    score: float,
    color_image: np.ndarray,
    bbox: Optional[list],
    mask: Optional[np.ndarray],
) -> Dict[str, Optional[str]]:
    os.makedirs(out_dir, exist_ok=True)

    if bbox is None:
        return {"crop_path": None, "masked_crop_path": None}

    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(color_image.shape[1], int(x2)); y2 = min(color_image.shape[0], int(y2))
    if x2 - x1 <= 2 or y2 - y1 <= 2:
        return {"crop_path": None, "masked_crop_path": None}

    crop = color_image[y1:y2, x1:x2].copy()
    fname_base = f"{label}_score{score:.3f}_t{int(time.time()*1000)}"

    crop_path = os.path.join(out_dir, fname_base + ".png")
    cv2.imwrite(crop_path, crop)

    masked_path = None
    if mask is not None:
        m = mask[y1:y2, x1:x2]
        if m.dtype != np.uint8:
            m = (m > 0).astype(np.uint8) * 255
        else:
            m = (m > 0).astype(np.uint8) * 255
        masked = cv2.bitwise_and(crop, crop, mask=m)
        masked_path = os.path.join(out_dir, fname_base + "_masked.png")
        cv2.imwrite(masked_path, masked)

    return {"crop_path": crop_path, "masked_crop_path": masked_path}


# ==============================
# （変更点）ランダムで次ラベルを選ぶ
# ==============================

def decide_next_label_random(
    per_label_best: Dict[str, Dict[str, Any]],
    history: list,
    rng: random.Random,
    avoid_immediate_repeat: bool = True,
    min_score_to_accept: Optional[float] = None,
) -> Dict[str, Any]:
    """
    ランダムで次に食べるラベルを決める（Thermalなし / LLMなし）
    戻り値:
      {"allow": bool, "food_label": "<exact label or null>", "reason": str}
    """
    if not per_label_best:
        return {"allow": False, "food_label": None, "reason": "no_candidates"}

    # 候補抽出（必要なら最低スコアでフィルタ）
    labels = list(per_label_best.keys())
    if min_score_to_accept is not None:
        labels = [l for l in labels if float(per_label_best[l].get("score", 0.0)) >= float(min_score_to_accept)]

    if not labels:
        return {"allow": False, "food_label": None, "reason": "all_candidates_below_threshold"}

    # 直前と同じラベルを避ける（候補が複数あるときのみ）
    last = history[-1] if history else None
    if avoid_immediate_repeat and last in labels and len(labels) >= 2:
        labels_wo_last = [l for l in labels if l != last]
        if labels_wo_last:
            labels = labels_wo_last

    chosen = rng.choice(labels)
    return {"allow": True, "food_label": chosen, "reason": "random_choice"}


# ---- ここから下は座標変換・ロボット ----

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
    if P_base is None:
        print("  base coord   : (未計算 / キャリブ未設定)")
        print("====================================")
        return

    x_b, y_b, z_b = P_base
    print(f"  base coord   : x={x_b:.3f}, y={y_b:.3f}, z={z_b:.3f} [m]")
    print("====================================")

    # mm 単位に変換（※あなたの補正 -240 と固定Z=220 はそのまま踏襲）
    x_mm, y_mm, z_mm = x_b * 1000 - 240, y_b * 1000, 210

    if not CheckIfNewPositionInWorkspace(x_mm, y_mm, z_mm + 50):
        print("⚠ Workspace外（approach）。移動中止。")
        move_first_position(arm=arm)
        return
    # arm.set_position(
    #     x_mm, y_mm, z_mm + 50,
    #     roll=135, pitch=0, yaw=90,
    #     speed=50, mvacc=1000, wait=True
    # )
    arm.set_position(
        x_mm, y_mm, z_mm + 50,
        roll=-135, pitch=0, yaw=-90,
        speed=50, mvacc=1000, wait=True
    )
    error = 15
    #error = 4

    if not CheckIfNewPositionInWorkspace(x_mm, y_mm, z_mm):
        print("⚠ Workspace外（target）。移動中止。")
        move_first_position(arm=arm)
        return

    arm.set_position(
        x_mm, y_mm, z_mm - error,
        roll=-135, pitch=0, yaw=-90,
        speed=50, mvacc=1000, wait=True
    )

    if not CheckIfNewPositionInWorkspace(x_mm + 80, y_mm, z_mm):
        print("⚠ Workspace外（push）。移動中止。")
        move_first_position(arm=arm)
        return

    arm.set_position(
        x_mm + 80, y_mm, z_mm - error,
        roll=-135, pitch=0, yaw=-90,
        speed=50, mvacc=1000, wait=True
    )

    arm.set_position(
        x_mm + 80, y_mm, z_mm - error,
        roll=-90, pitch=0, yaw=-90,
        speed=50, mvacc=1000, wait=True
    )


def move_food_to_mouth(arm: XArmAPI):
    arm.set_position(430, 20, 300, -90, 0, -90)

def move_first_position(arm: XArmAPI):
    arm.set_position(360, -100, 320, -90, 0, -90)


def main():
    times = 0
    t_program_start = time.perf_counter()  # ★全体計測 start
    print("=== Meal-Assistance Robot Main (Condition B: No Thermal / RANDOM chooses label) ===")
    print("Project root:", PROJECT_ROOT)

    # PROMPT_MODE が llm のときだけ OpenAI クライアントを用意
    client = None
    if PROMPT_MODE == "llm":
        if OpenAI is None:
            print("❌ openai パッケージが無いので PROMPT_MODE='llm' は使えません。")
            return
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ PROMPT_MODE='llm' ですが OPENAI_API_KEY が設定されていません。")
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
        max_area_frac=0.05,
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=True,
    )

    eat_history = []

    # ランダム生成器（再現性のため固定seed可）
    rng = random.Random()

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
                auto_prompts = build_clip_prompts_with_gpt(client, color_image)  # type: ignore
                if auto_prompts:
                    pipe.update_clip_prompts(auto_prompts)

            rgb_out = pipe.process_frame(color_image, depth_frame=depth_image)
            instances = rgb_out.get("instances", {})
            fps = rgb_out.get("fps")

            if not instances:
                print("⚠ 食材が認識できませんでした。次のループへ。")
                continue

            # --- ラベルごとのベストを構成 ---
            per_label_best: Dict[str, Dict[str, Any]] = {}
            for lbl, inst in instances.items():
                center_px = inst.get("center_px")
                depth_m = inst.get("depth_m")

                # depth_m が取れてない/0なら depth_frame から取得
                if (depth_m is None) or (isinstance(depth_m, (int, float)) and depth_m <= 0):
                    if center_px is not None:
                        d = get_depth_m_at_center(depth_frame, center_px)
                        depth_m = d

                per_label_best[lbl] = {
                    "score": float(inst.get("score", 0.0)),
                    "bbox": inst.get("bbox"),
                    "center_px": center_px,
                    "depth_m": depth_m,
                    "mask": inst.get("mask"),
                }

            print("\n--- RGB Perception (Per-label best) ---")
            for lbl, rec in sorted(per_label_best.items(), key=lambda kv: kv[1]["score"], reverse=True):
                print(f"{lbl:20s} score={rec['score']:.3f} center={rec['center_px']} depth_m={rec['depth_m']} fps={fps}")

            # --- 各ラベルのcrop/マスクcropを保存 ---
            saved_paths = {}
            for lbl, rec in per_label_best.items():
                saved_paths[lbl] = save_crop_and_masked(
                    out_dir=DEBUG_CROP_DIR,
                    label=lbl,
                    score=rec["score"],
                    color_image=color_image,
                    bbox=rec["bbox"],
                    mask=rec.get("mask"),
                )

            # --- （変更点）ランダムで「次に食べるラベル」を決める ---
            decision = decide_next_label_random(
                per_label_best=per_label_best,
                history=eat_history,
                rng=rng,
                avoid_immediate_repeat=AVOID_IMMEDIATE_REPEAT,
                min_score_to_accept=MIN_SCORE_TO_ACCEPT,
            )
            allow = decision.get("allow", False)
            chosen_label = decision.get("food_label", None)
            reason = decision.get("reason", "")

            print("\n--- RANDOM Decision (Choose Label / No Thermal) ---")
            print("allow      :", allow)
            print("food_label :", chosen_label)
            print("reason     :", reason)

            if not allow or not chosen_label:
                print("⚠ 判定により、この一口はスキップします。")
                continue

            # --- ラベルから座標を引っ張る ---
            if chosen_label not in per_label_best:
                print("⚠ chosen_label が per_label_best に無い。最高スコアにフォールバックします。")
                chosen_label, chosen_rec = max(per_label_best.items(), key=lambda kv: kv[1]["score"])
            else:
                chosen_rec = per_label_best[chosen_label]

            bbox = chosen_rec.get("bbox")
            center_px = chosen_rec.get("center_px")
            depth_m = chosen_rec.get("depth_m")

            print("\n--- EXECUTE TARGET (from RANDOM label) ---")
            print("chosen_label:", chosen_label)
            print("bbox       :", bbox)
            print("center_px  :", center_px)
            print("depth_m    :", depth_m)
            print("crop_path  :", saved_paths.get(chosen_label, {}).get("crop_path"))
            print("masked_crop:", saved_paths.get(chosen_label, {}).get("masked_crop_path"))

            if center_px is None or depth_m is None:
                print("⚠ 選択ラベルのcenter/depthが不正。スキップ。")
                continue

            # --- ロボット移動 ---
            move_robot_to_food(
                arm=arm,
                center_px=tuple(center_px) if isinstance(center_px, (list, tuple)) else center_px,
                depth_m=float(depth_m),
                calib=calib,
                label=chosen_label,
            )

            eat_history.append(chosen_label)

            # --- 可視化 ---
            vis = color_image.copy()
            for lbl, inst in instances.items():
                if inst.get("mask") is not None:
                    vis = draw_mask_on_image(vis, inst["mask"])

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"CHOSEN: {chosen_label}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Feeding Perception (Per-label best + RANDOM chosen)", vis)
            print("  → ウィンドウに RGB 認識結果を表示しました。何かキーを押すと閉じます。")
            times += 1
            print(str(times)+"回目の作業です。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            move_food_to_mouth(arm=arm)
            print("初期位置に戻すためにfを押してください")
            cv2.namedWindow("WAIT_KEY", cv2.WINDOW_NORMAL)
            cv2.imshow("WAIT_KEY", np.zeros((80, 420, 3), dtype=np.uint8))

            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('f'):
                    move_first_position(arm=arm)
                    break
                if key == ord('q') or key == 27:
                    print("中断しました。")
                    break

            cv2.destroyWindow("WAIT_KEY")

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
                # ★全体計測 end
        t_program_end = time.perf_counter()
        print(f"[TIME] total_program_time (incl waitKey) = {t_program_end - t_program_start:.3f} sec")
        print(eat_history)


if __name__ == "__main__":
    main()
