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
from typing import Dict, Any, Optional, Tuple, List, Union

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
from src.planner.task_planner import TaskPlanner


# ==============================
# 設定項目（ここを環境に合わせて変更）
# ==============================

# xArm の IP
XARM_IP = "192.168.1.199"

# キャリブレーション JSON ファイルのパス
CALIB_PATH = os.path.join(PROJECT_ROOT, "calibrations", "calib_config.json")
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
SAFE_TEMP_MAX = 50


# ==============================
# ユーティリティ関数
# ==============================

def load_calibration(path: str) -> Dict[str, Any]:
    """
    calib_config.json を読み込み、thermal が指定されていれば npz から
    Thermal intrinsics/extrinsics を読み込んで統合する。

    最終的に返す data には少なくとも以下が入る:
      - K_color (3x3)
      - T_realsense_base (4x4)    [あれば]
      - K_thermal (3x3)
      - dist_thermal (Nx1)
      - T_realsense_thermal (4x4) = Thermal <- RGB
    """
    import os, json
    import numpy as np

    def _make_T(R, t3x1):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t3x1.reshape(3)
        return T

    def _invert_rt(R, t3x1):
        # p_rgb = R_th2rgb * p_th + t_th2rgb の逆
        Rinv = R.T
        tinv = -Rinv @ t3x1
        return Rinv, tinv

    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # JSONの場所を基準に相対パスを解決
    base_dir = os.path.dirname(os.path.abspath(path))
    def _resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

    # --- JSON側のnumpy化 ---
    if "K_color" in data:
        data["K_color"] = np.asarray(data["K_color"], dtype=np.float64)
    if "dist_coeffs" in data:
        data["dist_coeffs"] = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    if "T_realsense_base" in data:
        data["T_realsense_base"] = np.asarray(data["T_realsense_base"], dtype=np.float64)
    if "T_realsense_thermal" in data:
        # もし直書きされてたらそれを優先（npzより優先）
        data["T_realsense_thermal"] = np.asarray(data["T_realsense_thermal"], dtype=np.float64)

    # --- thermal npz を読む ---
    th_cfg = data.get("thermal", None)
    if th_cfg is not None:
        TH_INTR_NPZ = th_cfg.get("intrinsics_npz", None)
        TH_EXT_NPZ  = th_cfg.get("extrinsics_npz", None)
        direction   = th_cfg.get("extrinsics_direction", "th2rgb").lower()

        # intrinsics
        if TH_INTR_NPZ is not None:
            intr_path = _resolve(TH_INTR_NPZ)
            if not os.path.exists(intr_path):
                raise FileNotFoundError(f"✗ thermal intrinsics npz not found: {intr_path}")

            th = np.load(intr_path, allow_pickle=True)
            K_th = th["K"].astype(np.float64)
            dist_th = th["dist"].astype(np.float64).reshape(-1, 1)

            data["K_thermal"] = K_th
            data["dist_thermal"] = dist_th

            # サイズが保存されていれば使う（無ければPI160固定）
            W = int(th["W"]) if "W" in th else 160
            H = int(th["H"]) if "H" in th else 120
            data["thermal_size"] = (W, H)

        # extrinsics
        # ※T_realsense_thermal がJSON直書きで無いときだけ作る
        if TH_EXT_NPZ is not None and ("T_realsense_thermal" not in data):
            ext_path = _resolve(TH_EXT_NPZ)
            if not os.path.exists(ext_path):
                raise FileNotFoundError(f"✗ thermal extrinsics npz not found: {ext_path}")

            ex = np.load(ext_path, allow_pickle=True)
            R_th2rgb = ex["R_th2rgb"].astype(np.float64)
            t_th2rgb = ex["t_th2rgb"].astype(np.float64).reshape(3, 1)

            if direction == "th2rgb":
                # 欲しいのは Thermal <- RGB なので逆にする
                R_rgb2th, t_rgb2th = _invert_rt(R_th2rgb, t_th2rgb)
                data["T_realsense_thermal"] = _make_T(R_rgb2th, t_rgb2th)
            elif direction == "rgb2th":
                # npzが最初から RGB->Thermal を入れている運用の場合
                data["T_realsense_thermal"] = _make_T(R_th2rgb, t_th2rgb)
            else:
                raise ValueError("extrinsics_direction must be 'th2rgb' or 'rgb2th'")

    # 以降の処理でfloat32が良ければここで落としてもOK
    # data["K_color"] = data["K_color"].astype(np.float32) など

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

def CheckIfNewPositionInWorkspace(x,y,z):
    if x > 500  or x < 250:
        return False
    if y < -200 or y > 200:
        return False
    if z < 94 or z > 400:
        return False
    return True



def build_manual_clip_prompts() -> Dict[str, Any]:
    """
    CLIP 用のラベル＆プロンプトを手動定義。
    必要に応じてここを編集すればよい。
    """
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
        "Cone": [
            "a plate of yellow sweet corn kernels",
            "a pile of glossy yellow corn kernels",
            "a yellow corn grains, isolated on white plate",
        ],
    }

def hard_thermal_safety_check(
    rec: Dict[str, Any],
    safe_temp_max: float,
    hot_ratio_max: float = 0.10,   # 「閾値超え画素が5%以下ならOK」など
    require_project: bool = False, # Trueにすると fallback_scale は除外
    min_npts: int = 15,            # 投影点が少ない候補を除外（project時のみ意味あり）
    max = 65,
) -> Tuple[bool, str]:
    """
    候補recが「温度的に安全」かをハード判定する。
    戻り: (safe, reason)
    """
    th = rec.get("thermal", {})
    if not th.get("ok", False):
        return False, "thermal not available"

    mode = th.get("mode", None)
    npts = int(th.get("npts", 0))

    if require_project and mode != "project":
        return False, f"mode={mode} rejected"

    if mode == "project" and npts < min_npts:
        return False, f"too few projected points (npts={npts})"

    p95 = th.get("p95", None)
    hot_ratio = th.get("hot_ratio", None)
    tmax = th.get("max", None)
    if tmax is not None and float(tmax) >= safe_temp_max + 10.0:
        return False, f"tmax={tmax:.1f} >= safe+10"


    if p95 is None or hot_ratio is None:
        return False, "missing p95/hot_ratio"

    # ハード判定（ここをあなたの安全基準に合わせて調整）
    # if float(p95) > safe_temp_max:
    #     return False, f"p95={p95:.1f} > safe={safe_temp_max:.1f}"
    if float(hot_ratio) > hot_ratio_max:
        return False, f"hot_ratio={hot_ratio:.2f} > {hot_ratio_max:.2f}"

    return True, "safe"


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
        T_realsense_base: 4x4, Base ← RealSense

    戻り値:
        np.array([x, y, z])  (単位: m)
    """
    if center_px is None or depth_m is None or depth_m <= 0:
        return None

    K = calib.get("K_color", None)
    T_rb = calib.get("T_realsense_base", None)  # Base ← RealSense

    if K is None or T_rb is None:
        print("⚠ calib に K_color / T_realsense_base がありません。")
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
    P_base = T_rb @ P_cam
    return P_base[:3]

def crop_thermal_region_by_color_bbox(
    bbox_color,
    thermal_data: np.ndarray,      # (H_th, W_th) 温度[C]（raw）
    calib: Dict[str, Any],
    depth_u16: np.ndarray,         # (H, W) aligned depth [mm] uint16
    sample_step: int = 8,
    min_valid_points: int = 10,
    margin: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]], str, int]:
    """
    RGB bbox + depth から Thermal(raw) 上の bbox を推定して crop する。
    戻り値:
      region: thermal_data の切り出し
      bbox_t: (x1_t, y1_t, x2_t, y2_t) in thermal image coordinates
      mode  : "project" or "fallback_scale"
      npts  : 投影に使えた点数
    """
    if bbox_color is None or thermal_data is None or depth_u16 is None:
        return None, None, "none", 0

    Kc = calib.get("K_color", None)
    Trt = calib.get("T_realsense_thermal", None)   # Thermal <- RGB
    Kt = calib.get("K_thermal", None)
    dist_t = calib.get("dist_thermal", None)

    if Kc is None or Trt is None or Kt is None or dist_t is None:
        print("⚠ calib keys missing (need K_color, T_realsense_thermal, K_thermal, dist_thermal)")
        return None, None, "missing_calib", 0

    h_th, w_th = thermal_data.shape[:2]
    H, W = depth_u16.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in bbox_color]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None, "bad_bbox", 0

    fx, fy = float(Kc[0, 0]), float(Kc[1, 1])
    cx, cy = float(Kc[0, 2]), float(Kc[1, 2])

    pts_uv = []

    # bbox内を格子状にサンプルして投影
    for v in range(y1, y2, sample_step):
        for u in range(x1, x2, sample_step):
            d_mm = int(depth_u16[v, u])
            if d_mm <= 0:
                continue
            Z = d_mm * 0.001

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            P_rs = np.array([X, Y, Z, 1.0], dtype=np.float64)

            P_th = Trt @ P_rs
            Xth, Yth, Zth = float(P_th[0]), float(P_th[1]), float(P_th[2])
            if Zth <= 1e-6:
                continue

            obj = np.array([[Xth, Yth, Zth]], dtype=np.float64)
            img_pts, _ = cv2.projectPoints(
                objectPoints=obj,
                rvec=np.zeros((3, 1), dtype=np.float64),
                tvec=np.zeros((3, 1), dtype=np.float64),
                cameraMatrix=Kt.astype(np.float64),
                distCoeffs=dist_t.astype(np.float64),
            )
            uu, vv = img_pts.reshape(-1)
            pts_uv.append((uu, vv))

    npts = len(pts_uv)

    # フォールバック：スケールで雑に
    if npts < min_valid_points:
        sx = w_th / float(W)
        sy = h_th / float(H)
        x1_t = int(np.clip(x1 * sx, 0, w_th - 1))
        x2_t = int(np.clip(x2 * sx, 0, w_th - 1))
        y1_t = int(np.clip(y1 * sy, 0, h_th - 1))
        y2_t = int(np.clip(y2 * sy, 0, h_th - 1))
        if x2_t <= x1_t or y2_t <= y1_t:
            return None, None, "fallback_scale", npts

        region = thermal_data[y1_t:y2_t, x1_t:x2_t]
        bbox_t = (x1_t, y1_t, x2_t, y2_t)
        return (region if region.size > 0 else None), bbox_t, "fallback_scale", npts

    us = np.array([p[0] for p in pts_uv], dtype=np.float64)
    vs = np.array([p[1] for p in pts_uv], dtype=np.float64)

    x1_t = int(np.floor(us.min())) - margin
    x2_t = int(np.ceil (us.max())) + margin
    y1_t = int(np.floor(vs.min())) - margin
    y2_t = int(np.ceil (vs.max())) + margin

    x1_t = max(0, min(w_th - 1, x1_t))
    x2_t = max(0, min(w_th - 1, x2_t))
    y1_t = max(0, min(h_th - 1, y1_t))
    y2_t = max(0, min(h_th - 1, y2_t))

    if x2_t <= x1_t or y2_t <= y1_t:
        return None, None, "project", npts

    region = thermal_data[y1_t:y2_t, x1_t:x2_t]
    bbox_t = (x1_t, y1_t, x2_t, y2_t)
    return (region if region.size > 0 else None), bbox_t, "project", npts


def attach_thermal_to_per_label_best(
    per_label_best: Dict[str, Dict[str, Any]],
    thermal_data: np.ndarray,   # (Hth,Wth) 温度[C] raw
    calib: Dict[str, Any],
    depth_u16: np.ndarray,      # aligned depth (H,W) uint16 mm
    sample_step: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """
    per_label_best[lbl]["thermal"] に以下を入れる:
      {
        "ok": bool,
        "bbox_t": [x1,y1,x2,y2] or None,
        "min": float, "max": float, "mean": float,
        "p95": float, "hot_ratio": float,
        "mode": str, "npts": int
      }
    """
    for lbl, rec in per_label_best.items():
        bbox = rec.get("bbox", None)
        if bbox is None:
            rec["thermal"] = {"ok": False, "bbox_t": None, "mode": "no_bbox", "npts": 0}
            continue

        region, bbox_t, mode, npts = crop_thermal_region_by_color_bbox(
            bbox_color=bbox,
            thermal_data=thermal_data,
            calib=calib,
            depth_u16=depth_u16,
            sample_step=sample_step,
        )

        if region is None or bbox_t is None:
            rec["thermal"] = {"ok": False, "bbox_t": None, "mode": mode, "npts": int(npts)}
            continue

        # stats（分布も）
        r = region.astype(np.float64)
        tmin = float(np.min(r))
        tmax = float(np.max(r))
        tmean = float(np.mean(r))
        p95 = float(np.percentile(r, 95))
        # “しきい値以上の割合” を特徴量として渡す（閾値で弾くのではなく）
        safe = float(calib.get("safe_temp_max", 50))  # 後で main で入れると楽
        hot_ratio = float(np.mean(r > safe))

        rec["thermal"] = {
            "ok": True,
            "bbox_t": [int(bbox_t[0]), int(bbox_t[1]), int(bbox_t[2]), int(bbox_t[3])],
            "min": tmin, "max": tmax, "mean": tmean,
            "p95": p95,
            "hot_ratio": hot_ratio,
            "mode": mode,
            "npts": int(npts),
        }
    return per_label_best



import json
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI

def choose_label_llm_grouped_2to3(
    client: OpenAI,
    safe_candidates: Dict[str, Dict[str, Any]],
    history: List[Dict[str, Any]],
    close_thr_c: float = 3.0,
    target_streak: int = 3,
    switch_min_delta_c: float = 4.0,   # ★切替は最低これくらい離れてほしい（なければ緩和）
    cold_thr: float = 25.0,
    hot_thr: float = 35.0,             # ★今回の温度帯だと40は高すぎる可能性大
) -> Dict[str, Any]:

    if not safe_candidates:
        return {"food_label": None, "reason": "no safe candidates"}

    # last temp
    last_temp = None
    last_label = None
    if history:
        last_label = history[-1].get("label")
        try:
            last_temp = float(history[-1].get("temp_mean_c", None))
        except Exception:
            last_temp = None

    streak_len = compute_temp_streak_len_adjacent(history, close_thr_c=close_thr_c)

    # build summary
    items = []
    for lbl, rec in safe_candidates.items():
        th = rec.get("thermal", {}) or {}
        t_mean = th.get("mean", None)
        try:
            t_mean = float(t_mean) if t_mean is not None else None
        except Exception:
            t_mean = None

        delta = None
        if last_temp is not None and t_mean is not None:
            delta = abs(t_mean - last_temp)

        items.append({
            "label": lbl,
            "temp_mean_c": t_mean,
            "bucket": temp_bucket(t_mean, cold_thr=cold_thr, hot_thr=hot_thr) if t_mean is not None else "unknown",
            "delta_to_last_c": delta,
        })

    allowed_all = [x["label"] for x in items]

    # ---- ★ここが肝：切替フェーズなら “切替候補” だけに絞る ----
    preferred = allowed_all
    if last_temp is not None and streak_len >= target_streak:
        last_bucket = temp_bucket(last_temp, cold_thr=cold_thr, hot_thr=hot_thr)

        # まず bucket を変える候補（unknown除外）
        switch_pool = [
            x for x in items
            if x["temp_mean_c"] is not None
            and x["bucket"] != "unknown"
            and x["bucket"] != last_bucket
        ]

        # さらに “十分離れてる” を優先
        switch_pool2 = [x for x in switch_pool if x["delta_to_last_c"] is not None and x["delta_to_last_c"] >= switch_min_delta_c]

        if switch_pool2:
            preferred = [x["label"] for x in switch_pool2]
        elif switch_pool:
            preferred = [x["label"] for x in switch_pool]
        else:
            # 切替候補がないなら、せめて delta が大きい順で上位を優先
            with_delta = [x for x in items if x["delta_to_last_c"] is not None]
            with_delta.sort(key=lambda x: x["delta_to_last_c"], reverse=True)
            if with_delta:
                preferred = [x["label"] for x in with_delta[:2]]  # 上位2つだけに絞る（強いソフト）

    # prompt
    prompt = f"""
You are a task planner for a meal-assistance robot.
All candidates are SAFE.

We want temperature grouping: take similar temperature for about 2-3 picks.
If streak_len >= {target_streak}, SWITCH temperature group (prefer a different bucket or larger delta).
This is a strong preference.

Candidates:
{json.dumps(items, ensure_ascii=False)}

History(last temp): {last_temp}
Streak_len: {streak_len}

Allowed labels (you MUST pick one):
{preferred}

Output STRICT JSON only:
{{
  "food_label": "<one of allowed labels>",
  "reason": "<short English reason>"
}}
"""

    # schema enum を preferred にする（ここで実質 “切替” を強制に近づける）
    schema = {
        "type": "object",
        "properties": {
            "food_label": {"type": "string", "enum": preferred},
            "reason": {"type": "string"},
        },
        "required": ["food_label", "reason"],
        "additionalProperties": False
    }

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": prompt}],
            text={"format": {"type": "json_schema", "name": "choose_label_grouped", "strict": True, "schema": schema}},
        )
        return json.loads(resp.output_text)
    except Exception as e:
        # fallback：preferred 内からランダムではなく「目的に合う」選び方
        # 切替フェーズなら delta最大、通常は delta最小
        cand = [x for x in items if x["label"] in preferred]
        if last_temp is not None:
            if streak_len >= target_streak:
                cand2 = [x for x in cand if x["delta_to_last_c"] is not None]
                best = max(cand2, key=lambda x: x["delta_to_last_c"]) if cand2 else cand[0]
            else:
                cand2 = [x for x in cand if x["delta_to_last_c"] is not None]
                best = min(cand2, key=lambda x: x["delta_to_last_c"]) if cand2 else cand[0]
        else:
            best = cand[0]
        return {"food_label": best["label"], "reason": f"fallback ({e.__class__.__name__})"}






def make_thermal_debug_view(thermal_data: np.ndarray) -> np.ndarray:
    """
    thermal_data (float, °C) を 8bitグレースケールに正規化して表示用画像を作る
    """
    td = thermal_data.astype(np.float32)
    lo, hi = np.percentile(td, 2), np.percentile(td, 98)
    if hi <= lo:
        hi = lo + 1.0
    img8 = np.clip((td - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    return vis



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
    if P_base is not None:
        # mm 単位に変換
        x_mm, y_mm, z_mm = x_b * 1000 - 240, y_b * 1000, 210
        error = 15
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
            x_mm, y_mm, z_mm - error,
            roll=-135, pitch=0, yaw=-90,
            speed=50, mvacc=1000,
            wait=True
        )
        CheckIfNewPositionInWorkspace(x_mm + 80, y_mm, z_mm)
        arm.set_position(
            x_mm + 80, y_mm, z_mm - error,
            roll=-135, pitch=0, yaw=-90,
            speed=50, mvacc=1000,
            wait=True
        )
        CheckIfNewPositionInWorkspace(x_mm + 80, y_mm, z_mm)
        arm.set_position(
            x_mm + 80, y_mm, z_mm - error,
            roll=-90, pitch=0, yaw=-90,
            speed=50, mvacc=1000,
            wait=True
        )


    
    #     # 掬い動作など…
    #
    # -------------------------------------------------

def move_food_to_mouth(arm:XArmAPI):
    arm.set_position(430, 20, 300, -90, 0, -90)
    return


def move_first_position(arm: XArmAPI):
    arm.set_position(360, -100, 320, -90, 0, -90)
    return

from typing import List, Dict, Any, Optional, Tuple
import math

def history_last_label_and_temp(history: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
    if not history:
        return None, None
    last = history[-1]
    lbl = last.get("label", None)
    t = last.get("temp_mean_c", None)
    try:
        t = float(t) if t is not None else None
    except Exception:
        t = None
    return (lbl if isinstance(lbl, str) else None), t

def temp_bucket(t_c: Optional[float], cold_thr: float = 25.0, hot_thr: float = 45.0) -> str:
    """
    cold: < cold_thr
    warm: [cold_thr, hot_thr]
    hot : > hot_thr
    """
    if t_c is None or (isinstance(t_c, float) and (math.isnan(t_c) or math.isinf(t_c))):
        return "unknown"
    elif t_c < cold_thr:
        return "cold"
    elif t_c <= hot_thr:
        return "warm"
    else:
        return "hot"

def compute_temp_streak_len_adjacent(history, close_thr_c=3.0) -> int:
    temps = []
    for h in history:
        t = h.get("temp_mean_c", None)
        try:
            t = float(t) if t is not None else None
        except Exception:
            t = None
        temps.append(t)

    if not temps or temps[-1] is None:
        return 0

    streak = 1
    for i in range(len(temps) - 1, 0, -1):
        a = temps[i]
        b = temps[i - 1]
        if a is None or b is None:
            break
        if abs(a - b) <= close_thr_c:
            streak += 1
        else:
            break
    return streak




# ==============================
# メイン処理ループ
# ==============================

def main():
        # === 計測開始（Enter押してこの周が始まった瞬間）===
    t_program_start = time.perf_counter()  # ★全体計測 start
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

            instances = rgb_out.get("instances", {})
            fps = rgb_out.get("fps")

            if not instances:
                print("⚠ 食材が認識できませんでした。次のループへ。")
                continue

            # --- Thermal から温度取得（A案でも必須） ---
            thermal = thermal_system.capture()
            if thermal is None:
                print("⚠ Thermal 画像取得に失敗。次のループへ。")
                continue

            thermal_data, thermal_img = thermal  # thermal_data: (Hth,Wth) float(°C)


                        # スコア最大の食材を1つ選ぶ
            # 1) per_label_best を作る（no_thermalと同じ形）
            per_label_best = {}
            for lbl, inst in instances.items():
                per_label_best[lbl] = {
                    "score": float(inst.get("score", 0.0)),
                    "bbox": inst.get("bbox"),
                    "center_px": inst.get("center_px"),
                    "depth_m": inst.get("depth_m"),
                    "mask": inst.get("mask"),
                }

            # 2) safe_temp_max を calib に入れておくと attach が楽
            calib["safe_temp_max"] = SAFE_TEMP_MAX

            # 3) thermal stats を付与
            per_label_best = attach_thermal_to_per_label_best(
                per_label_best=per_label_best,
                thermal_data=thermal_data,
                calib=calib,
                depth_u16=depth_image,
                sample_step=8,
            )

            # ---- ハード安全フィルタ：安全候補だけ残す ----
            safe_candidates = {}
            unsafe_reasons = {}

            for lbl, rec in per_label_best.items():
                safe, why = hard_thermal_safety_check(
                    rec,
                    safe_temp_max=SAFE_TEMP_MAX,
                    hot_ratio_max=0.10,
                    require_project=False,  # 安全寄りなら True
                    min_npts=15,
                    max = 65
                )
                rec.setdefault("thermal", {})["hard_safe"] = safe
                rec["thermal"]["hard_safe_reason"] = why

                if safe:
                    safe_candidates[lbl] = rec
                else:
                    unsafe_reasons[lbl] = why

            print("[HARD SAFETY] unsafe excluded:", unsafe_reasons)
            print("[HARD SAFETY] safe labels:", list(safe_candidates.keys()))

            if len(safe_candidates) == 0:
                print("⚠ 安全な候補が無いので停止（全候補が高温/不確実）")
                continue
            # =========================
            # 5) HARD+LLM Decision（ここが質問のブロック）
            # =========================
            try:
                sel = choose_label_llm_grouped_2to3(
                        client=client,
                        safe_candidates=safe_candidates,
                        history=eat_history,
                        close_thr_c=3.0,
                        target_streak=3,
                )
                chosen_label = sel["food_label"]
                reason = sel["reason"]
            except Exception as e:
                print("⚠ choose_label_llm_from_safe failed:", e)
                chosen_label = max(safe_candidates.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
                reason = "fallback: choose highest RGB score among safe candidates"

            allow = True  # safe_candidatesがある時点で True
            print("\n--- HARD+LLM Decision ---")
            print("allow      :", allow)
            print("food_label :", chosen_label)
            print("reason     :", reason)

            # ここから先は chosen を「safe_candidates」から取る（重要）
            chosen = safe_candidates[chosen_label]
            bbox = chosen["bbox"]
            center_px = chosen["center_px"]
            depth_m = chosen["depth_m"]
            th = chosen.get("thermal", {})
            bbox_t = th.get("bbox_t", None)

            # 5) Thermal raw のデバッグ表示（bbox_t）
            th_vis = make_thermal_debug_view(thermal_data)
            if bbox_t is not None:
                x1,y1,x2,y2 = bbox_t
                cv2.rectangle(th_vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(th_vis, f"{chosen_label}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.imshow("THERMAL RAW + projected bbox", th_vis)
            cv2.waitKey(0)
            cv2.destroyWindow("THERMAL RAW + projected bbox")

            # --- OK: ロボットを動かす ---
            move_robot_to_food(
                arm=arm,
                center_px=center_px,
                depth_m=depth_m,
                calib=calib,
                label=chosen_label,
            )
            chosen_th = chosen.get("thermal", {}) or {}
            t_mean = chosen_th.get("mean", None)
            try:
                t_mean = float(t_mean) if t_mean is not None else None
            except Exception:
                t_mean = None

            eat_history.append({
                "label": chosen_label,
                "temp_mean_c": t_mean,
            })



            # --- デバッグ用に RGB+マスク+BBox を表示 ---
            vis = color_image.copy()
            for chosen_label, inst in instances.items():
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
            print("初期位置に戻すためにfを押してください")
            cv2.namedWindow("WAIT_KEY", cv2.WINDOW_NORMAL)
            cv2.imshow("WAIT_KEY", np.zeros((80, 420, 3), dtype=np.uint8))

            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('f'):
                    move_first_position(arm=arm)
                    break
                if key == ord('q') or key == 27:  # q or ESC で中断したい場合
                    print("中断しました。")
                    break

            cv2.destroyWindow("WAIT_KEY")

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
                # ★全体計測 end
        t_program_end = time.perf_counter()
        print(f"[TIME] total_program_time (incl waitKey) = {t_program_end - t_program_start:.3f} sec")


if __name__ == "__main__":
    main()