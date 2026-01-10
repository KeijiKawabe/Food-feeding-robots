# main_feeding_A2_fixed.py
"""
食事介助ロボット・メインスクリプト（v1 / 条件B: Hot/Cold交互 + バラエティ）

要点:
- Hard thermal safety で safe_candidates を作る（unsafeは除外）
- HOT_THR(=40℃) を境に hot/cold bucket を作る
- 基本は bucket を交互に選ぶ（cold->hot->cold->...）
  - ただし次に狙う bucket が空なら、反対側 bucket で妥協
- 同じ bucket 内では直近 avoid_k 回（例:2）の「同bucketラベル」を避ける（可能なら）
- 最終選択は LLM に投げるが、enum を preferred に絞ってルールを効かせる

注意:
- 既存の src/* の実装はそのまま使う想定です。
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional, Tuple, List

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
from src.planner.task_planner import TaskPlanner  # （使うなら）


# ==============================
# 設定項目（ここを環境に合わせて変更）
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

SAFE_TEMP_MAX = 50

# 温度帯（bucket）境界
HOT_THR = 40.0  # 40℃以上を hot、未満を cold とする

# 同bucket内のバラエティ（直近K回を避ける）
AVOID_K = 2


# ==============================
# ユーティリティ関数
# ==============================

def load_calibration(path: str) -> Dict[str, Any]:
    import os, json
    import numpy as np

    def _make_T(R, t3x1):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t3x1.reshape(3)
        return T

    def _invert_rt(R, t3x1):
        Rinv = R.T
        tinv = -Rinv @ t3x1
        return Rinv, tinv

    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(path))

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

    if "K_color" in data:
        data["K_color"] = np.asarray(data["K_color"], dtype=np.float64)
    if "dist_coeffs" in data:
        data["dist_coeffs"] = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    if "T_realsense_base" in data:
        data["T_realsense_base"] = np.asarray(data["T_realsense_base"], dtype=np.float64)
    if "T_realsense_thermal" in data:
        data["T_realsense_thermal"] = np.asarray(data["T_realsense_thermal"], dtype=np.float64)

    th_cfg = data.get("thermal", None)
    if th_cfg is not None:
        TH_INTR_NPZ = th_cfg.get("intrinsics_npz", None)
        TH_EXT_NPZ  = th_cfg.get("extrinsics_npz", None)
        direction   = th_cfg.get("extrinsics_direction", "th2rgb").lower()

        if TH_INTR_NPZ is not None:
            intr_path = _resolve(TH_INTR_NPZ)
            if not os.path.exists(intr_path):
                raise FileNotFoundError(f"✗ thermal intrinsics npz not found: {intr_path}")

            th = np.load(intr_path, allow_pickle=True)
            K_th = th["K"].astype(np.float64)
            dist_th = th["dist"].astype(np.float64).reshape(-1, 1)

            data["K_thermal"] = K_th
            data["dist_thermal"] = dist_th

            W = int(th["W"]) if "W" in th else 160
            H = int(th["H"]) if "H" in th else 120
            data["thermal_size"] = (W, H)

        if TH_EXT_NPZ is not None and ("T_realsense_thermal" not in data):
            ext_path = _resolve(TH_EXT_NPZ)
            if not os.path.exists(ext_path):
                raise FileNotFoundError(f"✗ thermal extrinsics npz not found: {ext_path}")

            ex = np.load(ext_path, allow_pickle=True)
            R_th2rgb = ex["R_th2rgb"].astype(np.float64)
            t_th2rgb = ex["t_th2rgb"].astype(np.float64).reshape(3, 1)

            if direction == "th2rgb":
                R_rgb2th, t_rgb2th = _invert_rt(R_th2rgb, t_th2rgb)
                data["T_realsense_thermal"] = _make_T(R_rgb2th, t_rgb2th)
            elif direction == "rgb2th":
                data["T_realsense_thermal"] = _make_T(R_th2rgb, t_th2rgb)
            else:
                raise ValueError("extrinsics_direction must be 'th2rgb' or 'rgb2th'")

    return data


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


def CheckIfNewPositionInWorkspace(x, y, z):
    if x > 500 or x < 150:
        return False
    if y < -200 or y > 200:
        return False
    if z < 94 or z > 400:
        return False
    return True


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
            "a plate of curry source",
            "BROWN source",
            "whatever brown in a box"
        ],
        "Cone": [
            "sweet corn kernels (maize kernels)",
            "a pile of yellow corn kernels",
            "close-up yellow corn kernels",
            "yellow"
        ],
    }


def hard_thermal_safety_check(
    rec: Dict[str, Any],
    safe_temp_max: float,
    hot_ratio_max: float = 0.10,
    require_project: bool = False,
    min_npts: int = 15,
    max: float = 65.0,
) -> Tuple[bool, str]:
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

    if float(hot_ratio) > hot_ratio_max:
        return False, f"hot_ratio={hot_ratio:.2f} > {hot_ratio_max:.2f}"

    return True, "safe"


def compute_bbox_center_depth_to_base(center_px, depth_m: float, calib: Dict[str, Any]) -> Optional[np.ndarray]:
    if center_px is None or depth_m is None or depth_m <= 0:
        return None

    K = calib.get("K_color", None)
    T_rb = calib.get("T_realsense_base", None)

    if K is None or T_rb is None:
        print("⚠ calib に K_color / T_realsense_base がありません。")
        return None

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u, v = center_px
    Z = depth_m

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    P_cam = np.array([X, Y, Z, 1.0], dtype=np.float32)

    P_base = T_rb @ P_cam
    return P_base[:3]


def crop_thermal_region_by_color_bbox(
    bbox_color,
    thermal_data: np.ndarray,
    calib: Dict[str, Any],
    depth_u16: np.ndarray,
    sample_step: int = 8,
    min_valid_points: int = 10,
    margin: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str, int]:
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
        return (region if region.size > 0 else None), (x1_t, y1_t, x2_t, y2_t), "fallback_scale", npts

    us = np.array([p[0] for p in pts_uv], dtype=np.float64)
    vs = np.array([p[1] for p in pts_uv], dtype=np.float64)

    x1_t = int(np.floor(us.min())) - margin
    x2_t = int(np.ceil(us.max())) + margin
    y1_t = int(np.floor(vs.min())) - margin
    y2_t = int(np.ceil(vs.max())) + margin

    x1_t = max(0, min(w_th - 1, x1_t))
    x2_t = max(0, min(w_th - 1, x2_t))
    y1_t = max(0, min(h_th - 1, y1_t))
    y2_t = max(0, min(h_th - 1, y2_t))

    if x2_t <= x1_t or y2_t <= y1_t:
        return None, None, "project", npts

    region = thermal_data[y1_t:y2_t, x1_t:x2_t]
    return (region if region.size > 0 else None), (x1_t, y1_t, x2_t, y2_t), "project", npts


def attach_thermal_to_per_label_best(
    per_label_best: Dict[str, Dict[str, Any]],
    thermal_data: np.ndarray,
    calib: Dict[str, Any],
    depth_u16: np.ndarray,
    sample_step: int = 8,
) -> Dict[str, Dict[str, Any]]:
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

        r = region.astype(np.float64)
        tmin = float(np.min(r))
        tmax = float(np.max(r))
        tmean = float(np.mean(r))
        p90 = float(np.percentile(r, 90))
        p95 = float(np.percentile(r, 95))
        tmedian = float(np.median(r))
        safe = float(calib.get("safe_temp_max", SAFE_TEMP_MAX))
        hot_ratio = float(np.mean(r > safe))

        rec["thermal"] = {
            "ok": True,
            "bbox_t": [int(bbox_t[0]), int(bbox_t[1]), int(bbox_t[2]), int(bbox_t[3])],
            "min": tmin, "max": tmax, "mean": tmean,
            "median": tmedian,
            "p90": p90,
            "p95": p95,
            "hot_ratio": hot_ratio,
            "mode": mode,
            "npts": int(npts),
        }

    return per_label_best


# ---------- bucket/履歴 ----------
import math

def temp_bucket(t_c: Optional[float], hot_thr: float = HOT_THR) -> str:
    if t_c is None or (isinstance(t_c, float) and (math.isnan(t_c) or math.isinf(t_c))):
        return "unknown"
    return "cold" if t_c < hot_thr else "hot"


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


def opposite_bucket(b: str) -> str:
    return "hot" if b == "cold" else "cold"


def last_k_labels_in_bucket(history: List[Dict[str, Any]], bucket: str, k: int, hot_thr: float) -> List[str]:
    """
    history末尾から見て、そのbucketに属するラベルだけを最大k個集める
    """
    out: List[str] = []
    for h in reversed(history):
        lbl = h.get("label", None)
        t = h.get("temp_mean_c", None)
        try:
            t = float(t) if t is not None else None
        except Exception:
            t = None
        b = temp_bucket(t, hot_thr=hot_thr)
        if b != bucket:
            continue
        if isinstance(lbl, str):
            out.append(lbl)
        if len(out) >= k:
            break
    return out  # most recent first


def choose_label_llm_alternate_hot_cold(
    client: OpenAI,
    safe_candidates: Dict[str, Dict[str, Any]],
    history: List[Dict[str, Any]],
    hot_thr: float = HOT_THR,
    avoid_k: int = AVOID_K,
) -> Dict[str, Any]:
    """
    条件B:
    - 原則：hot/cold を交互に選ぶ
    - 反対bucketが空なら、もう片方で妥協
    - 同bucket内は直近avoid_kラベルを避ける（できれば）
    - 最後に score 高い方が有利（LLM tie-break）
    """
    if not safe_candidates:
        return {"food_label": None, "reason": "no safe candidates"}

    last_label, last_temp = history_last_label_and_temp(history)
    last_bucket = temp_bucket(last_temp, hot_thr=hot_thr) if last_temp is not None else "unknown"

    items = []
    for lbl, rec in safe_candidates.items():
        th = rec.get("thermal", {}) or {}
        t_rep = th.get("p90", None)  # 代表温度（あなたの運用に合わせて p90）
        try:
            t_rep_f = float(t_rep) if t_rep is not None else None
        except Exception:
            t_rep_f = None

        items.append({
            "label": lbl,
            "score": float(rec.get("score", 0.0)),
            "temp_c": t_rep_f,
            "bucket": temp_bucket(t_rep_f, hot_thr=hot_thr) if t_rep_f is not None else "unknown",
        })

    cold_pool = [x for x in items if x["bucket"] == "cold"]
    hot_pool  = [x for x in items if x["bucket"] == "hot"]

    # 次に狙うbucket（交互）
    if last_bucket in ("cold", "hot"):
        target_bucket = opposite_bucket(last_bucket)
    else:
        # 初回/unknownは候補が多い側
        target_bucket = "cold" if len(cold_pool) >= len(hot_pool) else "hot"

    target_pool = hot_pool if target_bucket == "hot" else cold_pool

    # 反対が空なら妥協
    if not target_pool:
        target_bucket = "hot" if target_bucket == "cold" else "cold"
        target_pool = hot_pool if target_bucket == "hot" else cold_pool

    # それでも空なら unknownばかり → 全候補
    if not target_pool:
        target_bucket = "either"
        target_pool = items

    avoid_labels: List[str] = []
    if target_bucket in ("cold", "hot"):
        avoid_labels = last_k_labels_in_bucket(history, bucket=target_bucket, k=avoid_k, hot_thr=hot_thr)

    preferred = [x["label"] for x in target_pool if x["label"] not in set(avoid_labels)]
    if not preferred:
        preferred = [x["label"] for x in target_pool]

    summary = sorted(items, key=lambda x: x["score"], reverse=True)

    prompt = f"""
You are a task planner for a meal-assistance robot.
All candidates are SAFE (hard thermal rules already applied).

RULES (must follow):
1) Alternate temperature buckets using threshold hot_thr={hot_thr}°C.
   If last bucket is cold -> choose hot next, if last bucket is hot -> choose cold next.
2) If the target bucket has no candidates, you may choose from the other bucket.
3) Within the chosen bucket, try to avoid repeating the last {avoid_k} labels of the SAME bucket if possible.

Candidates:
{json.dumps(summary, ensure_ascii=False)}

Last label: {last_label}
Last temp: {last_temp}
Last bucket: {last_bucket}
Target bucket: {target_bucket}
Avoid labels in target bucket (most recent first): {avoid_labels}

Allowed labels (you MUST pick one):
{preferred}

Output STRICT JSON only:
{{
  "food_label": "<one of allowed labels>",
  "reason": "<short English reason (mention alternation + variety + score if needed)>"
}}
"""

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
            model="gpt-4o",
            input=[{"role": "user", "content": prompt}],
            text={"format": {"type": "json_schema", "name": "choose_label_alternate", "strict": True, "schema": schema}},
        )
        return json.loads(resp.output_text)

    except Exception as e:
        # fallback: preferred内でscore最大
        cand = [x for x in items if x["label"] in preferred]
        best = max(cand, key=lambda x: x["score"]) if cand else max(items, key=lambda x: x["score"])
        return {
            "food_label": best["label"],
            "reason": f"fallback: choose highest score in allowed (error={e.__class__.__name__})"
        }


# ---------- 見た目デバッグ ----------
def make_thermal_debug_view(thermal_data: np.ndarray) -> np.ndarray:
    td = thermal_data.astype(np.float32)
    lo, hi = np.percentile(td, 2), np.percentile(td, 98)
    if hi <= lo:
        hi = lo + 1.0
    img8 = np.clip((td - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)


# ---------- ロボット動作（あなたの現状維持） ----------
def move_robot_to_food(
    arm: XArmAPI,
    center_px,
    depth_m: float,
    calib: Dict[str, Any],
    label: str,
):
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

    if P_base is not None:
        # NOTE: あなたのオフセット運用を維持
        x_mm, y_mm, z_mm = x_b * 1000 - 240, y_b * 1000, 210
        error = 15

        CheckIfNewPositionInWorkspace(x_mm, y_mm, z_mm + 50)
        arm.set_position(
            x_mm, y_mm, z_mm + 50,
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


def move_food_to_mouth(arm: XArmAPI):
    arm.set_position(430, 20, 300, -90, 0, -90)


def move_first_position(arm: XArmAPI):
    arm.set_position(360, -100, 320, -90, 0, -90)


# ==============================
# メイン処理ループ
# ==============================

def main():
    t_program_start = time.perf_counter()
    print("=== Meal-Assistance Robot Main ===")
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
        print("   Base 座標への変換はスキップされます。")
        calib = {}

    arm = init_xarm(XARM_IP)

    rs_pipeline = init_realsense()
    align_to = rs.stream.color
    align = rs.align(align_to)

    thermal_system = ThermalGPTSystem(openai_api_key=api_key, target_temp=SAFE_TEMP_MAX)

    if not os.path.exists(SAM2_CFG):
        print("❌ SAM2 config が見つかりません:", SAM2_CFG)
        return
    if not os.path.exists(SAM2_CKPT):
        print("❌ SAM2 checkpoint が見つかりません:", SAM2_CKPT)
        return

    times = 0

    clip_prompts = build_manual_clip_prompts()
    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        device="cuda",
        maskgen_interval=1,
        min_area=1000,
        max_area_frac=0.05,  # ← 背景がでかいマスクを落とす（必要なら更に下げる）
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=True,
    )

    eat_history: List[Dict[str, Any]] = []  # [{"label": str, "temp_mean_c": float}, ...]

    WIN_TH = "THERMAL RAW + projected bbox"

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

            # --- RGB pipeline ---
            rgb_out = pipe.process_frame(color_image, depth_frame=depth_image)
            instances = rgb_out.get("instances", {})

            if not instances:
                print("⚠ 食材が認識できませんでした。次のループへ。")
                continue

            # --- Thermal capture ---
            thermal = thermal_system.capture()
            if thermal is None:
                print("⚠ Thermal 画像取得に失敗。次のループへ。")
                continue

            thermal_data, thermal_img = thermal

            # --- per_label_best ---
            per_label_best: Dict[str, Dict[str, Any]] = {}
            for lbl, inst in instances.items():
                per_label_best[lbl] = {
                    "score": float(inst.get("score", 0.0)),
                    "bbox": inst.get("bbox"),
                    "center_px": inst.get("center_px"),
                    "depth_m": inst.get("depth_m"),
                    "mask": inst.get("mask"),
                }

            calib["safe_temp_max"] = SAFE_TEMP_MAX

            per_label_best = attach_thermal_to_per_label_best(
                per_label_best=per_label_best,
                thermal_data=thermal_data,
                calib=calib,
                depth_u16=depth_image,
                sample_step=8,
            )

            # ===== 断定ログ：各ラベルの温度統計と bucket =====
            print("\n[THERMAL STATS PER LABEL]")
            for lbl, rec in per_label_best.items():
                th = rec.get("thermal", {}) or {}
                t_rep = th.get("p90", None)
                try:
                    t_rep_f = float(t_rep) if t_rep is not None else None
                except Exception:
                    t_rep_f = None
                b = temp_bucket(t_rep_f, hot_thr=HOT_THR)
                print(
                    f"- {lbl:16s} p90={t_rep_f}C  max={th.get('max')}  p95={th.get('p95')}  "
                    f"hot_ratio={th.get('hot_ratio')}  mode={th.get('mode')}  npts={th.get('npts')}  bucket={b}"
                )
            print("")

            # ---- Hard safety filter ----
            safe_candidates: Dict[str, Dict[str, Any]] = {}
            unsafe_reasons: Dict[str, str] = {}

            for lbl, rec in per_label_best.items():
                safe, why = hard_thermal_safety_check(
                    rec,
                    safe_temp_max=SAFE_TEMP_MAX,
                    hot_ratio_max=0.10,
                    require_project=False,  # 精度優先なら True（fallback_scale落とす）
                    min_npts=15,
                    max=65,
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

            # ===== LLM selection: Alternation + Variety =====
            try:
                sel = choose_label_llm_alternate_hot_cold(
                    client=client,
                    safe_candidates=safe_candidates,
                    history=eat_history,
                    hot_thr=HOT_THR,
                    avoid_k=AVOID_K,
                )
                chosen_label = sel["food_label"]
                reason = sel["reason"]
            except Exception as e:
                print("⚠ choose_label_llm_alternate_hot_cold failed:", e)
                chosen_label = max(safe_candidates.items(), key=lambda kv: kv[1].get("score", 0.0))[0]
                reason = "fallback: choose highest RGB score among safe candidates"

            print("\n--- HARD+LLM Decision (Alternate Hot/Cold) ---")
            print("allow      :", True)
            print("food_label :", chosen_label)
            print("reason     :", reason)

            chosen = safe_candidates[chosen_label]
            bbox = chosen["bbox"]
            center_px = chosen["center_px"]
            depth_m = chosen["depth_m"]
            th = chosen.get("thermal", {}) or {}
            bbox_t = th.get("bbox_t", None)

            # --- Thermal debug window（落ちないように保護） ---
            th_vis = make_thermal_debug_view(thermal_data)
            if bbox_t is not None:
                x1, y1, x2, y2 = bbox_t
                cv2.rectangle(th_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(th_vis, f"{chosen_label}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow(WIN_TH, th_vis)
            cv2.waitKey(0)
            try:
                cv2.destroyWindow(WIN_TH)
            except cv2.error:
                pass

            # --- Robot action ---
            move_robot_to_food(
                arm=arm,
                center_px=center_px,
                depth_m=depth_m,
                calib=calib,
                label=chosen_label,
            )

            # --- history update（代表温度を記録） ---
            t_rep = th.get("p90", None)
            try:
                t_rep = float(t_rep) if t_rep is not None else None
            except Exception:
                t_rep = None

            eat_history.append({
                "label": chosen_label,
                "temp_mean_c": t_rep,
            })

            # --- RGB debug ---
            vis = color_image.copy()
            for lbl2, inst in instances.items():
                if inst.get("mask") is not None:
                    vis = draw_mask_on_image(vis, inst["mask"])

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Feeding Perception (RGB + SAM2 + CLIP)", vis)
            print("  → ウィンドウに RGB 認識結果を表示しました。何かキーを押すと閉じます。")
            times += 1
            print(f"{times}回目の作業です。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # --- mouth / reset ---
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

            try:
                cv2.destroyWindow("WAIT_KEY")
            except cv2.error:
                pass

    except KeyboardInterrupt:
        print("\n⏹ キーボード割り込みにより終了します。")

    finally:
        try:
            rs_pipeline.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            thermal_system.cleanup()
        except Exception:
            pass

        try:
            arm.disconnect()
            print("✓ xArm 切断")
        except Exception:
            pass

        print("✓ 全てクリーンアップしました。")
        t_program_end = time.perf_counter()
        print(f"[TIME] total_program_time (incl waitKey) = {t_program_end - t_program_start:.3f} sec")
        print(eat_history)


if __name__ == "__main__":
    main()
