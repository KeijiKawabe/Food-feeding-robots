# debug_live_recognition.py
# -*- coding: utf-8 -*-
"""
Live camera + MainCodeと同じ物体認識で、CLIPスコア等をターミナル出力するデバッグ用スクリプト。

- 目的: 「MainCodeの認識結果」と同一の候補（bbox/crop/label/score等）を、
        毎フレーム terminal に出すだけ（CSV/保存なし）。
- 方針: 物体認識は再実装しない。MainCodeの pipeline / 関数をそのまま呼ぶ。

使い方例:
  python debug_live_recognition.py --camera realsense --show
  python debug_live_recognition.py --camera opencv --device_id 0 --show
"""

import os
import sys
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# optional: GUI表示
import cv2

# optional: RealSense
try:
    import pyrealsense2 as rs
    HAS_RS = True
except Exception:
    HAS_RS = False


# ==========================================================
# ADAPTER: ここだけ MainCode に合わせて編集
# ==========================================================
def build_main_perception():
    """
    MainCodeと同じ物体認識パイプラインを生成して返す。
    例: PerceptionPipeline(...) / 既存の init 関数など。
    """
    # プロジェクトルートをimportパスに追加（必要に応じて調整）
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path:
        sys.path.append(ROOT)

    # ---- 例1: src/pipeline.py に PerceptionPipeline がある場合 ----
    # from src.pipeline import PerceptionPipeline
    # pipe = PerceptionPipeline(...)
    # return pipe

    # ---- 例2: main の初期化関数がある場合 ----
    # import main_feeding_A2_fixed as mainmod
    # return mainmod.build_perception_pipeline()

    # ---- 例3: Hydra等でconfigから作る場合 ----
    # return make_pipeline_from_config("configs/xxx.yaml")

    # ↓↓↓ デフォルトは「PerceptionPipelineを探す」形式（あなたが中身を合わせる）↓↓↓
    from src.pipeline import PerceptionPipeline  # ←あなたの構成に合わせてパス調整

    # ここは MainCode と同じ引数に合わせる（0〜数行の編集ポイント）
    # 例: PerceptionPipeline(clip_model="ViT-B/32", sam2_ckpt=..., device="cuda")
    pipe = PerceptionPipeline()
    return pipe


def run_main_perception(pipe, color_bgr: np.ndarray, depth_m: Optional[np.ndarray]) -> Any:
    """
    MainCodeと同じ「1ステップ認識」を実行し、MainCodeと同じ形式の出力を返す。

    重要: ここで呼ぶメソッド/関数が Main と同一であること。
    """
    # ---- よくある候補（あなたの実装に合わせてどれか1つに固定） ----
    # out = pipe.process(color_bgr, depth_m)
    # out = pipe.infer(color_bgr, depth_m)
    # out = pipe.run(color_bgr, depth_m)
    # out = pipe(color_bgr, depth_m)

    # ↓↓↓ 仮: よくある "process" を優先して呼ぶ（あなたのMainに合わせて1行でOK）↓↓↓
    if hasattr(pipe, "process"):
        return pipe.process(color_bgr=color_bgr, depth_m=depth_m)
    if hasattr(pipe, "infer"):
        return pipe.infer(color_bgr=color_bgr, depth_m=depth_m)
    if callable(pipe):
        return pipe(color_bgr, depth_m)
    raise AttributeError("pipe に process/infer/__call__ が見つかりません。ADAPTERでMainと同じ呼び方に合わせてください。")


def extract_candidates_for_print(out: Any) -> List[Dict[str, Any]]:
    """
    MainCode出力から「ターミナル表示したい候補配列」を取り出す。

    ここも Main の出力形式に合わせて最小限調整してOK。
    期待する候補dictの例:
      {
        "bbox_xyxy": [x1,y1,x2,y2],
        "label": "yogurt",
        "score": 0.82,
        "clip_topk": [("yogurt",0.82), ("rice",0.55), ...],
        "temp_mean_c": 32.1,  # 任意
      }
    """
    # 例: out が既に candidates(list) を返す場合
    if isinstance(out, list):
        return out

    # 例: out["candidates"] がある
    if isinstance(out, dict):
        for key in ["candidates", "items", "detections", "objects"]:
            if key in out and isinstance(out[key], list):
                return out[key]

        # 例: per_label_best 形式（あなたが前に使ってた形）
        # per_label_best: {label: {"score":..., "bbox":..., ...}, ...}
        if "per_label_best" in out and isinstance(out["per_label_best"], dict):
            cand = []
            for lbl, rec in out["per_label_best"].items():
                c = {"label": lbl}
                c["score"] = float(rec.get("score", 0.0))
                c["bbox_xyxy"] = rec.get("bbox_xyxy") or rec.get("bbox") or None
                c["clip_topk"] = rec.get("clip_topk") or rec.get("topk") or None
                c["temp_mean_c"] = rec.get("temp_mean_c") or rec.get("thermal", {}).get("p50") or rec.get("thermal", {}).get("p90")
                cand.append(c)
            # score降順
            cand.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return cand

    # どうしても不明なら空
    return []
# ==========================================================


# --------------------------
# Camera: RealSense
# --------------------------
class RealSenseReader:
    def __init__(self, width=640, height=480, fps=30, enable_depth=True):
        if not HAS_RS:
            raise RuntimeError("pyrealsense2 が import できません。--camera opencv を使うか、realsense環境を確認してください。")
        self.enable_depth = enable_depth
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color) if enable_depth else None
        self.depth_scale = None
        if enable_depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

    def read(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        if self.enable_depth:
            frames = self.align.process(frames)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame")
        color_bgr = np.asanyarray(color.get_data())

        depth_m = None
        if self.enable_depth:
            depth = frames.get_depth_frame()
            if depth:
                depth_raw = np.asanyarray(depth.get_data()).astype(np.float32)
                depth_m = depth_raw * float(self.depth_scale)

        return color_bgr, depth_m

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# --------------------------
# Camera: OpenCV
# --------------------------
class OpenCVReader:
    def __init__(self, device_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("OpenCV camera read failed")
        return frame, None

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


# --------------------------
# Pretty print
# --------------------------
def format_candidate(c: Dict[str, Any], topk_n: int = 5) -> str:
    lbl = c.get("label", None)
    score = c.get("score", None)
    bbox = c.get("bbox_xyxy", None)
    t = c.get("temp_mean_c", None)

    s = []
    s.append(f"label={lbl} score={score:.3f}" if isinstance(score, (int, float)) else f"label={lbl} score={score}")
    if bbox is not None:
        s.append(f"bbox={bbox}")
    if isinstance(t, (int, float)):
        s.append(f"temp={t:.1f}C")

    # clip_topk: list of tuples or list of dicts
    topk = c.get("clip_topk", None)
    if topk:
        # normalize
        norm = []
        for item in topk:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                norm.append((str(item[0]), float(item[1])))
            elif isinstance(item, dict):
                p = item.get("prompt") or item.get("label") or item.get("text")
                sc = item.get("score")
                if p is not None and sc is not None:
                    norm.append((str(p), float(sc)))
        norm = norm[:topk_n]
        if norm:
            s.append("topk=" + ", ".join([f"{p}:{sc:.2f}" for p, sc in norm]))

    return " | ".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=str, default="realsense", choices=["realsense", "opencv"])
    ap.add_argument("--device_id", type=int, default=0, help="opencv用")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no_depth", action="store_true", help="realsenseでもdepth不要なら指定")
    ap.add_argument("--show", action="store_true", help="OpenCVウィンドウ表示（任意）")
    ap.add_argument("--topn", type=int, default=5, help="表示する候補数（score降順）")
    ap.add_argument("--topk", type=int, default=5, help="候補内で表示するCLIP top-k")
    ap.add_argument("--hz", type=float, default=5.0, help="ターミナル出力頻度(Hz)。高すぎると見づらい")
    args = ap.parse_args()

    # build pipeline (MainCodeと同じ)
    pipe = build_main_perception()

    # camera
    if args.camera == "realsense":
        reader = RealSenseReader(args.width, args.height, args.fps, enable_depth=(not args.no_depth))
    else:
        reader = OpenCVReader(args.device_id, args.width, args.height)

    print("=== Live recognition debug (terminal only) ===")
    print(f"camera={args.camera} show={args.show} output_hz={args.hz}")

    last_print = 0.0
    frame_i = 0
    try:
        while True:
            color_bgr, depth_m = reader.read()
            frame_i += 1

            t0 = time.time()
            out = run_main_perception(pipe, color_bgr, depth_m)
            dt = (time.time() - t0) * 1000.0

            cands = extract_candidates_for_print(out)
            # それっぽくscore降順
            cands_sorted = sorted(cands, key=lambda x: float(x.get("score", 0.0)) if x.get("score") is not None else 0.0, reverse=True)

            now = time.time()
            if now - last_print >= (1.0 / max(0.1, args.hz)):
                last_print = now
                print("\n" + "-" * 80)
                print(f"frame={frame_i}  infer_ms={dt:.1f}  num_candidates={len(cands_sorted)}")
                for i, c in enumerate(cands_sorted[:args.topn]):
                    print(f"[{i}] {format_candidate(c, topk_n=args.topk)}")

            if args.show:
                vis = color_bgr.copy()
                # bboxがあれば描画（任意）
                for i, c in enumerate(cands_sorted[:args.topn]):
                    bb = c.get("bbox_xyxy", None)
                    if bb and len(bb) == 4:
                        x1, y1, x2, y2 = map(int, bb)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        lbl = c.get("label", "None")
                        sc = c.get("score", 0.0)
                        cv2.putText(vis, f"{i}:{lbl}:{sc:.2f}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("live_debug", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
            else:
                # GUIなしでも q で止めたい人向け（Windowsだと効かないこともある）
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        reader.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
