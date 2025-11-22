# test_scripts/test_pipeline_realsense.py

import os
import sys
import time

import cv2
import numpy as np

# RealSense
import pyrealsense2 as rs

# プロジェクトの src を import できるようにパス追加
ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(PROJECT_ROOT)

from src.pipeline import PerceptionPipeline
from src.utils.misc import draw_mask_on_image


def main():
    # --- SAM2 の設定ファイル/重みのパス ---
    CFG = os.path.join(
        ROOT,
        "..",
        "..",
        "sam2",
        "sam2",
        "configs",
        "sam2.1",
        "sam2.1_hiera_b+.yaml",
    )
    CKPT = os.path.join(
        ROOT,
        "..",
        "..",
        "sam2",
        "checkpoints",
        "sam2.1_hiera_base_plus.pt",
    )

    if not os.path.exists(CFG):
        print("❌ SAM2 config が見つかりません:", CFG)
        return
    if not os.path.exists(CKPT):
        print("❌ SAM2 checkpoint が見つかりません:", CKPT)
        return

    # --- PerceptionPipeline 初期化 ---
    pipe = PerceptionPipeline(
        sam2_cfg=CFG,
        sam2_ckpt=CKPT,
        device="cuda",          # GPU が無ければ "cpu" でもOK（遅くなる）
        maskgen_interval=3,     # 3フレームごとにマスク生成（負荷軽減）
        min_area=1000,
        max_area_frac=0.5,
        clip_model="ViT-B/32",
        clip_prompts=None,      # ClipScorer 側のデフォルト or 後で update_clip_prompts() で上書き
        enable_depth=True,
    )

    # -------------------------------
    # RealSense セットアップ
    # -------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # 一般的な設定（適宜変更可）
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # depth を color にアライン
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("▶ RealSense パイプライン開始中...")
    profile = pipeline.start(config)
    print("✓ RealSense スタート")

    try:
        while True:
            # --- フレーム取得 ---
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("⚠ フレーム取得失敗。スキップ。")
                continue

            # numpy 配列に変換
            depth_image = np.asanyarray(depth_frame.get_data())   # (H, W), uint16, 単位 mm
            color_image = np.asanyarray(color_frame.get_data())   # (H, W, 3), BGR

            # --- パイプライン実行 ---
            out = pipe.process_frame(color_image, depth_image)

            mask = out["mask"]
            bbox = out["bbox"]
            label = out["label"]
            score = out["score"]
            center_px = out["center_px"]
            depth_m = out["depth_m"]
            fps = out["fps"]

            # --- 可視化 ---
            vis = color_image.copy()

            # マスクがある場合はオーバーレイ
            if mask is not None:
                vis = draw_mask_on_image(vis, mask)

            # BBOX がある場合は矩形を描画
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ラベルや深さ情報を表示
            text_lines = []
            if label is not None:
                text_lines.append(f"Label: {label}")
            if score is not None:
                text_lines.append(f"CLIP score: {score:.2f}")
            if depth_m is not None:
                text_lines.append(f"Depth: {depth_m:.3f} m")
            if fps is not None:
                text_lines.append(f"FPS: {fps:.1f}")

            y0 = 20
            for i, txt in enumerate(text_lines):
                y = y0 + i * 18
                cv2.putText(
                    vis,
                    txt,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 中心ピクセルもマーキング
            if center_px is not None:
                cx, cy = center_px
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

            cv2.imshow("PerceptionPipeline (RealSense)", vis)

            # コンソールにも簡単に出力
            print(
                f"[PIPELINE] label={label}, score={score}, depth_m={depth_m}, fps={fps:.2f}"
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("▶ 'q' キーが押されたので終了します。")
                break

    except KeyboardInterrupt:
        print("\n⏹ キーボード割り込みにより終了します。")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("✓ RealSense 停止・ウィンドウを閉じました。")


if __name__ == "__main__":
    main()
