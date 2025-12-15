import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs

# =========================
# パス設定：動いているSAM2テストと同じスタイル
# =========================
THIS_DIR = os.path.dirname(__file__)
# src を import パスに追加
sys.path.append(os.path.join(THIS_DIR, "..", "src"))

from perception.sam2_wrapper import SAM2Engine
from perception.clip_plate_detector import (
    init_clip_for_plate,
    find_plate_mask,
)

# SAM2 の config / ckpt パスも「動いているコード」と同じスタイルにする
ROOT = THIS_DIR
CFG  = os.path.join(ROOT, "..", "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml")
CKPT = os.path.join(ROOT, "..", "..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")


# =========================
# RealSense からカラー画像を1枚取得
# =========================
def capture_realsense_color(width=640, height=480, fps=30) -> np.ndarray:
    """
    RealSense からカラー画像を1枚だけ取得して返す。

    Returns:
        image_bgr: (H, W, 3) の BGR 画像 (np.uint8)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)

    # 露光など安定のため数フレーム捨てる
    for _ in range(5):
        frames = pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        pipeline.stop()
        raise RuntimeError("RealSense からカラー画像が取得できませんでした。")

    image_bgr = np.asanyarray(color_frame.get_data())
    pipeline.stop()
    return image_bgr


def main():
    # # =========================
    # # 1) テスト：画像を読み込む
    # # =========================
    # print("RealSense の代わりに input_test_lol.png を読み込みます...")

    # # テスト画像を読み込む（BGR）
    # image_bgr = cv2.imread("input_test_lol.png")

    # if image_bgr is None:
    #     print("❌ Error: input_test_lol.png が読み込めません。パスを確認してください。")
    #     return
    
    # print("captured frame shape:", image_bgr.shape)

    # 保存して確認
    print("RealSense からカラー画像を取得中...")
    image_bgr = capture_realsense_color()  
    cv2.imwrite("realsense_color_frame.png", image_bgr)
    print("saved: realsense_color_frame.png")


    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # =========================
    # 2) SAM2 初期化 & マスク生成
    # =========================
    if not os.path.exists(CFG) or not os.path.exists(CKPT):
        raise SystemExit(f"SAM2 config/ckpt not found.\nCFG: {CFG}\nCKPT: {CKPT}")

    print("SAM2Engine 初期化中...")
    sam2 = SAM2Engine(
        CFG,
        CKPT,
        device="cuda",
        points_per_side=8,
        min_mask_region_area=500,
    )

    print("SAM2 でマスク生成中...")
    masks_raw = sam2.generate_masks(image_rgb)
    masks_bin = [m["segmentation"].astype(np.uint8) for m in masks_raw]
    print(f"生成されたマスク数: {len(masks_bin)}")
    if not masks_bin:
        print("⚠ マスクが1つも生成されませんでした")
        return

    # ==========================================
    # 追加: すべてのマスク画像を保存・出力する
    # ==========================================
    # 保存用のディレクトリを作成
    save_dir = "sam2_all_masks"
    os.makedirs(save_dir, exist_ok=True)
    print(f"全マスク画像を '{save_dir}' ディレクトリに保存します...")

    # 全マスクの位置関係を確認するための「まとめ画像」を用意
    vis_all_masks = image_bgr.copy()

    for i, mask_u8 in enumerate(masks_bin):
        # --- 1. マスクで切り抜いた画像（対象物以外を黒にする）を保存 ---
        # image_bgr と mask (0 or 1) を AND演算
        masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_u8)
        
        # ファイル名に番号をつけて保存 (例: mask_005_cutout.png)
        filename_cutout = os.path.join(save_dir, f"mask_{i:03d}_cutout.png")
        cv2.imwrite(filename_cutout, masked_img)

        # --- 2. (任意) 白黒のバイナリマスク自体も保存したい場合 ---
        # filename_bin = os.path.join(save_dir, f"mask_{i:03d}_binary.png")
        # cv2.imwrite(filename_bin, mask_u8 * 255)

        # --- 3. まとめ画像用に輪郭を描画 ---
        # ランダムな色を生成 (B, G, R)
        color = np.random.randint(0, 255, 3).tolist()
        
        # 輪郭を検出して描画
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_all_masks, contours, -1, color, 2)
        
        # マスクの中心に番号(ID)を書く（どの画像が何番か分かるように）
        M = cv2.moments(mask_u8)
        if M["m00"] != 0:
            cx_m = int(M["m10"] / M["m00"])
            cy_m = int(M["m01"] / M["m00"])
            # 文字が見えやすいように白文字＋黒縁取り
            cv2.putText(vis_all_masks, str(i), (cx_m, cy_m), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(vis_all_masks, str(i), (cx_m, cy_m), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # 全マスクの輪郭とIDが入った一覧画像を保存
    cv2.imwrite(os.path.join(save_dir, "all_masks_index.png"), vis_all_masks)
    print("保存完了")

    # ==========================================
    # (ここまで追加)
    # ==========================================
    # =========================
    # 3) CLIP で「皿っぽいマスク」を選ぶ
    # =========================
    print("CLIP 初期化中...")
    model, preprocess, text_feats = init_clip_for_plate()

    print("CLIP で皿マスクを推定中...")
    best_idx, best_score, best_bbox, all_scores, all_bboxes = find_plate_mask(
        image_rgb, masks_bin, model, preprocess, text_feats
    )

    if best_bbox is None:
        print("⚠ それっぽい皿マスクが見つかりませんでした")
        return

    x0, y0, x1, y1 = best_bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    print("=== Plate Detection (CLIP + SAM2, RealSense frame) ===")
    print(f"best_idx          : {best_idx}")
    print(f"best_score        : {best_score:.4f}")
    print(f"bbox (x0,y0,x1,y1): {x0}, {y0}, {x1}, {y1}")
    print(f"center (cx,cy)    : {cx:.1f}, {cy:.1f}")
        # =========================
    # 4) 可視化画像を保存（CLIP & SAM2 bbox）
    # =========================
    vis = image_bgr.copy()

    # 1) CLIP が決めた bbox（緑） ← best_bbox
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # 2) SAM2 が元々持っている bbox（青） ← masks_raw[best_idx]["bbox"] は xywh
    mx, my, mw, mh = masks_raw[best_idx]["bbox"]
    sx0, sy0 = int(mx), int(my)
    sx1, sy1 = int(mx + mw), int(my + mh)
    cv2.rectangle(vis, (sx0, sy0), (sx1, sy1), (255, 0, 0), 2)

    # 3) 中心点（赤） ← CLIP bbox の中心
    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    # 4) 画像として保存
    out_path = "debug_plate_clip_realsense3.png"
    cv2.imwrite(out_path, vis)
    print(f"saved: {out_path}")


    # =========================
    # 4) 可視化画像を保存
    # =========================
    vis = image_bgr.copy()
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    cv2.imwrite("debug_plate_clip_realsense.png", vis)
    print("saved: debug_plate_clip_realsense.png")


if __name__ == "__main__":
    main()
