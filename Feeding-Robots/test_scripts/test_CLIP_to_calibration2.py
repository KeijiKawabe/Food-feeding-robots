# test_scripts/test_plate_reach.py

import os
import sys
import time
from typing import Dict, Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

# -----------------------------
# プロジェクト内モジュールのパス設定
# -----------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.pipeline import PerceptionPipeline
from src.utils.misc import draw_mask_on_image

# -----------------------------
# 設定
# -----------------------------

# xArm の IP
XARM_IP = "192.168.1.199"

# hand-eye キャリブレーション結果 (.npy, 4x4 の T_base_camera を想定)
CALIB_PATH = os.path.join(PROJECT_ROOT, "calibrations", "T_Base_rgb.npy")

# SAM2 の設定ファイル / 重み
SAM2_CFG = os.path.join(
    PROJECT_ROOT, "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml"
)
SAM2_CKPT = os.path.join(
    PROJECT_ROOT, "..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt"
)

# 皿のラベル名（CLIP用）
PLATE_LABEL = "plate"

# 皿の上何 m に TCP を持っていくか
OFFSET_Z_M = 0.30  # 30cm 上

# === RealSense Color 内部パラメータ（要確認） ===
# ★重要: これらの値が実際のカメラと一致しているか確認してください
FX = 608.54150390625
FY = 607.1893920898438
CX = 309.4483947753906
CY = 264.0105285644531


# =========================================================
# 補助関数
# =========================================================

def check_if_new_position_in_workspace(x_mm: float, y_mm: float, z_mm: float) -> bool:
    """
    x, y, z: Base座標系 [mm]
    ワークスペースの範囲内なら True, それ以外は False を返す。
    """
    if x_mm > 680 or x_mm < 300:
        return False
    if y_mm < -330 or y_mm > 420:
        return False
    if z_mm < 94 or z_mm > 550:
        return False
    return True


def load_calibration(path: str) -> Dict[str, Any]:
    """
    hand-eye の結果 T_base_camera (Base → Camera) が保存されている .npy を読む。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    arr = np.load(path, allow_pickle=True)

    # パターンA: 4x4 行列だけが保存されている場合
    if isinstance(arr, np.ndarray) and arr.shape == (4, 4):
        T = arr.astype(np.float32)
        return {"T_base_camera": T}

    # パターンB: dict を np.save している場合
    if isinstance(arr, np.ndarray) and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            # キーの名前を統一
            if "T_base_realsense" in obj:
                obj["T_base_camera"] = np.asarray(obj["T_base_realsense"], dtype=np.float32)
            elif "T_base_camera" in obj:
                obj["T_base_camera"] = np.asarray(obj["T_base_camera"], dtype=np.float32)
            else:
                raise ValueError("Calibration dict does not contain T_base_camera or T_base_realsense")
            return obj

    raise ValueError(
        f"Calibration file {path} has unsupported format: "
        f"type={type(arr)}, shape={getattr(arr, 'shape', None)}"
    )


def init_xarm(ip: str) -> XArmAPI:
    """
    xArm の初期化
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


def init_realsense() -> (rs.pipeline, rs.align):
    """
    RealSense のパイプラインを開始
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    print("✓ RealSense スタート")

    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline, align


def build_clip_prompts_for_plate():
    """
    皿検出専用の CLIP プロンプト辞書
    """
    return {
        PLATE_LABEL: [
            "a white dinner plate",
            "a dish on a table",
            "a round plate with food",
        ]
    }


def compute_bbox_center_depth_to_base(
    center_px,
    depth_m: float,
    calib: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    画像上の中心画素 + 深度[m] から、Base 座標系の 3D 点 (x,y,z) を計算。
    
    ★重要な修正点:
    - T_base_camera は Base → Camera の変換
    - カメラ座標 → Base座標 には逆行列が必要
    """
    if center_px is None or depth_m is None or depth_m <= 0:
        return None

    # Base → Camera の変換行列を取得
    T_base_to_cam = calib.get("T_base_camera")
    if T_base_to_cam is None:
        print("⚠ calib に T_base_camera がありません。")
        return None

    # ★修正: Camera → Base に逆変換
    try:
        T_cam_to_base = np.linalg.inv(T_base_to_cam)
    except np.linalg.LinAlgError:
        print("❌ T_base_camera の逆行列が計算できません（特異行列）")
        return None

    fx, fy = FX, FY
    cx, cy = CX, CY

    u, v = center_px
    Z = depth_m  # [m]

    # 画素 + 深度 → カメラ座標 [m]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    P_cam = np.array([X, Y, Z, 1.0], dtype=np.float32)
    
    print(f"[DEBUG] center_px = ({u}, {v})")
    print(f"[DEBUG] depth = {Z:.3f} m")
    print(f"[DEBUG] P_cam (Camera座標) = [{X:.3f}, {Y:.3f}, {Z:.3f}]")

    # Camera → Base 座標系へ変換
    P_base = T_cam_to_base @ P_cam
    print(f"[DEBUG] P_base (Base座標) = [{P_base[0]:.3f}, {P_base[1]:.3f}, {P_base[2]:.3f}]")
    
    return P_base[:3]


def move_robot_above_plate(
    arm: XArmAPI,
    P_plate_base: np.ndarray,
    offset_z_m: float = OFFSET_Z_M,
):
    """
    皿の位置 (Base 座標系, [m]) の真上 offset_z_m に TCP を動かす
    """
    x_b, y_b, z_b = P_plate_base  # [m]

    # 目標位置 [m]
    target_x_m = x_b
    target_y_m = y_b
    target_z_m = z_b + offset_z_m

    # mm に変換
    target_x = target_x_m * 1000.0
    target_y = target_y_m * 1000.0
    target_z = target_z_m * 1000.0

    print("\n========== [PLATE REACH TEST] ==========")
    print(f"皿の推定位置 (Base): x={x_b:.3f}, y={y_b:.3f}, z={z_b:.3f} [m]")
    print(f"                     x={x_b*1000:.1f}, y={y_b*1000:.1f}, z={z_b*1000:.1f} [mm]")
    print(f"目標位置 (Base): x={target_x_m:.3f}, y={target_y_m:.3f}, z={target_z_m:.3f} [m]")
    print(f"                x={target_x:.1f}, y={target_y:.1f}, z={target_z:.1f} [mm]")

    # ワークスペースチェック
    if not check_if_new_position_in_workspace(target_x, target_y, target_z):
        print("\n⚠ 目標位置がワークスペース外です。ロボットは動きません。")
        print(f"  許容範囲: X=[300, 680], Y=[-330, 420], Z=[94, 550] mm")
        return

    # 現在の姿勢を取得
    ret, pose = arm.get_position(is_radian=False)
    if ret != 0 or pose is None:
        print("⚠ 現在姿勢の取得に失敗。ロール・ピッチ・ヨーを固定値で使います。")
        roll = -180.0
        pitch = 0.0
        yaw = 0.0
    else:
        roll = pose[3]
        pitch = pose[4]
        yaw = pose[5]
        print(f"現在の姿勢: roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f} [deg]")

    print("========================================\n")

    # 安全確認
    input("⚠ ロボットが上記の位置まで移動します。周囲の安全を確認し、Enterで開始 > ")

    # 移動実行
    code = arm.set_position(
        x=target_x,
        y=target_y,
        z=target_z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed=100,   # mm/s
        mvacc=1000,  # mm/s^2
        wait=True,
    )
    
    if code == 0:
        print("✓ ロボットが目標位置に到達しました")
    else:
        print(f"⚠ set_position がエラーコードを返しました: {code}")


# =========================================================
# メイン
# =========================================================

def main():
    print("=== Plate Reach Test (SAM2 + CLIP + RealSense + xArm) ===")
    print("Project root:", PROJECT_ROOT)

    # キャリブレーション読込
    try:
        calib = load_calibration(CALIB_PATH)
        print("✓ Calibration loaded from:", CALIB_PATH)
        print(f"  T_base_camera shape: {calib['T_base_camera'].shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ キャリブレーションエラー: {e}")
        return

    # SAM2 の設定ファイル確認
    if not os.path.exists(SAM2_CFG):
        print("❌ SAM2 config が見つかりません:", SAM2_CFG)
        return
    if not os.path.exists(SAM2_CKPT):
        print("❌ SAM2 checkpoint が見つかりません:", SAM2_CKPT)
        return

    # xArm 初期化
    arm = init_xarm(XARM_IP)

    # RealSense 初期化
    pipeline, align = init_realsense()

    # PerceptionPipeline 初期化
    clip_prompts = build_clip_prompts_for_plate()
    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        device="cuda",
        maskgen_interval=1,
        min_area=1000,
        max_area_frac=0.7,
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
        enable_depth=True,
    )

    try:
        input("\n皿とロボットの配置ができたら Enter を押してください > ")

        # フレーム取得
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            print("❌ フレーム取得に失敗しました。")
            return

        # ★修正: 深度の単位を確認
        depth_image = np.asanyarray(depth_frame.get_data())  # uint16, mm単位
        color_image = np.asanyarray(color_frame.get_data())  # BGR

        print(f"[DEBUG] depth_image shape: {depth_image.shape}, dtype: {depth_image.dtype}")
        print(f"[DEBUG] color_image shape: {color_image.shape}, dtype: {color_image.dtype}")

        # Perception 実行
        out = pipe.process_frame(
            color_image,
            depth_frame=depth_image,
            target_label=PLATE_LABEL,
        )

        label = out.get("label")
        bbox = out.get("bbox")
        center_px = out.get("center_px")
        depth_m = out.get("depth_m")

        print("\n--- Perception Result ---")
        print("label    :", label)
        print("bbox     :", bbox)
        print("center_px:", center_px)
        print("depth_m  :", depth_m)

        if label != PLATE_LABEL:
            print(f"❌ CLIP の結果が plate ではありませんでした: {label}")
            print("→ 皿が検出されたか画像を確認してください")
        
        if bbox is None or center_px is None or depth_m is None:
            print("❌ bbox / center / depth が取得できませんでした。")
            # 可視化だけ実行
            if out.get("mask") is not None:
                vis = color_image.copy()
                vis = draw_mask_on_image(vis, out["mask"])
                cv2.imshow("Detection Result", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return

        # 可視化
        vis = color_image.copy()
        if out.get("mask") is not None:
            vis = draw_mask_on_image(vis, out["mask"])
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = center_px
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"depth: {depth_m:.3f}m", (cx+10, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Plate Detection", vis)
        print("→ 皿の検出結果をウィンドウに表示しました。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # pixel + depth → Base 座標
        P_base = compute_bbox_center_depth_to_base(center_px, depth_m, calib)
        if P_base is None:
            print("❌ 皿の Base 座標計算に失敗しました。")
            return

        # 皿の真上へロボットを動かす
        move_robot_above_plate(arm, P_base, offset_z_m=OFFSET_Z_M)

        print("\n✅ Plate Reach Test 完了。実際の位置関係を目視で確認してください。")

    except KeyboardInterrupt:
        print("\n⏹ キーボード割り込みで終了します。")

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            arm.disconnect()
            print("✓ xArm 切断")
        except Exception:
            pass


if __name__ == "__main__":
    main()