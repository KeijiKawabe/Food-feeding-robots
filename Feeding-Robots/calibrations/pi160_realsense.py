#カメラと温度カメラのCalibrationコード
import os
import sys
import time
import cv2
import numpy as np

import pyrealsense2 as rs

# プロジェクトルートをimportパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.thermal.pi160_controller import PI160Controller


# ========= ユーザ設定 =========
PATTERN_SIZE = (11, 4)         # (cols, rows)
D_M = 0.018                # 20mm = 0.02m
THERMAL_INTRINSICS_NPZ = "./out/pi160_intrinsics_live.npz"
OUT_NPZ = "./out/pi160_to_realsense_extrinsics.npz"

NEED_FRAMES = 20               # 目標
MIN_FRAMES  = 10               # 最低
SAVE_COOLDOWN_SEC = 0.8

USE_RAW_FOR_DETECT = True
RAW_TO_TEMP = True
INVERT_THERMAL = False         # iキーで切り替え
# =============================


def create_asym_objp(cols: int, rows: int, d: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    k = 0
    for i in range(rows):
        for j in range(cols):
            objp[k, 0] = (2 * j + (i % 2)) * d
            objp[k, 1] = i * d
            objp[k, 2] = 0.0
            k += 1
    return objp


def build_blob_detector_dark() -> cv2.SimpleBlobDetector:
    """黒い円（暗いblob）用"""
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 0
    p.maxThreshold = 255
    p.thresholdStep = 5
    p.minDistBetweenBlobs = 6

    p.filterByColor = True
    p.blobColor = 0  # dark

    p.filterByArea = True
    p.minArea = 20
    p.maxArea = 5000

    p.filterByCircularity = True
    p.minCircularity = 0.05

    p.filterByConvexity = False
    p.filterByInertia = True
    p.minInertiaRatio = 0.01
    return cv2.SimpleBlobDetector_create(p)


def thermal_to_gray8(img: np.ndarray, invert: bool = False) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()

    if g.dtype == np.uint16:
        g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif g.dtype in (np.float32, np.float64):
        g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g8 = g.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g8 = clahe.apply(g8)
    if invert:
        g8 = 255 - g8
    return g8


def rgb_to_gray8(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    return g


def find_asym_grid(g8, blob, pattern_size):
    """(11,4) と (4,11) で試す"""
    candidates = [pattern_size, (pattern_size[1], pattern_size[0])]
    for psz in candidates:
        ok, centers = cv2.findCirclesGrid(
            g8, psz,
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
            blobDetector=blob
        )
        if ok:
            return True, centers, psz
    return False, None, None


def capture_thermal_once(cam: PI160Controller):
    cam.get_palette_image()
    cam.get_thermal_data()
    time.sleep(0.05)
    data = cam.get_thermal_data()
    pal  = cam.get_palette_image()
    return data, pal


def get_realsense_color_intrinsics(profile) -> tuple[np.ndarray, np.ndarray]:
    """
    RealSenseのcolor intrinsicsをOpenCV形式(K, dist)へ。
    distは基本 Brown-Conrady (k1,k2,p1,p2,k3) を想定。
    モデルが違う場合は dist=0 にフォールバック。
    """
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float64)

    # RealSenseの係数配列は [k1,k2,p1,p2,k3] 形式のことが多い
    # ただし model が INVERSE_BROWN_CONRADY の場合は OpenCVと一致しないので dist=0推奨
    if intr.model in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
        dist = np.array(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
    else:
        print(f"⚠ RealSense distortion model={intr.model} is not directly OpenCV-compatible. Using dist=0.")
        dist = np.zeros((5, 1), dtype=np.float64)

    return K, dist


def rvec_tvec_to_Rt(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t


def compute_th2rgb_from_pnp(R_th, t_th, R_rgb, t_rgb):
    """
    OpenCV solvePnP は X_cam = R * X_obj + t を返す。
    Thermalカメラ座標 X_th から RGBカメラ座標 X_rgb への変換:
      X_rgb = R_tr * X_th + t_tr
    ここで:
      R_tr = R_rgb * R_th^T
      t_tr = t_rgb - R_tr * t_th
    """
    R_tr = R_rgb @ R_th.T
    t_tr = t_rgb - (R_tr @ t_th)
    return R_tr, t_tr


def rotmat_to_rotvec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def average_transforms(R_list, t_list):
    """
    回転：回転ベクトルを単純平均（小さいばらつき前提）
    並進：平均
    """
    rvecs = np.array([rotmat_to_rotvec(R) for R in R_list], dtype=np.float64)
    tvecs = np.array([t.reshape(3) for t in t_list], dtype=np.float64)

    r_mean = rvecs.mean(axis=0)
    t_mean = tvecs.mean(axis=0)

    R_mean, _ = cv2.Rodrigues(r_mean.reshape(3, 1))
    t_mean = t_mean.reshape(3, 1)
    return R_mean, t_mean


def main():
    # ---- load thermal intrinsics ----
    if not os.path.exists(THERMAL_INTRINSICS_NPZ):
        print(f"✗ thermal intrinsics not found: {THERMAL_INTRINSICS_NPZ}")
        return
    th_npz = np.load(THERMAL_INTRINSICS_NPZ)
    K_th = th_npz["K"].astype(np.float64)
    dist_th = th_npz["dist"].astype(np.float64).reshape(-1, 1)

    # ---- init PI160 ----
    cam_th = PI160Controller()
    if (not getattr(cam_th, "lib", None)) or (not getattr(cam_th, "handle", None)):
        print("✗ Thermal camera not initialized.")
        return

    # ---- init RealSense ----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # warmup
    for _ in range(10):
        pipeline.wait_for_frames()

    K_rgb, dist_rgb = get_realsense_color_intrinsics(profile)

    blob_th = build_blob_detector_dark()
    blob_rgb = build_blob_detector_dark()

    last_saved_t = 0.0
    invert = INVERT_THERMAL

    # 収集
    R_tr_list = []
    t_tr_list = []

    print("=== Extrinsics PI160 (Thermal) -> RealSense (RGB) ===")
    print("操作:")
    print("  i : Thermal反転")
    print(f"  c : 計算実行（{MIN_FRAMES}枚以上）")
    print("  q / ESC : 終了\n")

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('i'):
                invert = not invert
                print(f"invert = {invert}")
                time.sleep(0.1)
                continue
            if key == ord('c'):
                if len(R_tr_list) < MIN_FRAMES:
                    print(f"✗ need >= {MIN_FRAMES}. now={len(R_tr_list)}")
                    continue
                print("computing final transform...")
                break

            # --- capture pair ---
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            rgb = np.asanyarray(color.get_data())  # (480,640,3) BGR

            data, pal = capture_thermal_once(cam_th)
            src_th = data if (USE_RAW_FOR_DETECT and data is not None) else pal
            if RAW_TO_TEMP and (src_th is not None) and (src_th.dtype == np.uint16) and (src_th.ndim == 2):
                src_th = src_th.astype(np.float32) * 0.1 - 100.0

            g8_th = thermal_to_gray8(src_th, invert=invert)
            g8_rgb = rgb_to_gray8(rgb)

            # --- detect grid ---
            ok_th, centers_th, psz_th = find_asym_grid(g8_th, blob_th, PATTERN_SIZE)
            ok_rgb, centers_rgb, psz_rgb = find_asym_grid(g8_rgb, blob_rgb, PATTERN_SIZE)

            vis_th = cv2.cvtColor(g8_th, cv2.COLOR_GRAY2BGR)
            vis_rgb = rgb.copy()

            if ok_th:
                cv2.drawChessboardCorners(vis_th, psz_th, centers_th, ok_th)
            if ok_rgb:
                cv2.drawChessboardCorners(vis_rgb, psz_rgb, centers_rgb, ok_rgb)

            cv2.imshow("thermal", vis_th)
            cv2.imshow("rgb", vis_rgb)

            if not (ok_th and ok_rgb):
                continue

            # pattern size が一致してないならスキップ（検出が回転解釈違いだと混乱するため）
            if psz_th != psz_rgb:
                print(f"skip: pattern size mismatch th={psz_th}, rgb={psz_rgb}")
                continue

            now = time.time()
            if (now - last_saved_t) < SAVE_COOLDOWN_SEC:
                continue

            # --- solvePnP for each camera ---
            objp = create_asym_objp(psz_th[0], psz_th[1], D_M).astype(np.float64)

            ok1, rvec_th, tvec_th = cv2.solvePnP(objp, centers_th, K_th, dist_th, flags=cv2.SOLVEPNP_ITERATIVE)
            ok2, rvec_rgb, tvec_rgb = cv2.solvePnP(objp, centers_rgb, K_rgb, dist_rgb, flags=cv2.SOLVEPNP_ITERATIVE)

            if not (ok1 and ok2):
                continue

            R_th, t_th = rvec_tvec_to_Rt(rvec_th, tvec_th)
            R_rgb, t_rgb = rvec_tvec_to_Rt(rvec_rgb, tvec_rgb)

            R_tr, t_tr = compute_th2rgb_from_pnp(R_th, t_th, R_rgb, t_rgb)

            R_tr_list.append(R_tr)
            t_tr_list.append(t_tr)
            last_saved_t = now

            print(f"✓ saved pair: {len(R_tr_list)}/{NEED_FRAMES}")

            if len(R_tr_list) >= NEED_FRAMES:
                print("✓ enough frames collected.")
                break

        cv2.destroyAllWindows()

        if len(R_tr_list) < MIN_FRAMES:
            print(f"Not enough pairs: {len(R_tr_list)} (need >= {MIN_FRAMES})")
            return

        # --- simple outlier rejection on translation ---
        t_arr = np.array([t.reshape(3) for t in t_tr_list])
        t_med = np.median(t_arr, axis=0)
        dist = np.linalg.norm(t_arr - t_med, axis=1)
        keep = dist < (np.median(dist) + 2.0 * np.std(dist) + 1e-9)

        R_kept = [R for R, k in zip(R_tr_list, keep) if k]
        t_kept = [t for t, k in zip(t_tr_list, keep) if k]
        print(f"kept {len(R_kept)}/{len(R_tr_list)} after translation outlier removal")

        R_mean, t_mean = average_transforms(R_kept, t_kept)

        print("\n=== Extrinsics (Thermal -> RGB) ===")
        print("R_th2rgb:\n", R_mean)
        print("t_th2rgb (m):\n", t_mean.reshape(3))

        os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
        np.savez(
            OUT_NPZ,
            R_th2rgb=R_mean.astype(np.float64),
            t_th2rgb=t_mean.astype(np.float64),
            n_pairs=np.array([len(R_tr_list)], dtype=np.int32),
            n_kept=np.array([len(R_kept)], dtype=np.int32),
            K_th=K_th, dist_th=dist_th,
            K_rgb=K_rgb, dist_rgb=dist_rgb,
            pattern=np.array(PATTERN_SIZE),
            d_m=np.array([D_M], dtype=np.float64),
            invert=np.array([int(invert)], dtype=np.int32),
        )
        print(f"\nSaved: {OUT_NPZ}")

    finally:
        pipeline.stop()
        cam_th.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
