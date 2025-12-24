#ThermalとRGB-Dカメラの外部パラメータCalibration
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
PATTERN_SIZE = (11, 4)         # (cols, rows) まずはこれを狙う
D_M = 0.018          # 20mm = 0.02m（あなたのボード定義に合わせる）
THERMAL_INTRINSICS_NPZ = "./out/pi160_intrinsics_live.npz"
OUT_NPZ = "./out/pi160_to_realsense_extrinsics.npz"

NEED_PAIRS = 20
MIN_PAIRS  = 10
SAVE_COOLDOWN_SEC = 0.8

# Thermal側の検出設定
USE_RAW_FOR_DETECT = True
RAW_TO_TEMP = True
THERMAL_INVERT = False

# RGB側の検出設定（←あなたが成功したテストを踏襲）
RGB_INVERT = False
RGB_PREP_MODE = "adaptive"   # "adaptive" or "clahe" or "raw"
# =============================


# ---------- 共通：ボード3D点 ----------
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


# ---------- RGB側：あなたの成功したテストコードの関数 ----------
def make_blob_detector_rgb(blob_color=255, min_area=80, max_area=30000, min_dist=12,
                           min_circularity=0.5, min_inertia=0.1):
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 5
    p.maxThreshold = 250
    p.thresholdStep = 10

    p.filterByColor = True
    p.blobColor = int(blob_color)  # 255: bright circles, 0: dark circles

    p.filterByArea = True
    p.minArea = float(min_area)
    p.maxArea = float(max_area)

    p.filterByCircularity = True
    p.minCircularity = float(min_circularity)

    p.filterByInertia = True
    p.minInertiaRatio = float(min_inertia)

    p.filterByConvexity = False
    p.minDistBetweenBlobs = float(min_dist)

    return cv2.SimpleBlobDetector_create(p)


def preprocess_rgb(gray, mode="adaptive", invert=False):
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    if mode == "adaptive":
        th = cv2.adaptiveThreshold(
            g, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )
        out = th
    elif mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(g)
    else:
        out = g

    if invert:
        out = 255 - out
    return out


# ---------- Thermal側：thermal_to_gray8 ----------
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

    # Thermalは局所コントラストが効くことが多い
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g8 = clahe.apply(g8)

    if invert:
        g8 = 255 - g8
    return g8


def make_blob_detector_thermal(blob_color=0, min_area=20, max_area=5000, min_dist=6,
                               min_circularity=0.05, min_inertia=0.01):
    """
    Thermalは「穴が暗い/冷たい」等で dark blob(0) が効くことが多いが、
    もし逆なら blob_color=255 に変える。
    """
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 0
    p.maxThreshold = 255
    p.thresholdStep = 5
    p.minDistBetweenBlobs = float(min_dist)

    p.filterByColor = True
    p.blobColor = int(blob_color)

    p.filterByArea = True
    p.minArea = float(min_area)
    p.maxArea = float(max_area)

    p.filterByConvexity = False
    p.filterByCircularity = True
    p.minCircularity = float(min_circularity)

    p.filterByInertia = True
    p.minInertiaRatio = float(min_inertia)

    return cv2.SimpleBlobDetector_create(p)


# ---------- デバッグ ----------
def debug_blobs(img8: np.ndarray, blob: cv2.SimpleBlobDetector):
    kps = blob.detect(img8)
    vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(vis, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return len(kps), vis


# ---------- findCirclesGrid（RGB/Thermalで別） ----------
def find_grid_rgb(img8, blob, prefer_pattern=(11, 4)):
    """
    RGBはあなたのテストの勝ちパターンに寄せる：
    ASYM優先 → ダメならCLUSTERING → さらにSYMも試す（保険）
    """
    candidates = [prefer_pattern, (prefer_pattern[1], prefer_pattern[0])]
    flag_sets = [
        cv2.CALIB_CB_ASYMMETRIC_GRID,
        cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
        cv2.CALIB_CB_SYMMETRIC_GRID,
        cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
    ]
    for psz in candidates:
        for flags in flag_sets:
            ok, centers = cv2.findCirclesGrid(img8, psz, flags=flags, blobDetector=blob)
            if ok:
                return True, centers, psz, flags
    return False, None, None, None


def find_grid_thermal(img8, blob, prefer_pattern=(11, 4)):
    """
    Thermalは基本ASymmetric想定だが、念のため両方試す。
    """
    candidates = [prefer_pattern, (prefer_pattern[1], prefer_pattern[0])]
    flag_sets = [
        cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
        cv2.CALIB_CB_ASYMMETRIC_GRID,
        cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
        cv2.CALIB_CB_SYMMETRIC_GRID,
    ]
    for psz in candidates:
        for flags in flag_sets:
            ok, centers = cv2.findCirclesGrid(img8, psz, flags=flags, blobDetector=blob)
            if ok:
                return True, centers, psz, flags
    return False, None, None, None


# ---------- キャプチャ ----------
def capture_thermal_once(cam: PI160Controller):
    cam.get_palette_image()
    cam.get_thermal_data()
    time.sleep(0.05)
    data = cam.get_thermal_data()
    pal  = cam.get_palette_image()
    return data, pal


def get_realsense_color_intrinsics(profile) -> tuple[np.ndarray, np.ndarray]:
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float64)

    if intr.model in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
        dist = np.array(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
    else:
        print(f"⚠ RealSense distortion model={intr.model} not OpenCV-compatible. Using dist=0.")
        dist = np.zeros((5, 1), dtype=np.float64)

    return K, dist


# ---------- 外部計算 ----------
def rvec_tvec_to_Rt(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t


def compute_th2rgb_from_pnp(R_th, t_th, R_rgb, t_rgb):
    # X_rgb = R_tr X_th + t_tr
    R_tr = R_rgb @ R_th.T
    t_tr = t_rgb - (R_tr @ t_th)
    return R_tr, t_tr


def rotmat_to_rotvec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def average_transforms(R_list, t_list):
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

    for _ in range(15):
        pipeline.wait_for_frames()

    K_rgb, dist_rgb = get_realsense_color_intrinsics(profile)

    # ---- detectors (分離) ----
    # RGB: 成功したテストの初期値
    blob_rgb = make_blob_detector_rgb(
        blob_color=255, min_area=80, max_area=30000, min_dist=12,
        min_circularity=0.5, min_inertia=0.1
    )
    # Thermal: 今までの流れ（必要なら blob_color を 255 に）
    blob_th = make_blob_detector_thermal(
        blob_color=0, min_area=20, max_area=5000, min_dist=6,
        min_circularity=0.05, min_inertia=0.01
    )

    # states
    invert_th = THERMAL_INVERT
    invert_rgb = RGB_INVERT
    prep_mode_rgb = RGB_PREP_MODE
    last_saved_t = 0.0

    R_tr_list, t_tr_list = [], []

    print("=== Extrinsics PI160(Thermal) -> RealSense(RGB) [v2] ===")
    print("keys:")
    print("  i : invert THERMAL")
    print("  o : invert RGB")
    print("  p : RGB preprocess mode (adaptive -> clahe -> raw)")
    print(f"  c : compute/save (need >= {MIN_PAIRS} pairs)")
    print("  q / ESC : quit\n")

    try:
        while True:
            # key
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('i'):
                invert_th = not invert_th
                print(f"THERMAL invert = {invert_th}")
                time.sleep(0.1)
                continue
            if key == ord('o'):
                invert_rgb = not invert_rgb
                print(f"RGB invert = {invert_rgb}")
                time.sleep(0.1)
                continue
            if key == ord('p'):
                prep_mode_rgb = "clahe" if prep_mode_rgb == "adaptive" else ("raw" if prep_mode_rgb == "clahe" else "adaptive")
                print(f"RGB preprocess = {prep_mode_rgb}")
                time.sleep(0.1)
                continue
            if key == ord('c'):
                if len(R_tr_list) < MIN_PAIRS:
                    print(f"✗ need >= {MIN_PAIRS}. now={len(R_tr_list)}")
                    continue
                print("computing final transform...")
                break

            # --- capture pair ---
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            data, pal = capture_thermal_once(cam_th)
            src_th = data if (USE_RAW_FOR_DETECT and data is not None) else pal
            if RAW_TO_TEMP and (src_th is not None) and (src_th.dtype == np.uint16) and (src_th.ndim == 2):
                src_th = src_th.astype(np.float32) * 0.1 - 100.0

            # --- preprocess (分離) ---
            img8_rgb = preprocess_rgb(gray, mode=prep_mode_rgb, invert=invert_rgb)
            img8_th  = thermal_to_gray8(src_th, invert=invert_th)
            if img8_th is None:
                continue

            # --- debug blobs ---
            kp_rgb, vis_kp_rgb = debug_blobs(img8_rgb, blob_rgb)
            kp_th,  vis_kp_th  = debug_blobs(img8_th,  blob_th)

            # --- detect grid (分離) ---
            ok_rgb, centers_rgb, psz_rgb, flags_rgb = find_grid_rgb(img8_rgb, blob_rgb, prefer_pattern=PATTERN_SIZE)
            ok_th,  centers_th,  psz_th,  flags_th  = find_grid_thermal(img8_th, blob_th, prefer_pattern=PATTERN_SIZE)

            # --- visualize ---
            vis_rgb = bgr.copy()
            vis_th = cv2.cvtColor(img8_th, cv2.COLOR_GRAY2BGR)

            cv2.putText(vis_rgb, f"RGB kp={kp_rgb} mode={prep_mode_rgb} inv={invert_rgb}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(vis_th, f"TH kp={kp_th} inv={invert_th}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if ok_rgb:
                cv2.drawChessboardCorners(vis_rgb, psz_rgb, centers_rgb, True)
                cv2.putText(vis_rgb, f"FOUND {psz_rgb} flags={flags_rgb}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if ok_th:
                cv2.drawChessboardCorners(vis_th, psz_th, centers_th, True)
                cv2.putText(vis_th, f"FOUND {psz_th} flags={flags_th}", (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # 見やすく拡大（Thermalは小さいので）
            vis_th_big = cv2.resize(vis_th, (160 * 4, 120 * 4), interpolation=cv2.INTER_NEAREST)
            vis_kp_th_big = cv2.resize(vis_kp_th, (160 * 4, 120 * 4), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("RGB", vis_rgb)
            cv2.imshow("RGB_preprocess", img8_rgb)
            cv2.imshow("RGB_blob_debug", vis_kp_rgb)

            cv2.imshow("THERMAL", vis_th_big)
            cv2.imshow("THERMAL_blob_debug", vis_kp_th_big)

            # --- accept pair only when BOTH ok and same pattern interpretation ---
            if not (ok_rgb and ok_th):
                continue

            if psz_rgb != psz_th:
                # 同じボードでも(11,4)と(4,11)の解釈が食い違うとPnPが壊れるので捨てる
                print(f"skip: pattern size mismatch rgb={psz_rgb}, th={psz_th}")
                continue

            now = time.time()
            if (now - last_saved_t) < SAVE_COOLDOWN_SEC:
                continue

            # --- solvePnP (各カメラの姿勢) ---
            objp = create_asym_objp(psz_rgb[0], psz_rgb[1], D_M).astype(np.float64)

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

            print(f"✓ saved pair: {len(R_tr_list)}/{NEED_PAIRS}")

            if len(R_tr_list) >= NEED_PAIRS:
                print("✓ enough pairs collected.")
                break

        cv2.destroyAllWindows()

        if len(R_tr_list) < MIN_PAIRS:
            print(f"Not enough pairs: {len(R_tr_list)} (need >= {MIN_PAIRS})")
            return

        # --- outlier rejection on translation ---
        t_arr = np.array([t.reshape(3) for t in t_tr_list])
        t_med = np.median(t_arr, axis=0)
        d = np.linalg.norm(t_arr - t_med, axis=1)
        thr = np.median(d) + 2.0 * np.std(d) + 1e-9
        keep = d < thr

        R_kept = [R for R, k in zip(R_tr_list, keep) if k]
        t_kept = [t for t, k in zip(t_tr_list, keep) if k]
        print(f"kept {len(R_kept)}/{len(R_tr_list)} after translation outlier removal (thr={thr:.4f}m)")

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
            invert_th=np.array([int(invert_th)], dtype=np.int32),
            invert_rgb=np.array([int(invert_rgb)], dtype=np.int32),
            rgb_prep_mode=np.array([prep_mode_rgb]),
        )
        print(f"\nSaved: {OUT_NPZ}")

    finally:
        pipeline.stop()
        cam_th.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
