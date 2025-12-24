import glob
import os
import cv2
import numpy as np

# ========= ユーザ設定 =========
RGB_GLOB    = "./data/rgb/*.png"      # 同期したRGB画像（番号対応推奨）
THERM_GLOB  = "./data/thermal/*.png"  # 同期したThermal画像
PATTERN_SIZE = (11, 4)
D_MM = 18.0

# 既知の内部パラメータ（RGBはRealSenseの既知値 or 別途キャリブ結果を入れる）
RGB_INTRINSICS_NPZ   = "./out/realsense_intrinsics.npz"  # K, dist, img_size
THERM_INTRINSICS_NPZ = "./out/pi160_intrinsics.npz"      # K, dist, img_size

OUT_NPZ = "./out/extrinsics_rgb_to_thermal.npz"
# =============================

def create_asym_objp(cols, rows, d):
    objp = np.zeros((rows * cols, 3), np.float32)
    k = 0
    for i in range(rows):
        for j in range(cols):
            objp[k, 0] = (2 * j + (i % 2)) * d
            objp[k, 1] = i * d
            k += 1
    return objp

def build_blob_detector():
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 8
    p.maxThreshold = 255
    p.filterByArea = True
    p.minArea = 30
    p.maxArea = 5000
    p.filterByCircularity = True
    p.minCircularity = 0.05
    p.filterByConvexity = True
    p.minConvexity = 0.7
    p.filterByInertia = True
    p.minInertiaRatio = 0.01
    return cv2.SimpleBlobDetector_create(p)

def thermal_to_gray8(img):
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    if g.dtype == np.uint16 or g.dtype == np.int16:
        g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif g.dtype in (np.float32, np.float64):
        g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g8 = g.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(g8)

def load_intrinsics(npz_path):
    d = np.load(npz_path)
    K = d["K"]
    dist = d["dist"]
    img_size = tuple(int(x) for x in d["img_size"])
    return K, dist, img_size

def main():
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)

    rgb_paths = sorted(glob.glob(RGB_GLOB))
    th_paths  = sorted(glob.glob(THERM_GLOB))
    if not rgb_paths or not th_paths:
        raise FileNotFoundError("RGB or Thermal images not found. Check RGB_GLOB / THERM_GLOB.")
    if len(rgb_paths) != len(th_paths):
        print(f"⚠ count mismatch: rgb={len(rgb_paths)} thermal={len(th_paths)}  -> min pairs used")
    n = min(len(rgb_paths), len(th_paths))

    K_rgb, dist_rgb, size_rgb = load_intrinsics(RGB_INTRINSICS_NPZ)
    K_th,  dist_th,  size_th  = load_intrinsics(THERM_INTRINSICS_NPZ)

    objp = create_asym_objp(PATTERN_SIZE[0], PATTERN_SIZE[1], D_MM)
    blob = build_blob_detector()

    objpoints = []
    imgpoints_rgb = []
    imgpoints_th  = []

    flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

    for i in range(n):
        rgb = cv2.imread(rgb_paths[i], cv2.IMREAD_COLOR)
        th_raw = cv2.imread(th_paths[i], cv2.IMREAD_UNCHANGED)

        if rgb is None or th_raw is None:
            continue

        gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        th = thermal_to_gray8(th_raw)

        ok_rgb, ctr_rgb = cv2.findCirclesGrid(gray_rgb, PATTERN_SIZE, flags=flags, blobDetector=blob)
        ok_th,  ctr_th  = cv2.findCirclesGrid(th,       PATTERN_SIZE, flags=flags, blobDetector=blob)

        if not (ok_rgb and ok_th):
            print(f"✗ pair {i:03d} fail  rgb={ok_rgb} th={ok_th}")
            continue

        # 画像サイズがintrinsicsと違うと破綻するのでチェック
        if (gray_rgb.shape[1], gray_rgb.shape[0]) != size_rgb:
            raise ValueError(f"RGB size mismatch. image={gray_rgb.shape[1], gray_rgb.shape[0]} intr={size_rgb}")
        if (th.shape[1], th.shape[0]) != size_th:
            raise ValueError(f"Thermal size mismatch. image={th.shape[1], th.shape[0]} intr={size_th}")

        objpoints.append(objp.copy())
        imgpoints_rgb.append(ctr_rgb.copy())
        imgpoints_th.append(ctr_th.copy())
        print(f"✓ pair {i:03d} ok")

    if len(objpoints) < 8:
        raise RuntimeError(f"Not enough valid pairs: {len(objpoints)} (need ~8+; recommend 15-30 pairs)")

    # stereoCalibrate（内部パラメータ固定で外部だけ推定）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    flags_sc = cv2.CALIB_FIX_INTRINSIC

    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_rgb, imgpoints_th,
        K_rgb, dist_rgb, K_th, dist_th,
        size_rgb, criteria=criteria, flags=flags_sc
    )

    print("\n=== Extrinsics (RGB -> Thermal) ===")
    print("Stereo RMS:", rms)
    print("R:\n", R)
    print("T (same unit as objp; here mm):\n", T.ravel())

    np.savez(OUT_NPZ, R=R, T=T, rms=np.array([rms]),
             K_rgb=K_rgb, dist_rgb=dist_rgb, K_th=K_th, dist_th=dist_th,
             pattern=np.array(PATTERN_SIZE), d_mm=np.array([D_MM]))
    print(f"\nSaved: {OUT_NPZ}")

if __name__ == "__main__":
    main()
# Feeding Robots Calibration Script