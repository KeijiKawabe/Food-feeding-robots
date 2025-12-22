import glob
import os
import cv2
import numpy as np

# ========= ユーザ設定 =========
THERMAL_GLOB = "./data/thermal/*.png"   # 16bit/8bitどちらでもOK
PATTERN_SIZE = (11, 4)                 # (cols, rows) = 44点
D_MM = 20.0                            # d=20mm (ピッチ40mmなら半分)
OUT_NPZ = "./out/pi160_intrinsics.npz"
# =============================

def create_asym_objp(cols: int, rows: int, d: float) -> np.ndarray:
    """
    Asymmetric circles grid の3D点（Z=0平面）。
    OpenCVの findCirclesGrid(CALIB_CB_ASYMMETRIC_GRID) と整合しやすい並び。
    単位はdの単位（ここではmm）。
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    k = 0
    for i in range(rows):        # row
        for j in range(cols):    # col
            objp[k, 0] = (2 * j + (i % 2)) * d
            objp[k, 1] = i * d
            objp[k, 2] = 0.0
            k += 1
    return objp

def build_blob_detector() -> cv2.SimpleBlobDetector:
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 8
    p.maxThreshold = 255

    p.filterByArea = True
    p.minArea = 30       # PI160は低解像度なので小さめから試す
    p.maxArea = 5000

    p.filterByCircularity = True
    p.minCircularity = 0.05

    p.filterByConvexity = True
    p.minConvexity = 0.7

    p.filterByInertia = True
    p.minInertiaRatio = 0.01

    return cv2.SimpleBlobDetector_create(p)

def thermal_to_gray8(img: np.ndarray) -> np.ndarray:
    """
    16bit/float/8bit を受けて、検出用の 8bitグレースケールに整形。
    """
    if img.ndim == 3:
        # palette画像など(BGR)ならグレー化
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()

    if g.dtype == np.uint16:
        # 16bitを正規化して8bit化（ロバストにするならパーセンタイル推奨）
        g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif g.dtype in (np.float32, np.float64):
        g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g8 = g.astype(np.uint8)

    # コントラスト強調（必要なら）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g8 = clahe.apply(g8)
    return g8

def reprojection_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist) -> float:
    total_err2 = 0.0
    total_n = 0
    for objp, imgp, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rv, tv, K, dist)
        err = imgp.reshape(-1, 2) - proj.reshape(-1, 2)
        total_err2 += float(np.sum(err * err))
        total_n += objp.shape[0]
    return float(np.sqrt(total_err2 / max(total_n, 1)))

def main():
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)

    images = sorted(glob.glob(THERMAL_GLOB))
    if not images:
        raise FileNotFoundError(f"No images found: {THERMAL_GLOB}")

    blob = build_blob_detector()
    objp = create_asym_objp(PATTERN_SIZE[0], PATTERN_SIZE[1], D_MM)

    objpoints = []
    imgpoints = []
    img_size = None

    flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

    for path in images:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"skip (read fail): {path}")
            continue

        g8 = thermal_to_gray8(raw)
        h, w = g8.shape[:2]
        img_size = (w, h)

        ok, centers = cv2.findCirclesGrid(
            g8, PATTERN_SIZE, flags=flags, blobDetector=blob
        )

        if not ok:
            print(f"✗ not found: {path}")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(centers.copy())
        print(f"✓ found: {path}  points={len(centers)}")

    if len(objpoints) < 8:
        raise RuntimeError(f"Not enough valid frames. valid={len(objpoints)} (need ~8+; recommend 15-30)")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    rmse = reprojection_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist)

    print("\n=== PI160 Intrinsics ===")
    print("OpenCV RMS:", rms)
    print("Reproj RMSE(px):", rmse)
    print("K:\n", K)
    print("dist:\n", dist.ravel())

    np.savez(OUT_NPZ, K=K, dist=dist, img_size=np.array(img_size), d_mm=np.array([D_MM]), pattern=np.array(PATTERN_SIZE))
    print(f"\nSaved: {OUT_NPZ}")

if __name__ == "__main__":
    main()
