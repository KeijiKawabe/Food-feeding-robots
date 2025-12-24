# test_scripts/pi160_intrinsics_live.py

import os
import sys
import time
import cv2
import numpy as np

# プロジェクトルートをimportパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.thermal.pi160_controller import PI160Controller


# ========= ユーザ設定 =========
PATTERN_SIZE = (11, 4)     # (cols, rows) = 44点  ※OpenCV findCirclesGrid と同じ順
D_MM = 18.0                # d=20mm（あなたのボード定義に合わせる）
OUT_NPZ = "./out/pi160_intrinsics_live.npz"

NEED_FRAMES = 20           # 推奨 15〜30
USE_RAW_FOR_DETECT = True  # True: raw(温度16bit) で検出 / False: paletteで検出
RAW_TO_TEMP = True         # raw -> ℃変換をするか（検出用はどちらでもOK）
# =============================


def create_asym_objp(cols: int, rows: int, d: float) -> np.ndarray:
    """
    Asymmetric circles grid の3D点（Z=0平面）。
    OpenCVの findCirclesGrid(CALIB_CB_ASYMMETRIC_GRID) と整合しやすい並び。
    単位は d の単位（ここでは mm）。
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

    # しきい値走査（サーマルはこれが効く）
    p.minThreshold = 0
    p.maxThreshold = 255
    p.thresholdStep = 5
    p.minDistBetweenBlobs = 6 # まず 6〜12 を試す（円の直径に合わせて調整）

    # 明るい点を拾う（←ここ重要）
    p.filterByColor = True
    p.blobColor = 0

    # 面積だけでまず拾う（形状制約は全部OFF推奨）
    p.filterByArea = True
    p.minArea = 20
    p.maxArea = 50
    p.filterByConvexity = False
    p.filterByCircularity = True
    p.minCircularity = 0.05

    p.filterByInertia = True
    p.minInertiaRatio = 0.01

    return cv2.SimpleBlobDetector_create(p)


def thermal_to_gray8(img: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    16bit/float/8bit/BGR を受けて、検出用の 8bitグレースケールに整形。
    """
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

    # 局所コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g8 = clahe.apply(g8)

    if invert:
        g8 = 255 - g8
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


def capture_once(cam: PI160Controller):
    """ThermalGPTSystem.capture() と同じ流れで 1枚取る"""
    if (not getattr(cam, "lib", None)) or (not getattr(cam, "handle", None)):
        return None, None

    # バッファクリア
    cam.get_palette_image()
    cam.get_thermal_data()
    time.sleep(0.1)

    data = cam.get_thermal_data()       # raw (例: uint16 (120,160))
    pal  = cam.get_palette_image()      # palette (例: uint8 (120,160,3))
    return data, pal

def debug_blobs(g8: np.ndarray, blob: cv2.SimpleBlobDetector):
    kps = blob.detect(g8)
    vis = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(
        vis, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return len(kps), vis


def try_find_patterns(g8, blob):
    cands = [
        (cv2.CALIB_CB_ASYMMETRIC_GRID, (11, 4)),
        (cv2.CALIB_CB_ASYMMETRIC_GRID, (4, 11)),
        (cv2.CALIB_CB_SYMMETRIC_GRID,  (11, 4)),
        (cv2.CALIB_CB_SYMMETRIC_GRID,  (4, 11)),
    ]
    for base_flag, psz in cands:
        ok, centers = cv2.findCirclesGrid(
            g8, psz,
            flags=base_flag + cv2.CALIB_CB_CLUSTERING,
            blobDetector=blob
        )
        print(f"try {('ASYM' if base_flag==cv2.CALIB_CB_ASYMMETRIC_GRID else 'SYM')} size={psz} -> {ok}")



def main():
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    cam = PI160Controller()

    if (not getattr(cam, "lib", None)) or (not getattr(cam, "handle", None)):
        print("✗ Thermal camera not initialized.")
        return

    blob = build_blob_detector()
    objp = create_asym_objp(PATTERN_SIZE[0], PATTERN_SIZE[1], D_MM)

    objpoints, imgpoints = [], []
    img_size = None
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
    invert = False

    # 表示用のダミー画像（最初は黒画面）
    last_vis = np.zeros((120, 160, 3), dtype=np.uint8)

    print("=== PI160 Intrinsics (press-to-capture) ===")
    print("操作:")
    print("  s : 撮影→グリッド検出→成功なら点を追加")
    print("  i : 反転（黒白が逆で検出しない時）")
    print("  c : 今ある点でキャリブレーション実行（8枚以上必要）")
    print("  q / ESC : 終了\n")

    try:
        while True:
            cv2.imshow("PI160 calib", last_vis)

            key = cv2.waitKey(0) & 0xFF  # ★ここがポイント：キー入力を待つ（撮影しない）
            if key in (ord('q'), 27):
                break

            if key == ord('i'):
                invert = not invert
                print(f"invert = {invert}")
                continue

            if key == ord('c'):
                if len(objpoints) < 8:
                    print(f"✗ need at least 8 valid frames. now={len(objpoints)}")
                    continue
                print("running calibration...")
                break

            if key != ord('s'):
                # 他のキーは無視
                continue

            # ---- s が押されたときだけ撮影する ----
            # ---- s が押されたときだけ撮影する ----
            data, pal = capture_once(cam)
            if data is None and pal is None:
                print("✗ capture failed (data/palette is None)")
                continue

            src = data if (USE_RAW_FOR_DETECT and data is not None) else pal

            if RAW_TO_TEMP and (src is not None) and (src.dtype == np.uint16) and (src.ndim == 2):
                src = src.astype(np.float32) * 0.1 - 100.0

            g8 = thermal_to_gray8(src, invert=invert)
            if g8 is None:
                print("✗ g8 conversion failed")
                continue

            # ===== ここに入れる（findCirclesGrid の前）=====
            n_kp, kp_vis = debug_blobs(g8, blob)
            print(f"blob keypoints = {n_kp}")
            cv2.imshow("blob_debug", kp_vis)
            cv2.waitKey(1)

            # パターン条件を総当りで試す（結果をprintするだけ）
            try_find_patterns(g8, blob)
            # ===============================================

            # 本命（従来どおりの設定でまず一回試す）
            ok, centers = cv2.findCirclesGrid(
                g8, PATTERN_SIZE,
                flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
                blobDetector=blob
            )


            # 表示更新
            vis = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
            if ok:
                cv2.drawChessboardCorners(vis, PATTERN_SIZE, centers, ok)
                objpoints.append(objp.copy())
                imgpoints.append(centers.copy())
                print(f"✓ captured+accepted: {len(objpoints)}/{NEED_FRAMES}")
                cv2.putText(vis, f"FOUND -> SAVED ({len(objpoints)})", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                print("✗ not found (this capture not saved)")
                cv2.putText(vis, "NOT FOUND (not saved)", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(vis, "keys: s(capture) i(invert) c(calib) q(quit)", (5, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            last_vis = vis

            if len(objpoints) >= NEED_FRAMES:
                print("✓ enough frames collected. (auto stop)")
                break


        cv2.destroyAllWindows()

        if len(objpoints) < 8:
            print(f"Not enough valid frames: {len(objpoints)} (need >= 8)")
            return

        # ---- キャリブレーション ----
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )
        rmse = reprojection_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist)

        print("\n=== PI160 Intrinsics ===")
        print("OpenCV RMS:", rms)
        print("Reproj RMSE(px):", rmse)
        print("K:\n", K)
        print("dist:\n", dist.ravel())

        np.savez(
            OUT_NPZ,
            K=K, dist=dist,
            img_size=np.array(img_size),
            d_mm=np.array([D_MM]),
            pattern=np.array(PATTERN_SIZE),
            n_frames=np.array([len(objpoints)]),
            invert=np.array([int(invert)]),
        )
        print(f"\nSaved: {OUT_NPZ}")

    finally:
        cam.disconnect()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
