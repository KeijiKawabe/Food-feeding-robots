import cv2
import numpy as np
import pyrealsense2 as rs

# ============================================
# 1. チェッカーボード設定
# ============================================
CHECKERBOARD = (11, 7)
SQUARE_SIZE = 0.03   # 25mm = 0.025m

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  
imgpoints = []  

# ============================================
# 2. RealSense の初期化（640×480）
# ============================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("=== RealSense Calibration Mode (640x480) ===")
print("Enter = 1枚保存 / q = キャリブレーション開始\n")

try:
    while True:
        # フレーム取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # チェッカーボード検出
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = img.copy()

        if ret:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret)

        cv2.imshow("RealSense 640x480 - Calibration", display)
        key = cv2.waitKey(1)

        # --- q キーで終了＆キャリブレーション ---
        if key == ord('q'):
            print("\nStarting calibration...")
            break

        # --- Enter キーで保存 ---
        if key == 13:
            if ret:
                print("[OK] 保存しました")
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                imgpoints.append(corners2)
            else:
                print("[NG] チェッカーボードが見つかりません")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# ============================================
# 3. キャリブレーション実行
# ============================================
if len(objpoints) < 5:
    print("❌ サンプルが不足しています（最低5枚以上推奨）")
    exit()

print(f"\nUsing {len(objpoints)} images for calibration...")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (640, 480), None, None
)

# ============================================
# 4. 結果表示
# ============================================
print("\n=== Calibration Results (640x480) ===")
print("RMS Reprojection Error:", ret)

print("\ncamera_matrix (K):\n", camera_matrix)
print("\nfx =", camera_matrix[0, 0])
print("fy =", camera_matrix[1, 1])
print("cx =", camera_matrix[0, 2])
print("cy =", camera_matrix[1, 2])

print("\ndist_coeffs:\n", dist_coeffs)

# 保存
np.save("camera_matrix_640x480.npy", camera_matrix)
np.save("dist_coeffs_640x480.npy", dist_coeffs)

print("\nSaved: camera_matrix_640x480.npy, dist_coeffs_640x480.npy")
print("Calibration complete.")