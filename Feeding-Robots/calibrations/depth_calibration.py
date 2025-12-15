import cv2
import numpy as np
import pyrealsense2 as rs

# ==============================
# 1. チェッカーボード設定
# ==============================
CHECKERBOARD = (9, 6)  # コーナー数
SQUARE_SIZE = 0.025    # 1マス25mm = 0.025m

# 3Dの基準座標（チェッカーの3D位置）
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D点
imgpoints = []  # 2D点

# ==============================
# 2. RealSense の Depth ストリーム（640×480）
# ==============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

print("=== RealSense Depth Calibration (640x480) ===")
print("Enter = 1枚保存 / q = キャリブレーション開始\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # 深度 (16bit) → 正規化グレースケール（表示用）
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_normalized = cv2.convertScaleAbs(depth_image, alpha=0.03)

        gray = depth_normalized

        # チェッカーボード検出
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if ret:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret)

        cv2.imshow("Depth Calibration 640x480", display)

        key = cv2.waitKey(1)

        # --- q: キャリブ開始 ---
        if key == ord('q'):
            print("\nStarting depth calibration...")
            break

        # --- Enter: 保存 ---
        if key == 13:  # ENTER
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

# ==============================
# 3. Depth カメラキャリブレーション実行
# ==============================
if len(objpoints) < 5:
    print("❌ サンプル数が足りません（最低5枚以上推奨）")
    exit()

print(f"\nUsing {len(objpoints)} samples for calibration...")

ret, depth_camera_matrix, depth_dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (640, 480), None, None
)

# ==============================
# 4. 結果表示
# ==============================
print("\n=== Depth Camera Calibration Results ===")
print("RMS Error:", ret)

print("\ndepth_camera_matrix:\n", depth_camera_matrix)
print("\nfx =", depth_camera_matrix[0, 0])
print("fy =", depth_camera_matrix[1, 1])
print("cx =", depth_camera_matrix[0, 2])
print("cy =", depth_camera_matrix[1, 2])

print("\ndist_coeffs:\n", depth_dist_coeffs)

np.save("depth_camera_matrix.npy", depth_camera_matrix)
np.save("depth_dist_coeffs.npy", depth_dist_coeffs)

print("\nSaved: depth_camera_matrix.npy, depth_dist_coeffs.npy")
