import numpy as np
import cv2
import pyrealsense2 as rs
import time
from collections import deque

# ================================
# Hand-eye の結果（Base→Camera）
# ================================
T_base_camera = np.array([
    [-0.79506212, -0.22989224,  0.56127158,  0.30186471],
    [-0.50617810,  0.76131737, -0.40519081,  0.03398601],
    [-0.33415558, -0.60625524, -0.72166102,  0.26712413],
    [0., 0., 0., 1.],
])

MARKER_SIZE = 0.028  # ArUco 一辺 [m]

# ================================
# Utility
# ================================
def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


# ================================
# RealSense setup
# ================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

camera_matrix = np.array([
    [389.846, 0., 321.177],
    [0., 389.846, 235.201],
    [0., 0., 1.]
])
dist_coeffs = np.zeros(5)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters()

# ================================
# データ保存用（過去 N 点）
# ================================
N = 200   # 200 サンプル分記録して誤差解析
positions = deque(maxlen=N)

# ================================
# Main loop
# ================================
print("=== Start HandEye Error Monitoring (Method B) ===")

while True:
    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()
    if not frame:
        continue

    img = np.asanyarray(frame.get_data())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is not None:

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs
        )
        rvec = rvecs[0]
        tvec = tvecs[0]

        # Camera→Marker
        T_cam_marker = rt_to_matrix(rvec, tvec)

        # Base→Marker（推定）
        T_base_marker = T_base_camera @ T_cam_marker
        p = T_base_marker[:3, 3]   # XYZ [m]

        positions.append(p)

        # ===== 誤差計算 =====
        if len(positions) > 5:
            arr = np.array(positions)

            mean_pos = np.mean(arr, axis=0)
            diffs = arr - mean_pos
            errors = np.linalg.norm(diffs, axis=1)

            rms = np.sqrt(np.mean(errors**2))
            max_err = np.max(errors)

            print("\n=== Marker Stability (Method B) ===")
            print("Current Position (Base):", p)
            print("Mean Position [m]:", mean_pos)
            print("RMS Error [m]:", rms)
            print("Max Error [m]:", max_err)

    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
