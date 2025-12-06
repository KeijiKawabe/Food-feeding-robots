import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time
import math

# ===============================
# 1. 設定
# ===============================
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH_M = 0.028  # 28mm マーカー

# <<< ここに hand_to_eye.py の結果をコピペする >>>
T_base_camera = np.array([
    [-0.79506212, -0.22989224,  0.56127158,  0.30186471],
    [-0.50617810,  0.76131737, -0.40519081,  0.03398601],
    [-0.33415558, -0.60625524, -0.72166102,  0.26712413],
    [0.0, 0.0, 0.0, 1.0]
])

# TCP からマーカーまでのオフセット
MARKER_OFFSET = 0.028  # 28mm → 0.028m


# ===============================
# 2. RealSense 初期化
# ===============================
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


# ===============================
# 3. xArm の姿勢 → 4×4行列
# ===============================
def pose_to_matrix(p):
    x, y, z, rx, ry, rz = p

    # RPY → 回転行列
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz),  math.cos(rz), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0  # mm → m

    return T


# ===============================
# 4. solvePnP → 4×4行列
# ===============================
def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


# ===============================
# 5. 誤差検証ループ
# ===============================
def verify_handeye():

    print("=== Hand-Eye Verification Mode ===")

    pipeline = init_realsense()
    arm = XArmAPI("192.168.1.199")
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()

    camera_matrix = np.array([
        [389.846, 0, 321.177],
        [0, 389.846, 235.201],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros(5)

    while True:

        # ---------------------------
        # ① Robot: base → marker
        # ---------------------------
        pose = arm.get_position(is_radian=True)[1]
        T_bg = pose_to_matrix(pose)

        R_tcp = T_bg[:3, :3]
        tcp_pos = T_bg[:3, 3]

        # TCP ローカル Z+ に 0.028m オフセット（あなたの仕様に合わせ調整可）
        offset_world = R_tcp @ np.array([0, 0, MARKER_OFFSET])
        marker_pos_robot = tcp_pos + offset_world

        T_bm_robot = np.eye(4)
        T_bm_robot[:3, :3] = R_tcp
        T_bm_robot[:3, 3] = marker_pos_robot

        # ---------------------------
        # ② Camera: base → marker
        # ---------------------------
        frames = pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is None:
            print("No marker detected.")
            cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
        )

        T_mc = rt_to_matrix(rvecs[0], tvecs[0])  # marker→camera
        T_bm_camera = T_base_camera @ T_mc       # base→marker

        # ---------------------------
        # ③ 3D誤差計算
        # ---------------------------
        p_cam = T_bm_camera[:3, 3]
        p_robot = T_bm_robot[:3, 3]

        diff = p_cam - p_robot
        error = np.linalg.norm(diff)

        print("\n===== Verification =====")
        print("marker_from_camera:", p_cam)
        print("marker_from_robot :", p_robot)
        print("difference (m):    ", diff)
        print("error norm (m):    ", error)

        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    verify_handeye()
