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
MARKER_LENGTH_M = 0.028  # 28mm

# カメラ内部パラメータ
fx, fy = 389.846, 389.846
cx, cy = 321.177, 235.201
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.zeros(5)


# ===============================
# 2. xArm / RealSense 初期化
# ===============================
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def init_xarm(ip="192.168.1.199"):
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)
    return arm


# ===============================
# 3. xArm の姿勢 → 回転行列・平行移動
# ===============================
def pose_to_matrix(p):
    """xArm の get_position() （mm, rad）を 4×4 変換行列に変換"""

    x, y, z, rx, ry, rz = p

    # xArm の RPY → 回転行列
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
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0  # mm → m

    return T


# ===============================
# 4. solvePnP → 4×4 行列
# ===============================
def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


# ===============================
# 5. Hand-Eye Calibration（Eye-to-Hand）
# ===============================
def handeye():

    pipeline = init_realsense()
    arm = init_xarm()
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()

    # OpenCV が要求する入力形式
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # ↓必要な姿勢だけセット（あなたの自由）
    pose_list = [
        [340, 30, 230, -70,  0, -90],
        [340, 30, 230, -50,  0, -90],
        [340, 30, 210, -90, -20, -90],
        [340, 30, 210, -90, -40, -90],
        [310, 30, 210, -70,   0, -80],
        [310, 30, 210, -50,   0, -60],
    ]

    print("=== Collecting Samples ===")

    for i, pose in enumerate(pose_list):

        arm.set_position(*pose, speed=20, mvacc=2000, wait=True)
        time.sleep(0.5)

        # ① ロボット姿勢（Base→Gripper）
        T_bg = pose_to_matrix(arm.get_position(is_radian=True)[1])
        R_bg = T_bg[:3, :3]
        t_bg = T_bg[:3, 3]

        # OpenCV は Gripper→Base が必要 → 逆行列へ変換
        R_gb = R_bg.T
        t_gb = -R_bg.T @ t_bg

        R_gripper2base.append(R_gb)
        t_gripper2base.append(t_gb)

        # ② カメラ画像からマーカー検出（Target→Camera）
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is None:
            print(f"[{i}] Marker not found.")
            continue

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
        )

        R_tc = cv2.Rodrigues(rvec[0])[0]
        t_tc = tvec[0].reshape(3)

        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)

        print(f"[{i}] OK")

    print("=== Solving Hand-Eye ===")

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.reshape(3)

    print("\n===== RESULT: T_base_camera =====")
    for row in T:
        print(row)
    print("=================================\n")

    return T


if __name__ == "__main__":
    handeye()
