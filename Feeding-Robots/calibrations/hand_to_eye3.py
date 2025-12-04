import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time


# ============================================
# ArUco
# ============================================
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH_M = 0.2  # 3.5cm


# ============================================
# RealSense 初期化
# ============================================
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


# ============================================
# xArm 初期化
# ============================================
def init_xarm(ip="192.168.1.199"):
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    time.sleep(1.0)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)
    return arm


# ============================================
# Utility
# ============================================
def pose_to_matrix(p):
    x, y, z, rx, ry, rz = p
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0  # mm→m
    return T


def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


# ============================================
# Hand–Eye Calibration（Eye-to-Hand）
# ============================================
def handeye():

    pipeline = init_realsense()
    arm = init_xarm()

    # ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()

    # RealSense intrinsics
    fx, fy = 389.846, 389.846
    cx, cy = 321.177, 235.201

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

    dist_coeffs = np.zeros(5)

    # 備考：TCP→Marker オフセット（X方向に28cm）
    T_gm = np.eye(4)
    T_gm[0, 3] = 0.28  # ここが超重要!!!

    # Hand–Eyeの入力
    R_g2b = []
    t_g2b = []
    R_m2c = []
    t_m2c = []

    # 角度・並進を含む十分な pose
    pose_list = [
        [340, 30, 210, -90, 0, -90],
        [340, -10, 210, -90, 0, -90],
        [340, 30, 160, -90, 0, -90],
        [310, 30, 230, -90, 0, -90],
        [280, -20, 260, -90, 0, -90],
        [340, 30, 230, -70, 0, -90],
        [340, 30, 230, -50, 0, -90],
        [340, 30, 210, -90, -20, -90],
        [340, 30, 210, -90, -40, -90],
        [340, 30, 210, -90, 0, -80],
        [340, 30, 210, -90, 0, -60],
    ]

    print("=== Collecting Hand–Eye samples ===")

    for i, p in enumerate(pose_list):

        arm.set_position(*p, speed=20, mvacc=2000, wait=True)
        time.sleep(0.5)

        # ----- robot side -----
        T_bg = pose_to_matrix(arm.get_position(is_radian=True)[1])  # base→gripper
        T_bm = T_bg @ T_gm                                          # base→marker
        T_mb = np.linalg.inv(T_bm)                                  # marker→base

        # ----- camera side -----
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        img = np.asanyarray(frame.get_data())

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is None:
            print(f"[{i}] marker missing, skip")
            continue

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
        )

        T_mc = rt_to_matrix(rvec[0], tvec[0])  # marker→camera

        # ----- push -----
        R_g2b.append(T_mb[:3, :3])
        t_g2b.append(T_mb[:3, 3])
        R_m2c.append(T_mc[:3, :3])
        t_m2c.append(T_mc[:3, 3])

        print(f"[{i}] OK")

    print("=== Solving Hand–Eye ===")

    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b, t_g2b,
        R_m2c, t_m2c,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_gc = np.eye(4)
    T_gc[:3, :3] = R_c2g
    T_gc[:3, 3] = t_c2g.reshape(3,)


    # base→camera
    T_bg0 = pose_to_matrix(arm.get_position(is_radian=True)[1])
    T_bc = T_bg0 @ T_gc

    print("\n===== RESULT: base → camera =====")
    print(T_bc)
    print("\nCamera origin (base frame):", T_bc[:3, 3])

    return T_bc


if __name__ == "__main__":
    handeye()
