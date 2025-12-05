import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time

# ===============================
# 1. 設定：サイズを実測値(0.035)にする
# ===============================
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH_M = 0.027  # ★ここを必ず3.5cm (0.035) にする

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

def pose_to_matrix(p):
    x, y, z, rx, ry, rz = p
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T

def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def handeye():
    pipeline = init_realsense()
    arm = init_xarm()
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()

    # Camera intrinsics
    fx, fy = 389.846, 389.846
    cx, cy = 321.177, 235.201
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)

    MARKER_OFFSET = 0.28 

    R_b2m = []
    t_b2m = []
    R_c2m = []
    t_c2m = []

    # 姿勢リスト
    pose_list = [
        [310, 30, 210, -90, 0, -90],
        [280, -20, 210, -90, 0, -90],
        [310, 30, 190, -90, 0, -90],
        [280, 30, 260, -90, 0, -90],
        [280, -20, 230, -90, 0, -90],
        [340, 30, 230, -70,  0, -90],
        [340, 30, 230, -50,  0, -90],
        [340, 30, 210, -90, -20, -90],
        [340, 30, 210, -90, -40, -90],
        [340, 30, 210, -90,   0, -80],
        [340, 30, 210, -90,   0, -60],
    ]

    print("=== Collecting samples ===")
    for i, pose in enumerate(pose_list):
        arm.set_position(*pose, speed=20, mvacc=2000, wait=True)
        time.sleep(0.5)

        # Robot: Base -> TCP -> Marker
        T_bg = pose_to_matrix(arm.get_position(is_radian=True)[1])
        tcp_R = T_bg[:3,:3]
        offset_world = tcp_R @ np.array([0, 0, MARKER_OFFSET])
        marker_pos = T_bg[:3, 3] + offset_world
        
        T_bm = np.eye(4)
        T_bm[:3, :3] = tcp_R
        T_bm[:3, 3] = marker_pos
        
        # Camera: Marker -> Camera
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        img = np.asanyarray(frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        
        if ids is None:
            print(f"[{i}] marker missing")
            continue

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
        )
        T_mc = rt_to_matrix(rvec[0], tvec[0])

        # リストに追加 (invはしない)
        R_b2m.append(T_bm[:3,:3])
        t_b2m.append(T_bm[:3,3])
        
        # OpenCVのcalibrateHandEye(TSAI)は Target->Camera を受け取る
        R_c2m.append(T_mc[:3,:3])
        t_c2m.append(T_mc[:3,3])
        print(f"[{i}] OK")

    print("=== Solving... ===")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_b2m, t_b2m,
        R_c2m, t_c2m,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_bc = np.eye(4)
    T_bc[:3,:3] = R_cam2base
    T_bc[:3, 3] = t_cam2base.reshape(3)

    print("\n===== COPY THIS MATRIX BELOW =====")
    print("T_base_camera = np.array([")
    for row in T_bc:
        print(f"  [{row[0]}, {row[1]}, {row[2]}, {row[3]}],")
    print("])")
    print("==================================")

if __name__ == "__main__":
    handeye()