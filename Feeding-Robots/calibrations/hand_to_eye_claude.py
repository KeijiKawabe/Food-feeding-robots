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
MARKER_LENGTH_M = 0.027  # ★実測値に合わせてください

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
    """xArm pose (mm, radian) → 4x4 matrix (Base → TCP)"""
    x, y, z, rx, ry, rz = p
    
    # RPY (Roll-Pitch-Yaw) から回転行列
    # xArmの順序: Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0  # mm → m
    return T

def rt_to_matrix(rvec, tvec):
    """ArUco rvec, tvec → 4x4 matrix (Marker → Camera)"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
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

    # ★オフセット: 実際のマーカー位置に合わせてください
    # TCPローカル座標系での [X, Y, Z] (m)
    offset_local = np.array([0, 0, 0.28])  # 例: Z方向に28cm
    
    # 回転オフセット（マーカーの向きがTCPと同じなら単位行列）
    R_offset_rot = np.eye(3)

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # 姿勢リスト
    pose_list = [
        [310, 30, 210, -90, 0, -90],
        [280, -20, 210, -90, 0, -90],
        [310, 30, 190, -90, 0, -90],
        [280, 30, 260, -90, 0, -90],
        [280, -20, 230, -90, 0, -90],
        [340, 30, 230, -70, 0, -90],
        [340, 30, 230, -50, 0, -90],
        [340, 30, 210, -90, -20, -90],
        [340, 30, 210, -90, -40, -90],
        [340, 30, 210, -90, 0, -80],
        [340, 30, 210, -90, 0, -60],
    ]

    print("=== Collecting samples ===")
    for i, pose in enumerate(pose_list):
        # 度数法をラジアンに変換
        pose_rad = pose[:3] + [math.radians(p) for p in pose[3:]]
        arm.set_position(*pose, speed=20, mvacc=2000, wait=True)
        time.sleep(0.5)

        # ===== Robot側: Base → Gripper (TCP) =====
        current_pose = arm.get_position(is_radian=True)[1]
        T_base_to_gripper = pose_to_matrix(current_pose)
        
        R_gripper = T_base_to_gripper[:3, :3]
        t_gripper = T_base_to_gripper[:3, 3]
        
        # Base → Marker の計算（オフセット考慮）
        R_marker = R_gripper @ R_offset_rot
        offset_world = R_gripper @ offset_local
        t_marker = t_gripper + offset_world
        
        # ===== Camera側: Camera → Marker =====
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
        
        # ArUcoが返すのは Marker → Camera
        T_marker_to_cam = rt_to_matrix(rvec[0], tvec[0])
        
        # ★重要: calibrateHandEye は Camera → Marker を要求するので逆行列
        T_cam_to_marker = np.linalg.inv(T_marker_to_cam)
        
        # リストに追加
        R_gripper2base.append(R_gripper)
        t_gripper2base.append(t_gripper)
        R_target2cam.append(T_cam_to_marker[:3, :3])
        t_target2cam.append(T_cam_to_marker[:3, 3])
        
        print(f"[{i}] OK - Marker at Base: {t_marker}")

    if len(R_gripper2base) < 3:
        print("❌ Not enough samples!")
        return None

    print(f"\n=== Solving with {len(R_gripper2base)} samples ===")
    
    # Hand-Eye Calibration
    # R_gripper2base, t_gripper2base: Base → Gripper
    # R_target2cam, t_target2cam: Camera → Target (Marker)
    # 返り値: Gripper → Camera の変換
    R_gripper2cam, t_gripper2cam = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Gripper → Camera の行列
    T_gripper_to_cam = np.eye(4)
    T_gripper_to_cam[:3, :3] = R_gripper2cam
    T_gripper_to_cam[:3, 3] = t_gripper2cam.reshape(3)
    
    # ★ Base → Camera を計算するには任意の姿勢で確認
    # ここでは最初の姿勢を使用
    first_pose = arm.get_position(is_radian=True)[1]
    T_base_to_gripper_first = pose_to_matrix(first_pose)
    T_base_to_cam = T_base_to_gripper_first @ T_gripper_to_cam

    print("\n===== RESULT =====")
    print("T_gripper_to_camera (Gripper → Camera):")
    print(T_gripper_to_cam)
    print("\nT_base_to_camera (Base → Camera, at current pose):")
    print(T_base_to_cam)
    print("\n推定カメラ位置 (Base座標系):")
    print(f"  X: {T_base_to_cam[0, 3]:.3f} m")
    print(f"  Y: {T_base_to_cam[1, 3]:.3f} m")
    print(f"  Z: {T_base_to_cam[2, 3]:.3f} m")
    print("==================")
    
    # コピペ用
    print("\n# コピペ用:")
    print("T_gripper_to_camera = np.array([")
    for row in T_gripper_to_cam:
        print(f"  [{row[0]}, {row[1]}, {row[2]}, {row[3]}],")
    print("])")
    
    return T_gripper_to_cam

if __name__ == "__main__":
    handeye()