import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time

# ===============================
# 1. 設定：サイズは実測値(0.035)
# ===============================
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH_M = 0.035 

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

    # ★修正1：オフセットは「手先のZ軸（突き出す方向）」に0.28m
    # GUIでX軸方向に伸びている＝手先座標系ではZ軸です
    offset_local = np.array([0, 0, 0.28]) 
    
    # ★回転オフセット：まずは「なし」で試す
    # （手先のZとマーカーのZが同じ向き＝伸びる方向を向いている場合）
    R_offset_rot = np.eye(3)

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

        # Robot: Base -> TCP
        T_bg = pose_to_matrix(arm.get_position(is_radian=True)[1])
        R_tcp = T_bg[:3,:3]
        t_tcp = T_bg[:3, 3]

        # ★修正2：マーカー位置・回転の計算
        # 回転：TCP回転 × オフセット回転
        R_marker_world = R_tcp @ R_offset_rot
        
        # 位置：TCP位置 + (TCP回転 × Z軸オフセット)
        offset_world = R_tcp @ offset_local
        marker_pos = t_tcp + offset_world
        
        T_bm = np.eye(4)
        T_bm[:3, :3] = R_marker_world
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
        # OpenCVの入力用：Target -> Camera (ArUcoの出力そのまま)
        T_mc = rt_to_matrix(rvec[0], tvec[0])

        # リストに追加
        # Base -> Marker
        R_b2m.append(T_bm[:3,:3])
        t_b2m.append(T_bm[:3,3])
        
        # Marker -> Camera
        R_c2m.append(T_mc[:3,:3])
        t_c2m.append(T_mc[:3,3])
        print(f"[{i}] OK")

    print("=== Solving... ===")
    # 戻り値は Base -> Camera
    R_base2cam, t_base2cam = cv2.calibrateHandEye(
        R_b2m, t_b2m,
        R_c2m, t_c2m,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_base_camera = np.eye(4)
    T_base_camera[:3,:3] = R_base2cam
    T_base_camera[:3, 3] = t_base2cam.reshape(3)

    print("\n===== COPY THIS MATRIX BELOW =====")
    print("T_base_camera = np.array([")
    for row in T_base_camera:
        print(f"  [{row[0]}, {row[1]}, {row[2]}, {row[3]}],")
    print("])")
    print("==================================")
    
    return T_base_camera

if __name__ == "__main__":
    handeye()