import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time

# =====================================================
# 1. hand_to_eye で求めた base→camera をそのまま貼る
# =====================================================

T_base_camera = np.array([
[-0.7913821,  -0.23014349,  0.56634649,  0.30084269],
[-0.5079836,   0.7629715,  -0.39978389,  0.03321184],
[-0.34009857, -0.60407654, -0.7207111,   0.26753683],
[0., 0., 0., 1.],
])

# =====================================================
# 2. TCP→Marker オフセット（TCPローカルX方向 28cm）
# =====================================================
MARKER_OFFSET = 0.028  # [m]


# =====================================================
# Utility
# =====================================================
# ===== 2. pose_to_matrix を統一（キャリブレーション側を修正）=====
def pose_to_matrix(p):
    """xArmのpose (mm, deg) → 4x4行列"""
    x, y, z, rx, ry, rz = p
    
    # xArmはdegreeで返すことが多いので、is_radian=Trueを確認
    # もし度数法なら: rx, ry, rz = np.radians([rx, ry, rz])
    
    # RPY (Roll-Pitch-Yaw) から回転行列
    # xArmの回転順序: Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0  # mm → m
    return T

def rt_to_matrix(rvec, tvec):
    """ArUco rvec,tvec → 4×4 行列 (marker→camera)"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


# =====================================================
# 3. RealSense setup
# =====================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

camera_matrix = np.array([
    [389.846,   0.    , 321.177],
    [  0.    , 389.846, 235.201],
    [  0.    ,   0.   ,   1.   ]
])
dist_coeffs = np.zeros(5)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters()


# =====================================================
# 4. xArm setup
# =====================================================
arm = XArmAPI("192.168.1.199")
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)
time.sleep(1.0)


# =====================================================
# 5. Verification Loop
# =====================================================
while True:
    # ===== Robot side: base→marker =====
    pose = arm.get_position(is_radian=True)[1]        # [x,y,z,rx,ry,rz]
    T_bg = pose_to_matrix(pose)                       # base→TCP

    R_tcp = T_bg[:3, :3]
    tcp_pos = T_bg[:3, 3]

    # TCP ローカル X方向 0.28m を世界座標に変換
    offset_world = R_tcp @ np.array([0.0, 0.0, MARKER_OFFSET])

    marker_pos_robot = tcp_pos + offset_world

    T_bm_robot = np.eye(4)
    T_bm_robot[:3, :3] = R_tcp       # 回転はTCPと同じと仮定
    T_bm_robot[:3, 3] = marker_pos_robot

    # ===== Camera side: base→marker =====
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("No color frame")
        continue

    img = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None:
        print("Marker missing")
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, 0.027, camera_matrix, dist_coeffs
    )

    # 最初のマーカーだけ使用
    rvec = rvecs[0]
    tvec = tvecs[0]

    T_mc = rt_to_matrix(rvec, tvec)         # marker→camera
    T_bm_camera = T_base_camera @ T_mc      # base→marker

    # ===== Compare two marker positions =====
    p_cam = T_bm_camera[:3, 3].astype(float).reshape(3,)
    p_robot = T_bm_robot[:3, 3].astype(float).reshape(3,)

    # 念のため shape を出す（デバッグ用）
    print("p_cam shape   :", p_cam.shape)
    print("p_robot shape :", p_robot.shape)

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
