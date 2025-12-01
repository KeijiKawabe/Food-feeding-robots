import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time

# ============================================
# 1. RealSense 初期化
# ============================================

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    # ★ 640×480 のカラーストリーム（内部パラメータと一致）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    return pipeline


# ArUco Pose 推定
def get_aruco_pose(frame, aruco_dict, parameters, camera_matrix, dist_coeff):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        return None, None

    marker_length = 0.035  # 3.5cm
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeff
    )

    return rvecs[0], tvecs[0]


# ============================================
# 2. xArm 初期化
# ============================================

def init_xarm(ip="192.168.1.199"):
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.set_mode(1)   # Position mode
    arm.set_state(0)
    time.sleep(1)
    return arm


def get_robot_pose(arm):
    p1 = np.array(arm.get_position(is_radian=True)[1])
    time.sleep(0.05)
    p2 = np.array(arm.get_position(is_radian=True)[1])
    return (p1 + p2) / 2


# ============================================
# Utility: 変換行列
# ============================================

def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def pose_to_matrix(p):
    x, y, z, rx, ry, rz = p
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z])
    return T


# ============================================
# Camera → Gripper（Eye-in-Hand）座標変換
# ============================================

def transform_cam_to_gripper(P_cam, R_cam2gripper, t_cam2gripper):
    P_cam = np.array(P_cam).reshape(3,)
    t = np.array(t_cam2gripper).reshape(3,)
    return R_cam2gripper @ P_cam + t


# ============================================
#  Hand-in-Eye キャリブレーション
# ============================================

def hand_in_eye():
    pipeline = init_realsense()
    arm = init_xarm()

    # ===========================
    # ★ RealSense 内部パラメータ（固定）
    # ===========================
    fx = 389.846
    fy = 389.846
    cx = 321.177
    cy = 235.201

    camera_matrix = np.array([
        [fx,  0, cx],
        [0,  fy, cy],
        [0,   0,  1]
    ], dtype=np.float64)

    dist_coeff = np.zeros(5)

    # ArUco辞書
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    # AX = XB 用データ
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # 10〜15姿勢が理想
    pose_list = [
        [300,   0, 250, 0, 0, 0],
        [300,  40, 260, 0, 0, 20],
        [300, -40, 260, 0, 0, -20],
        [340,   0, 260, 20, 0, 0],
        [260,   0, 260, -20, 0, 0],
    ]

    print("=== Start Eye-in-Hand Calibration ===")

    for pose in pose_list:
        arm.set_position(*pose, speed=20, mvacc=2000, wait=True)
        time.sleep(0.8)

        # ① base → gripper
        p = get_robot_pose(arm)
        T_bg = pose_to_matrix(p)

        # ② cam → marker
        frame = pipeline.wait_for_frames().get_color_frame()
        color_image = np.asanyarray(frame.get_data())
        rvec, tvec = get_aruco_pose(color_image, aruco_dict, parameters, camera_matrix, dist_coeff)

        if rvec is None:
            print("Marker not detected, skipping...")
            continue

        T_cm = rt_to_matrix(rvec, tvec)

        R_gripper2base.append(T_bg[:3, :3])
        t_gripper2base.append(T_bg[:3, 3])

        R_target2cam.append(T_cm[:3, :3])
        t_target2cam.append(T_cm[:3, 3])

        print("Captured one pair.")

    # -------------------------
    # AX = XB を解く（Eye-in-Hand）
    # 出力は「カメラ → グリッパー」
    # -------------------------
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    print("\n===== RESULT: T_camera→gripper =====")
    print("Rotation:\n", R_cam2gripper)
    print("Translation:\n", t_cam2gripper)

    return R_cam2gripper, t_cam2gripper


# ============================================
# 実行テスト
# ============================================

if __name__ == "__main__":
    R, t = hand_in_eye()

    # Camera → Gripper 座標変換テスト
    P_cam = [0.05, 0.01, 0.30]

    P_gripper = transform_cam_to_gripper(P_cam, R, t)

    print("\n===== Transform Test =====")
    print("P_cam =", P_cam)
    print("P_gripper =", P_gripper)
