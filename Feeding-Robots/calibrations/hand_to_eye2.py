import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time

# ============================================
# 0. ArUco マーカー設定（Charuco は使わない）
# ============================================

ARUCO_DICT = cv2.aruco.DICT_6X6_250   # 印刷したマーカーと合わせる
MARKER_LENGTH_M = 0.035              # マーカーの一辺サイズ[m] 例：3.5cm


# ============================================
# 1. RealSense 初期化
# ============================================

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


# ============================================
# 2. xArm 初期化
# ============================================

def init_xarm(ip="192.168.1.199"):
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    # motion_enableが完了するまで十分に待つ
    time.sleep(2)  # 2秒間に延長

    # 1:位置制御モード (PTP)
    arm.set_mode(0)
    # 0:稼働状態 (Enable)
    arm.set_state(0)
    # モード/状態設定が反映されるまで待つ
    time.sleep(1) # ここも1秒に延長
    
    # ロボットが確実に動ける状態か確認するためのダミー移動（オプション）
    # arm.set_position(x, y, z, rx, ry, rz, is_radian=False, speed=100, wait=True)
    
    return arm

def get_robot_pose(arm):
    """xArm の TCP pose を2回読んで平均を取る"""
    p1 = np.array(arm.get_position(is_radian=True)[1])
    time.sleep(0.05)
    p2 = np.array(arm.get_position(is_radian=True)[1])
    return (p1 + p2) / 2.0


# ============================================
# 3. Utility: Pose → 4x4 Transform
# ============================================

def pose_to_matrix_base2gripper(p):
    """xArm pose → base→gripper の4x4行列"""
    x, y, z, rx, ry, rz = p
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], float))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T


def rt_to_matrix(rvec, tvec):
    """Rodrigues rvec,tvec → 4x4行列"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3,)
    return T


# ============================================
# 4. ArUco Pose Estimation（marker → camera）
# ============================================

def create_aruco():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    return aruco_dict, parameters


def get_aruco_pose(frame, aruco_dict, parameters, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is None or len(ids) == 0:
        return None, None

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
    )

    # 最初のマーカーだけ使用
    return rvecs[0], tvecs[0]


# ============================================
# 5. Hand–Eye Calibration (Eye-to-Hand)
# ============================================

def handeye_eye_to_hand():
    pipeline = init_realsense()
    arm = init_xarm()

    aruco_dict, parameters = create_aruco()

    # ===== RealSense 内部パラメータ（事前キャリブ値を入れる）=====
    fx = 389.846
    fy = 389.846
    cx = 321.177
    cy = 235.201

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    dist_coeffs = np.zeros(5)

    # Hand–Eye データ
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # ★ あなたが指定した pose_list をそのまま使用 ★
    pose_list = [
        # 基準 & 並進だけのパターン
        [340.0,  30.0, 210.0, -90.0,   0.0, -90.0],  # 基準
        [340.0, -20.0, 210.0, -90.0,   0.0, -90.0],  # 左移動
        [340.0,  30.0, 160.0, -90.0,   0.0, -90.0],  # 下移動
        [310.0,  30.0, 260.0, -90.0,   0.0, -90.0],  # 上後移動
        [280.0, -20.0, 260.0, -90.0,   0.0, -90.0],  # 後右移動

        # Roll（rx）方向：Z少し上げ目
        [340.0,  30.0, 230.0, -70.0,   0.0, -90.0],
        [340.0,  30.0, 230.0, -50.0,   0.0, -90.0],

        # Pitch（ry）方向
        [340.0,  30.0, 210.0, -90.0, -20.0, -90.0],
        [340.0,  30.0, 210.0, -90.0, -40.0, -90.0],

        # Yaw（rz）方向
        [340.0,  30.0, 210.0, -90.0,   0.0, -80.0],
        [340.0,  30.0, 210.0, -90.0,   0.0, -60.0],
    ]

    print("=== Start Eye-to-Hand Calibration (Aruco) ===")

    for i, pose in enumerate(pose_list):
        x, y, z, rx_deg, ry_deg, rz_deg = pose
        code = arm.set_position(x, y, z, rx_deg, ry_deg, rz_deg,
                        speed=20, mvacc=2000, wait=True)
        print(f"[{i}] set_position return code = {code}")

        p_now = arm.get_position(is_radian=True)
        print(f"[{i}] robot pose now = {p_now}")

        time.sleep(0.8)

        # ① base→gripper
        p = get_robot_pose(arm)
        T_bg = pose_to_matrix_base2gripper(p)
        T_gb = np.linalg.inv(T_bg)  # gripper→base

        # ② marker→camera
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if not frame:
            print(f"[{i}] No color frame. Skip.")
            continue

        img = np.asanyarray(frame.get_data())
        rvec, tvec = get_aruco_pose(img, aruco_dict, parameters, camera_matrix, dist_coeffs)
        if rvec is None:
            print(f"[{i}] ArUco not detected. Skip.")
            continue

        T_tc = rt_to_matrix(rvec, tvec)

        R_gripper2base.append(T_gb[:3, :3])
        t_gripper2base.append(T_gb[:3, 3])
        R_target2cam.append(T_tc[:3, :3])
        t_target2cam.append(T_tc[:3, 3])

        print(f"[{i}] captured OK.")

    if len(R_gripper2base) < 3:
        print("Not enough valid poses for hand-eye calibration.")
        return None

    print("\nCalibrating...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_gc = np.eye(4)
    T_gc[:3, :3] = R_cam2gripper
    T_gc[:3, 3] = t_cam2gripper.reshape(3,)

    # 最初の姿勢の base→gripper を使って base→camera を算出
    p0 = get_robot_pose(arm)
    T_bg0 = pose_to_matrix_base2gripper(p0)
    T_bc = T_bg0 @ T_gc

    print("\n===== RESULT: base → camera =====")
    print(T_bc)
    print("\nCamera origin (base frame):", T_bc[:3, 3])

    return T_bc


# ============================================
# 6. 実行
# ============================================

if __name__ == "__main__":
    T_base_cam = handeye_eye_to_hand()
