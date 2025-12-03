import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time

# ============================================
# 0. Charuco ボードの設定（印刷したものと一致させる）
# ============================================
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
SQUARE_LENGTH_M   = 0.030  # 30mm
MARKER_LENGTH_M   = 0.024  # 24mm
ARUCO_DICT_NAME   = cv2.aruco.DICT_4X4_50


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
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(1)
    return arm


def get_robot_pose(arm):
    p1 = np.array(arm.get_position(is_radian=True)[1])
    time.sleep(0.05)
    p2 = np.array(arm.get_position(is_radian=True)[1])
    return (p1 + p2) / 2.0


# ============================================
# 3. Utility: Pose → 4x4 Transform
# ============================================

def pose_to_matrix_base2gripper(p):
    x, y, z, rx, ry, rz = p
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], float))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0  # mm→m
    return T


def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3,)
    return T


# ============================================
# 4. Charuco Pose Estimation: board → camera
# ============================================

def create_charuco_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    board = cv2.aruco.CharucoBoard_create(
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        aruco_dict
    )
    return aruco_dict, board


def get_charuco_pose(frame, aruco_dict, board, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if ids is None or len(ids) == 0:
        return None, None

    # interpolate charuco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if charuco_ids is None or len(charuco_ids) < 4:
        return None, None

    # estimate board pose
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
    )
    if not retval:
        return None, None

    return rvec, tvec


# ============================================
# 5. Hand–Eye Calibration (Eye-to-Hand)
# ============================================

def handeye_eye_to_hand():
    # RealSense 起動
    pipeline = init_realsense()

    # xArm 起動
    arm = init_xarm()

    # Charuco ボード
    aruco_dict, board = create_charuco_board()

    # === カメラ内部パラメータ ===
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
    R_target2cam   = []
    t_target2cam   = []

    # xArm の姿勢（任意に増やしてOK）
    pose_list = [
        [300,   0, 250,  0, 0, 0],
        [300,  40, 260,  0, 0, 20],
        [300, -40, 260,  0, 0, -20],
        [260,   0, 250,  0, 20, 0],
        [340,   0, 250,  0, -20, 0],
        [300,   0, 280, 20, 0, 0],
        [300,   0, 220, -20,0, 0],
    ]

    print("=== Start Eye-to-Hand Charuco Calibration ===")

    for i, pose in enumerate(pose_list):
        x,y,z, rx_deg,ry_deg,rz_deg = pose

        # xArm を動かす
        arm.set_position(x, y, z, rx_deg, ry_deg, rz_deg,
                         speed=20, mvacc=2000, wait=True)
        time.sleep(0.8)

        # ---------------------------
        # ① Robot: base → gripper
        # ---------------------------
        p = get_robot_pose(arm)
        T_bg = pose_to_matrix_base2gripper(p)
        T_gb = np.linalg.inv(T_bg)  # gripper → base

        # ---------------------------
        # ② Camera: board → camera
        # ---------------------------
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        rvec, tvec = get_charuco_pose(color_image, aruco_dict, board, camera_matrix, dist_coeffs)
        if rvec is None:
            print(f"[{i}] Charuco not found. Skip.")
            continue

        T_tc = rt_to_matrix(rvec, tvec)  # board→cam

        # Collect
        R_gripper2base.append(T_gb[:3,:3])
        t_gripper2base.append(T_gb[:3,3])
        R_target2cam.append(T_tc[:3,:3])
        t_target2cam.append(T_tc[:3,3])

        print(f"[{i}] Captured necessary data.")

    # =========================================
    # Hand–Eye (Eye-to-Hand)
    # =========================================
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_gc = np.eye(4)
    T_gc[:3,:3] = R_cam2gripper
    T_gc[:3,3]  = t_cam2gripper.reshape(3,)

    # base→camera を求めるために
    # 最初の姿勢の base→gripper を再取得
    p0 = get_robot_pose(arm)
    T_bg0 = pose_to_matrix_base2gripper(p0)

    # base → camera
    T_bc = T_bg0 @ T_gc

    print("\n===== RESULT: base → camera =====")
    print(T_bc)
    print("\nCamera origin (base frame):", T_bc[:3,3])

    return T_bc


# ============================================
# 6. 実行
# ============================================
if __name__ == "__main__":
    T_base_cam = handeye_eye_to_hand()

    if T_base_cam is not None:
        P_cam = np.array([0.0, 0.0, 0.30])
        P_base = transform_cam_to_base = T_base_cam @ np.array([P_cam[0], P_cam[1], P_cam[2],1])
        print("\nTransform test (camera→base):", P_base[:3])
