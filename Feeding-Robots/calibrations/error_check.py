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
MARKER_LENGTH_M = 0.028  # 28 mm

camera_matrix = np.array([
    [389.846, 0, 321.177],
    [0, 389.846, 235.201],
    [0, 0, 1]
])
dist_coeffs = np.zeros(5)

# Hand-eye で求めた Base→Camera
T_base_camera = np.array([
    [-0.79506212, -0.22989224,  0.56127158,  0.30186471],
    [-0.50617810,  0.76131737, -0.40519081,  0.03398601],
    [-0.33415558, -0.60625524, -0.72166102,  0.26712413],
    [0.0, 0.0, 0.0, 1.0],
])

# マーカーのオフセット（TCP→Marker の距離）
MARKER_OFFSET = 0.028  # [m] = 28mm

# マーカーの上 +10mm の安全マージン
SAFE_Z_OFFSET = 0.05  # [m] = 10mm


# ===============================
# 2. ワークスペースチェック
# ===============================
def CheckIfNewPositionInWorkspace(x, y, z):
    """
    x, y, z: [mm] 単位の Base 座標
    True ならワークスペース内、False なら外
    """
    if x > 680 or x < 300:
        return False
    if y < -330 or y > 420:
        return False
    if z < 94 or z > 550:
        return False
    return True


# ===============================
# 3. xArm 姿勢 → 4×4行列
# ===============================
def pose_to_matrix(p):
    x, y, z, rx, ry, rz = p

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
# 5. タスク誤差測定メイン
# ===============================
def task_error_test():

    # --- RealSense 初期化 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # --- xArm 初期化 ---
    arm = XArmAPI("192.168.1.199")
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()

    print("=== Task Error Test (marker上 +10mm に移動) ===")

    while True:
        # ① カメラで Marker を検出
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

        # Marker→Camera
        T_mc = rt_to_matrix(rvecs[0], tvecs[0])

        # Base→Marker（カメラ推定）
        T_bm_camera = T_base_camera @ T_mc
        p_marker = T_bm_camera[:3, 3]  # [m]

        # マーカー上 +10mm の安全ターゲット
        safe_target = p_marker + np.array([0.0, 0.0, SAFE_Z_OFFSET])

        # TCPオフセットを考慮した TCP 目標位置（[m]）
        tcp_goal = safe_target - np.array([0.0, 0.0, MARKER_OFFSET])

        # m → mm 変換
        x_mm, y_mm, z_mm = tcp_goal * 1000.0

        print("\n--- New Target ---")
        print("Marker (base) [m]:          ", p_marker)
        print("Safe target Z+10mm [m]:     ", safe_target)
        print("TCP goal (base) [m]:        ", tcp_goal)
        print("TCP goal (base) [mm]:       ", x_mm, y_mm, z_mm)

        # ② ワークスペースチェック
        if not CheckIfNewPositionInWorkspace(x_mm, y_mm, z_mm):
            print("⚠ Workspace limit exceeded! Robot will NOT move.")
            cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        print("✅ In workspace. Moving robot...")

        # ③ ロボットを TCP ゴール位置へ移動
        arm.set_position(
            float(x_mm), float(y_mm), float(z_mm),
            speed=20, mvacc=2000,
            wait=True
        )
        time.sleep(0.5)

        # ④ 実際の TCP 位置を取得
        tcp_actual_pose = arm.get_position(is_radian=True)[1]
        T_bg_actual = pose_to_matrix(tcp_actual_pose)
        p_robot = T_bg_actual[:3, 3]  # [m]

        # ⑤ タスク誤差を計算（安全ターゲットに対する誤差でもOK）
        task_error = np.linalg.norm(p_robot - tcp_goal)

        print("Actual TCP (base) [m]:      ", p_robot)
        print("Task Error (m):             ", task_error)

        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    task_error_test()
