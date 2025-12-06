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

camera_matrix = np.array([
    [389.846, 0, 321.177],
    [0, 389.846, 235.201],
    [0, 0, 1]
])
dist_coeffs = np.zeros(5)

# Hand-eye の Base→Camera 結果（あなたの値を入れる）


T_base_camera = [
[-0.77826454, -0.19968072,  0.59534184,  0.28414271],
[-0.51377954,  0.74757863, -0.42089996,  0.06078295],
[-0.36101923, -0.63344597, -0.68440581,  0.24964611],
[0., 0., 0., 1.],
]



# 🔵 重要：TCP→Marker の相対オフセット（m）
# ここを実際の測定値に置き換える
OFFSET_TCP_MARKER = np.array([0.0, 0.0, 0.028])  # 仮にTCPのZ+30mmと仮定


# ===============================
# 2. Utility
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
    T[:3,:3] = R
    T[:3,3] = np.array([x, y, z]) / 1000.0  # mm→m
    return T


def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = tvec.reshape(3)
    return T


# ===============================
# 3. Base→Marker をロボット側で計算
# ===============================
def calc_marker_from_robot(pose):
    T_bg = pose_to_matrix(pose)
    R_tcp = T_bg[:3, :3]
    tcp_pos = T_bg[:3, 3]

    # Marker = TCP + R_tcp * offset_tcp_marker
    marker_pos_robot = tcp_pos + R_tcp @ OFFSET_TCP_MARKER
    return marker_pos_robot


# ===============================
# 4. Base→Marker をカメラ側で計算
# ===============================
def calc_marker_from_camera(rvec, tvec):
    T_mc = rt_to_matrix(rvec, tvec)
    T_bm_camera = T_base_camera @ T_mc
    return T_bm_camera[:3, 3]


# ===============================
# 5. メイン：パターンCの精度測定
# ===============================
def handeye_accuracy_test():

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # xArm
    arm = XArmAPI("192.168.1.199")
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1.0)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()

    robot_positions = []
    camera_positions = []

    print("=== Hand-Eye Accuracy Test (Pattern C) ===")
    print("TCPにマーカーを固定した状態で、ロボットを様々な姿勢に動かしてください。")
    print("各姿勢で camera と robot の Base→Marker を比較し、手眼キャリブレーションの誤差を測定します。")
    print("qキーで終了。")

    while True:

        # ① カメラでマーカー検出
        frames = pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is None:
            cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                break
            continue
        time.sleep(2)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
        )

        # ② Camera 側 Base→Marker
        p_cam = calc_marker_from_camera(rvec[0], tvec[0].reshape(3))

        # ③ Robot 側 Base→Marker
        pose = arm.get_position(is_radian=True)[1]
        p_robot = calc_marker_from_robot(pose)

        # 記録
        camera_positions.append(p_cam)
        robot_positions.append(p_robot)

        # 誤差
        diff = p_cam - p_robot
        error = np.linalg.norm(diff)

        # 統計
        errors = np.linalg.norm(np.array(camera_positions) - np.array(robot_positions), axis=1)
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        max_err = np.max(errors)

        print("\n--- New Sample ---")
        print("Robot marker pos :", p_robot)
        print("Camera marker pos:", p_cam)
        print("Diff:", diff)
        print("Error (m):", error)
        print("Mean Error:", mean_err)
        print("STD:", std_err)
        print("Max Error:", max_err)

        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    handeye_accuracy_test()
