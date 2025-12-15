import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R
import time

# =============================
# 必要な行列パラメータ
# =============================
# --- 手眼キャリブレーションで求めた Base→Camera ---
T_base_camera = np.array([
    [ 0.86581087, -0.13502627, -0.48180851, -13.0905596 ],
    [ 0.49130086,  0.04689409,  0.86972663, -198.188808 ],
    [-0.09484197, -0.98973171,  0.10693994, -106.769127 ],
    [ 0.0,         0.0,         0.0,          1.0        ]
])


# --- TCP → Marker の位置関係（固定値） ---
# 例：マーカー中心が TCP の上方向に 3cm
T_gripper_marker = np.eye(4)
T_gripper_marker[:3, 3] = np.array([0.00, 0.00, 0.23])


# =============================
# RealSense + ArUco 設定
# =============================
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.076

fx, fy = 608.54150390625, 607.1893920898438
cx, cy = 309.4483947753906, 264.0105285644531
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)


def rt_to_matrix(R_mat, tvec):
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.reshape(3)
    return T


# =============================
# RealSense capture
# =============================
def capture_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    return np.asanyarray(color.get_data())


# =============================
# ArUco Pose Detection
# =============================
# 返り値: 4x4 行列 (Marker -> Camera)
def detect_marker_pose(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Dictionary の互換処理 ---
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    except AttributeError:
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)

    # --- DetectorParameters の互換処理 ---
    try:
        parameters = cv2.aruco.DetectorParameters()
    except:
        parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    if ids is None:
        return None

    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH, camera_matrix, dist_coeffs
    )

    R_mat, _ = cv2.Rodrigues(rvec[0][0])
    t = tvec[0][0]

    return rt_to_matrix(R_mat, t)


# =============================
# 誤差評価メイン
# =============================
def main():

    # --- RealSense ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # --- xArm ---
    arm = XArmAPI("192.168.1.199")
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1)

    print("\n=== Hand-Eye 誤差評価（直接法）===")
    print("ロボットを様々な姿勢に動かし、Enterで測定、qで終了\n")

    while True:
        key = input("Enter → 測定  /  q → 終了 : ")
        if key == "q":
            break

        # --- カメラで Marker 取得 ---
        img = capture_frame(pipeline)
        T_marker_camera = detect_marker_pose(img)
        if T_marker_camera is None:
            print("❌ マーカー未検出")
            continue

        # --- Robot Base → TCP 取得 ---
        pose = arm.get_position(is_radian=False)[1]
        x, y, z, roll, pitch, yaw = pose
        Rg = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()

        T_base_gripper = np.eye(4)
        T_base_gripper[:3, :3] = Rg
        T_base_gripper[:3, 3] = np.array([x/1000, y/1000, z/1000])


        # =============================
        # ① Robot 側の Hand (TCP) 位置
        # =============================
        P_robot = T_base_gripper[:3, 3]


        # =============================
        # ② Camera 経由の Hand 位置
        # =============================
        T_base_marker = T_base_camera @ T_marker_camera.inverse()
        T_base_hand_est = T_base_marker @ T_gripper_marker



        P_camera = T_base_hand_est[:3, 3]
        print(P_camera)
        print(P_robot)

        # =============================
        # ③ 誤差
        # =============================
        error = P_camera - P_robot
        norm_mm = np.linalg.norm(error) * 1000

        print("\n--- New Sample ---")
        print("Robot hand pos (m):     ", P_robot)
        print("Camera-estimated pos:   ", P_camera)
        print("Diff (m):               ", error)
        print(f"Error norm:             {norm_mm:.2f} mm")


    pipeline.stop()
    print("終了しました。")


if __name__ == "__main__":
    main()
