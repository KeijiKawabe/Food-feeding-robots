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
MARKER_LENGTH_M = 0.0265  # 28mm

# カメラ内部パラメータ
fx, fy = 608.54150390625, 607.1893920898438
cx, cy = 309.4483947753906, 264.0105285644531
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.zeros(5)


# ===============================
# 2. xArm / RealSense 初期化
# ===============================
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


# ===============================
# 3. xArm の姿勢 → 回転行列・平行移動
# ===============================
def pose_to_matrix(p):
    """xArm の get_position() （mm, rad）を 4×4 変換行列に変換"""

    x, y, z, rx, ry, rz = p

    # xArm の RPY → 回転行列
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
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0  # mm → m

    return T


# ===============================
# 4. solvePnP → 4×4 行列
# ===============================
def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T
import numpy as np

def residual_SE3(A, X, B):
    """
    AX = XB のズレ || A*X - X*B || を評価
    出力:
        平行移動誤差 [mm], 回転誤差 [deg]
    """
    left  = A @ X
    right = X @ B
    T_err = np.linalg.inv(left) @ right

    # 回転誤差
    R = T_err[:3, :3]
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    rot_deg = np.degrees(abs(angle))

    # 並進誤差 [mm]
    trans_mm = np.linalg.norm(T_err[:3, 3]) * 1000.0
    return trans_mm, rot_deg


def check_handeye_quality(R_g2b_list, t_g2b_list,
                          R_t2c_list, t_t2c_list,
                          X):
    """
    R_g2b_list, t_g2b_list : Gripper → Base （OpenCVの入力形式）
    R_t2c_list, t_t2c_list : Target  → Camera（OpenCVの入力形式）
    X                     : Base → Camera の4x4変換

    AX = XB の内部残渣から，Calibrationの純粋精度を評価
    """

    n = len(R_g2b_list)
    residuals_t = []
    residuals_r = []

    for i in range(n - 1):

        # ---- Base transform G_i, G_j ----
        G_i = np.eye(4)
        G_i[:3, :3] = R_g2b_list[i]
        G_i[:3, 3]  = t_g2b_list[i]

        G_j = np.eye(4)
        G_j[:3, :3] = R_g2b_list[i + 1]
        G_j[:3, 3]  = t_g2b_list[i + 1]

        # A = G_i^{-1} * G_j
        A = np.linalg.inv(G_i) @ G_j


        # ---- Camera transform C_i, C_j ----
        C_i = np.eye(4)
        C_i[:3, :3] = R_t2c_list[i]
        C_i[:3, 3]  = t_t2c_list[i]

        C_j = np.eye(4)
        C_j[:3, :3] = R_t2c_list[i + 1]
        C_j[:3, 3]  = t_t2c_list[i + 1]

        # B = C_i * C_j^{-1}
        B = C_i @ np.linalg.inv(C_j)


        # ---- residual ----
        terr, rerr = residual_SE3(A, X, B)
        residuals_t.append(terr)
        residuals_r.append(rerr)

    # Convert to arrays
    residuals_t = np.array(residuals_t)
    residuals_r = np.array(residuals_r)

    print("\n=== Hand-Eye Internal Residuals (AX = XB) ===")
    print(f"Translation RMS : {np.sqrt(np.mean(residuals_t**2)):.3f} mm")
    print(f"Rotation RMS    : {np.sqrt(np.mean(residuals_r**2)):.3f} deg")
    print(f"Max translation : {np.max(residuals_t):.3f} mm")
    print(f"Max rotation    : {np.max(residuals_r):.3f} deg")
    print("=============================================\n")


# ===============================
# 5. Hand-Eye Calibration（Eye-to-Hand）
# ===============================
def handeye():

    pipeline = init_realsense()
    arm = init_xarm()
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()

    # OpenCV が要求する入力形式
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # ↓必要な姿勢だけセット（あなたの自由）


    pose_list = [     
        [280, -150, 220, -90, 0, -90],
        [320, -100, 240, -90, 0, -90],
        [360, -50, 260, -90, 0, -90],
        [400, 0, 220, -90, 0, -90],
        [420, 50, 240, -90, 0, -90],

        [280, 100, 260, -90, 0, -90],
        [320, 80, 280, -90, 0, -90],
        [360, 60, 220, -90, 0, -90],
        [400, 30, 300, -90, 0, -90],
        [420, -20, 260, -90, 0, -90],

        [300, -120, 300, -90, 0, -90],
        [350, -80, 220, -90, 0, -90],
        [390, 10, 270, -90, 0, -90],
        [330, 40, 230, -90, 0, -90],
        [380, 90, 250, -90, 0, -90], 
        [280, 30, 240, -70,  0, -90],
        [280, 30, 240, -100,  0, -90],
        [280, 30, 240, -50,  0, -90],
        [280, 30, 240, -90,  0, -90],
        [280, 30, 240, -90,  20, -90],
        [280, 30, 240, -90,  -40, -90],
        [280, 30, 240, -90,  0, -70],
        [280, 30, 240, -90,  0, -120],
        [280, 30, 240, -90,  0, -90],
    ]


    print("=== Collecting Samples ===")

    for i, pose in enumerate(pose_list):

        arm.set_position(*pose, speed=20, mvacc=2000, wait=True)
        time.sleep(0.5)

        # ① ロボット姿勢（Base→Gripper）
        T_bg = pose_to_matrix(arm.get_position(is_radian=True)[1])
        R_bg = T_bg[:3, :3]
        t_bg = T_bg[:3, 3]

        # OpenCV は Gripper→Base が必要 → 逆行列へ変換
        R_gb = R_bg.T
        t_gb = -R_bg.T @ t_bg

        R_gripper2base.append(R_gb)
        t_gripper2base.append(t_gb)

        # ② カメラ画像からマーカー検出（Target→Camera）
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is None:
            print(f"[{i}] Marker not found.")
            continue

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
        )

        R_tc = cv2.Rodrigues(rvec[0])[0]
        t_tc = tvec[0].reshape(3)

        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)

        print(f"[{i}] OK")

    print("=== Solving Hand-Eye ===")

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.reshape(3)

    print("\n===== RESULT: T_base_camera =====")
    for row in T:
        print(row)
    print("=================================\n")

    check_handeye_quality(
        R_gripper2base, t_gripper2base,
        R_target2cam,  t_target2cam,
        T
    )
    np.save("T_Base_rgb.npy", T)

    return T


if __name__ == "__main__":
    handeye()
