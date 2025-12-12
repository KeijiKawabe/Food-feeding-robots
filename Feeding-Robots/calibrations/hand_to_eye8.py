import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R
import time

# ================================
# 1. カメラ設定（Aruco）
# ================================
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.01 # 28mm

# RealSense intrinsics（必要なら置き換え）→ダメそうなら内部パラメータもキャリブレーションして算出
fx, fy = 608.54150390625, 607.1893920898438
cx, cy = 309.4483947753906, 264.0105285644531
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)


# ================================
# 2. 4x4 変換行列ユーティリティ
# ================================
def rt_to_matrix(R_mat, tvec):
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.reshape(3)
    return T

def invert(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ================================
# 3. RealSense 1フレーム取得
# ================================
def capture_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())
    return img


# ================================
# 4. ArUco Pose 取得
# ================================
def detect_marker_pose(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ✅ ここを修正
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # DetectorParameters_create も新しい書き方だとこう：
    try:
        params = cv2.aruco.DetectorParameters()  # 新しめの OpenCV
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()  # 古い OpenCV 互換

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None:
        return None, None

    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH, camera_matrix, dist_coeffs
    )

    R_mat, _ = cv2.Rodrigues(rvec[0][0])
    t = tvec[0][0]
    return R_mat, t



# ================================
# 5. AX = XB solver
# ================================
def solve_hand_eye(A_list, B_list):

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for A, B in zip(A_list, B_list):
        R_g = A[:3, :3]
        t_g = A[:3, 3]
        R_t = B[:3, :3]
        t_t = B[:3, 3]
        R_gripper2base.append(R_g)
        t_gripper2base.append(t_g)
        R_target2cam.append(R_t)
        t_target2cam.append(t_t)

    R_cam, t_cam = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R_cam, t_cam


# ================================
# 6. メイン処理
# ================================
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

    print("\n=== Eye-to-Hand Calibration ===")
    print("任意の姿勢にロボットを動かし、qキーで撮影してください。")
    print("10～20サンプルが推奨です。\n")

    robot_poses = [
        [380, -160, 240, -120, 0, -90],
        [320, -100, 240, -90, 0, -90],
        [360, -50, 260, -90, 0, -90],
        [400, 0, 220, -100, 0, -90],
        [420, 50, 240, -90, 0, -90],

        [280, 100, 260, -90, 0, -90],
        [320, 80, 280, -90, 20,-90],
        [360, 60, 220, -90, -20, -90],
        [400, 30, 300, -90, 0, -90],
        [420, -20, 260, -90, 0, -90],

        [300, -120, 300, -90, 0, -90],
        [350, -80, 220, -90, 0, -90],
        [390, 10, 270, -90, 0, -70],
        [380, 90, 250, -90, 0, -100],
        [330, 40, 230, -90, 0, -90],
    ]

    A_list = []
    B_list = []
    prev_Tg = None
    prev_Tm = None

    for i, pose_target in enumerate(robot_poses):
        print(f"\n--- [{i+1}/{len(robot_poses)}] Moving robot to pose: {pose_target} ---")

        # 1) move robot
        code = arm.set_position(
            x=pose_target[0],
            y=pose_target[1],
            z=pose_target[2],
            roll=pose_target[3],
            pitch=pose_target[4],
            yaw=pose_target[5],
            speed=50, 
            wait=True
        )

        if code != 0:
            print(f"⚠ ロボット移動エラー: code={code}")
            continue

        time.sleep(0.5)  # 安定待ち


        # ====================
        # Step 1: RealSense 画像
        # ====================
        img = capture_frame(pipeline)
        Rm, tm = detect_marker_pose(img)
        if Rm is None:
            print("❌ マーカーが検出されません。")
            continue

        T_camera_marker = rt_to_matrix(Rm, tm)

        # ====================
        # Step 2: xArm 姿勢取得
        # ====================
        pose = arm.get_position(is_radian=False)[1]
        x, y, z, roll, pitch, yaw = pose
        Rg = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
        Tg = rt_to_matrix(Rg, np.array([x/1000, y/1000, z/1000]))

        # 初回は記録だけ
        if prev_Tg is None:
            prev_Tg = Tg
            prev_Tm = T_camera_marker
            print("✓ 1枚目を登録")
            continue

        # 相対姿勢
        A = invert(prev_Tg) @ Tg             # Robot relative
        B = invert(prev_Tm) @ T_camera_marker  # Marker relative

        A_list.append(A)
        B_list.append(B)

        print(f"✓ サンプル追加（現在 {len(A_list)} 枚）")

        prev_Tg = Tg
        prev_Tm = T_camera_marker

    # ====================
    # Solve AX = XB
    # ====================
    print("\n=== Solving AX = XB ===")
    R_cam, t_cam = solve_hand_eye(A_list, B_list)

    T_base_camera = rt_to_matrix(R_cam, t_cam)
    print("\n===== RESULT: T_base_camera =====")
    print(T_base_camera)

    pipeline.stop()


if __name__ == "__main__":
    main()
