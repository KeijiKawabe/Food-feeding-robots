import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R
import time

# ==========================================
# 1. ユーザー設定パラメータ (ここを修正してください)
# ==========================================

# 【必須】キャリブレーション結果の T_base_camera (4x4)
# 単位: メートル
T_base_camera = np.array([
    [ 0.86581087, -0.13502627, -0.48180851, -0.01309 ],  # x (例)
    [ 0.49130086,  0.04689409,  0.86972663, -0.19818 ],  # y (例)
    [-0.09484197, -0.98973171,  0.10693994, -0.10676 ],  # z (例)
    [ 0.0,         0.0,         0.0,         1.0     ]
])

# 【必須】Gripper(TCP) から Marker へのオフセット
# 「ロボットの手先(TCP)から見て、マーカーの中心はどこにあるか？」
# ※ ここがズレていると、正確な検証ができません。ノギス等で正確に測ってください。
T_tcp_to_marker = np.eye(4)
# 例: マーカーがTCPの Z軸方向に +3cm, Y軸方向に +1cm の位置に貼ってある場合
# T_tcp_to_marker[0, 3] = 0.00  # x (m)
# T_tcp_to_marker[1, 3] = 0.01  # y (m)
# T_tcp_to_marker[2, 3] = 0.03  # z (m)
# 今回は仮に「TCPの真下(Z方向) 0mm」として単位行列のままにします（必要に応じて変更してください）
T_tcp_to_marker[:3, 3] = np.array([0.00, 0.00, 0.25])

# xArm IPアドレス
ROBOT_IP = "192.168.1.199"

# ArUco設定
ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.076  # 単位: メートル

# ==========================================
# 2. ユーティリティ関数
# ==========================================
def get_camera_intrinsics(profile):
    """RealSenseのストリームから内部パラメータを動的に取得"""
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                              [0, intr.fy, intr.ppy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs, dtype=np.float32)
    return camera_matrix, dist_coeffs

def rt_to_matrix(rvec, tvec):
    """回転ベクトルと並進ベクトルを4x4行列に変換"""
    R_mat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.reshape(3)
    return T

def detect_marker(image, camera_matrix, dist_coeffs):
    """画像からマーカーを検出し、Camera->Marker行列を返す"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ArUcoバージョンの互換性対応
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        # 古いOpenCVの場合
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None:
        return None, None, None

    # Pose Estimation
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH, camera_matrix, dist_coeffs
    )
    
    # 1つ目のマーカーを使用
    rvec = rvecs[0][0]
    tvec = tvecs[0][0]
    
    T_cam_marker = rt_to_matrix(rvec, tvec)
    return T_cam_marker, rvec, tvec

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    # --- RealSense 初期化 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    # 内部パラメータを自動取得
    camera_matrix, dist_coeffs = get_camera_intrinsics(profile)

    # --- xArm 初期化 ---
    print(f"Connecting to xArm at {ROBOT_IP}...")
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1)

    print("\n" + "="*50)
    print(" Eye-to-Hand 精度検証 (Verification)")
    print("="*50)
    print(" [手順]")
    print(" 1. ロボットを手動またはスクリプトで動かす")
    print(" 2. Enterキーを押すと、その位置で誤差を計算")
    print(" 3. 'q' を押すと終了")
    print("-" * 50)

    try:
        while True:
            # ユーザー入力待機
            key = input("\nWait Input (Enter: Measure, q: Quit) > ").strip().lower()
            if key == 'q':
                break

            # ---------------------------
            # A. カメラ計測 (Visual Path)
            # ---------------------------
            # 1. 画像取得
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())

            # 2. マーカー検出 (Camera -> Marker)
            T_cam_marker, rvec, tvec = detect_marker(img, camera_matrix, dist_coeffs)

            if T_cam_marker is None:
                print("❌ マーカーが見つかりません。")
                cv2.imshow("View", img)
                cv2.waitKey(1)
                continue
            
            # 3. 座標変換計算
            # 経路: Base -> Camera -> Marker -> TCP (Gripper)
            # T_base_tcp(est) = T_base_cam * T_cam_marker * (T_tcp_marker)^-1
            
            T_base_marker = T_base_camera @ T_cam_marker
            T_base_tcp_est = T_base_marker @ np.linalg.inv(T_tcp_to_marker)
            
            pos_est = T_base_tcp_est[:3, 3] # 推定されたTCP位置 (x, y, z)

            # (表示用) 軸描画
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH)
            cv2.imshow("View", img)
            cv2.waitKey(1)


            # ---------------------------
            # B. ロボット実測 (Robot Path)
            # ---------------------------
            # 経路: Base -> TCP (直接取得)
            code, pose = arm.get_position(is_radian=False)
            if code != 0:
                print("❌ ロボット通信エラー")
                continue
            
            # [mm, degree] -> [m, rad] -> Matrix
            x, y, z, roll, pitch, yaw = pose
            t_robot = np.array([x, y, z]) / 1000.0 # mm to meter
            
            # (位置比較だけなら回転行列は不要だが、念のため作成)
            R_robot = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
            T_base_tcp_real = np.eye(4)
            T_base_tcp_real[:3, :3] = R_robot
            T_base_tcp_real[:3, 3] = t_robot
            
            pos_real = t_robot


            # ---------------------------
            # C. 誤差評価
            # ---------------------------
            diff = pos_est - pos_real
            error_mm = np.linalg.norm(diff) * 1000.0

            print("\n--- [Result] ---")
            print(f"Robot Real (Base->TCP):  [x={pos_real[0]:.4f}, y={pos_real[1]:.4f}, z={pos_real[2]:.4f}] m")
            print(f"Visual Est (Base->TCP):  [x={pos_est[0]:.4f}, y={pos_est[1]:.4f}, z={pos_est[2]:.4f}] m")
            print("-" * 30)
            print(f"Diff (x,y,z):            [{diff[0]*1000:.1f}, {diff[1]*1000:.1f}, {diff[2]*1000:.1f}] mm")
            print(f"★ Total Error (Dist):     {error_mm:.2f} mm")
            
            if error_mm > 20.0:
                 print("⚠️ 誤差が大きすぎます。T_tcp_to_marker のオフセット設定か、キャリブレーションを見直してください。")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        arm.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()