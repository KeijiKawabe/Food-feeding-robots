import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R
import time
import os  # <--- 追加: ディレクトリ操作用

# ================================
# 1. 設定 & 定数
# ================================
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.076 # 単位: メートル

# 保存するディレクトリの名前
SAVE_DIR = "captured_images" # <--- 追加

# RealSense intrinsics
fx, fy = 608.54150390625, 607.1893920898438
cx, cy = 309.4483947753906, 264.0105285644531
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)

# ================================
# 2. ユーティリティ
# ================================
def rt_to_matrix(R_mat, tvec):
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.reshape(3)
    return T

# ================================
# 3. RealSense・ArUco処理
# ================================
def capture_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())
    return img

def detect_marker_pose(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None:
        return None, None

    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH, camera_matrix, dist_coeffs
    )
    
    # 複数見つかった場合は最初の1つを使う
    R_mat, _ = cv2.Rodrigues(rvec[0][0])
    t = tvec[0][0] # shape: (3,)
    return R_mat, t

# ================================
# 4. メイン処理
# ================================
def main():
    # --- 画像保存用ディレクトリの作成 ---
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"ディレクトリ作成: {SAVE_DIR}")

    # --- RealSenseの起動 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # --- xArmの起動 ---
    # IPアドレスは環境に合わせて変更してください
    arm = XArmAPI("192.168.1.199")
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(1)

    print("\n=== Eye-to-Hand Calibration (Corrected) ===")
    print("単位はすべて [メートル] で計算します。")
    print("マーカー認識中... 't'キーで撮影＆データ追加、'q'キーで終了。\n")

    # データ格納用リスト
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_base2gripper_list = []
    t_base2gripper_list = []
    R_cam2target_list = []
    t_cam2target_list = []

    sample_count = 0

    while True:
        # 1. 画像取得 & マーカー検出
        img = capture_frame(pipeline)
        
        # 描画用にコピーを作成（保存用画像にも軸を描画したいため）
        display_img = img.copy()
        
        Rm, tm = detect_marker_pose(display_img) # Camera -> Marker

        key = 0
        if Rm is not None:
            cv2.drawFrameAxes(display_img, camera_matrix, dist_coeffs, 
                              np.array([cv2.Rodrigues(Rm)[0]]), tm, MARKER_LENGTH * 0.3)
            cv2.imshow("Frame", display_img)
            key = cv2.waitKey(10) & 0xFF
        else:
            # マーカーが見つからない場合
            cv2.putText(display_img, "Marker Not Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Frame", display_img)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            continue

        if key == ord('q'):
            break
        
        # 't'キーが押されたらデータ記録
        if key != ord('t'):
            continue
        
        # 2. ロボット姿勢取得
        code, pose = arm.get_position(is_radian=False)
        if code != 0:
            print("❌ ロボット通信エラー")
            continue
            
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose
        print(f"\n[{sample_count + 1}] Gripper Pose capture success.")
        
        # mm -> m 変換
        t_g = np.array([x_mm/1000.0, y_mm/1000.0, z_mm/1000.0])
        R_g = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()

        # リストに追加
        R_gripper2base_list.append(R_g)
        t_gripper2base_list.append(t_g)
        R_base2gripper_list.append(np.transpose(R_g))
        t_base2gripper_list.append(-np.matmul(np.transpose(R_g),t_g))
        
        # マーカー姿勢 (Target -> Camera)
        R_c2t = np.transpose(Rm)
        t_c2t = -np.matmul(np.transpose(Rm),tm)

        R_cam2target_list.append(R_c2t)
        t_cam2target_list.append(t_c2t)

        # --- 画像保存処理 (追加部分) ---
        # ファイル名: capture_0.png, capture_1.png ...
        image_filename = os.path.join(SAVE_DIR, f"capture_{sample_count}.png")
        cv2.imwrite(image_filename, display_img)
        print(f"   画像保存: {image_filename}")

        sample_count += 1
        print(f"✓ データ追加 (計 {sample_count} 枚)")

    cv2.destroyAllWindows()
    pipeline.stop()

    if sample_count < 3:
        print("データ不足のため終了します。")
        return

    print("\n=== 計算中 (Eye-to-Hand) ===")
    
    try:
        R_base2camera, t_base2camera, R_gripper2marker, t_gripper2marker = cv2.calibrateRobotWorldHandEye(
            R_cam2target_list,
            t_cam2target_list,
            R_base2gripper_list,
            t_base2gripper_list,
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        )
    except cv2.error as e:
        print(f"計算エラー: {e}")
        return

    # 結果行列の構築
    T_gripper2marker = np.eye(4)
    T_gripper2marker[:3, :3] = R_gripper2marker
    T_gripper2marker[:3, 3] = t_gripper2marker.flatten()

    T_base2camera = np.eye(4)
    T_base2camera[:3, :3] = R_base2camera
    T_base2camera[:3, 3] = t_base2camera.flatten()

    print("\n===== CALIBRATION RESULT =====")
    print("T_base_camera (メートル単位):")
    print(np.round(T_base2camera, 6))
    print("T_gripper_marker (メートル単位):")
    print(np.round(T_gripper2marker, 6))

    # === 検証コード ===
    print("\n=== 精度検証ループ ===")
    
    # 修正: .len() ではなく len() 関数を使用
    for i in range(len(R_gripper2base_list)):
        T_base2gripper = np.eye(4)
        T_base2gripper[:3, :3] = R_base2gripper_list[i]
        T_base2gripper[:3, 3] = t_base2gripper_list[i].flatten()
        
        T_cam2marker = np.eye(4)
        T_cam2marker[:3, :3] = R_cam2target_list[i]
        T_cam2marker[:3, 3] = t_cam2target_list[i].flatten()

        # 確認用: 全体の座標変換チェーンが整合しているか確認する場合の計算例
        # Base -> Camera -> Marker -> Gripper -> Base (一周して単位行列になるべき、等の確認)
        # ここでは計算のみ行い、print等は必要に応じて追加してください
        samePoint = np.matmul(
            np.linalg.inv(np.matmul(T_cam2marker, T_base2camera)),
            np.matmul(T_gripper2marker, T_base2gripper)
        )
        # 必要であればここで誤差を表示
        # print(f"Check {i}: \n{np.round(samePoint, 4)}")

if __name__ == "__main__":
    main()