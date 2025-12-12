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
MARKER_LENGTH = 0.076

# RealSense intrinsics（必要なら置き換え）
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
    R_mat = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t
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
    print("ロボットアームを任意の姿勢に動かし、Enterキーでデータを取得してください。")
    print("qを入力するとキャリブレーションを開始します。")
    print("推奨サンプル数は10～20です。\n")

    A_list = []
    B_list = []
    prev_Tg = None
    prev_Tm = None
    sample_count = 0

    while True:
        # --- ユーザー入力待ち ---
        user_input = input(f"\n[{sample_count + 1}枚目] ロボット姿勢を変更し、Enterキーで撮影（qで終了）: ").strip().lower()

        if user_input == 'q' and sample_count >= 2:
            break
        elif user_input == 'q' and sample_count < 2:
            print("⚠ キャリブレーションには最低2枚のデータが必要です。続行してください。")
            continue
        elif user_input != '' and user_input != 'q':
            print("⚠ 無効な入力です。Enterまたはqを入力してください。")
            continue


        # ====================
        # Step 1: RealSense 画像と ArUco Pose 取得
        # ====================
        img = capture_frame(pipeline)
        Rm, tm = detect_marker_pose(img)
        
        # 検出結果の表示（オプション）
        if Rm is not None:
            # 検出されたマーカーの座標系を描画
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, 
                              np.array([cv2.Rodrigues(Rm)[0]]), 
                              tm, 
                              MARKER_LENGTH * 2)
            cv2.putText(img, f"Sample: {sample_count + 1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("RealSense Frame (Press Enter to capture)", img)
        cv2.waitKey(1) # 画像表示を更新


        if Rm is None:
            print("❌ マーカーが検出されません。ロボットの姿勢またはカメラ位置を変更してください。")
            continue

        T_camera_marker = rt_to_matrix(Rm, tm)

        # ====================
        # Step 2: xArm 姿勢取得
        # ====================
        # 現在のロボットの姿勢（ベース座標系からグリッパーまでの変換）を取得
        pose = arm.get_position(is_radian=False)[1]
        x, y, z, roll, pitch, yaw = pose
        
        # m-mm 変換に注意 (xArmのpositionはmm単位、キャリブレーションはm単位)
        Rg = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
        Tg = rt_to_matrix(Rg, np.array([x/1000, y/1000, z/1000]))

        print(f"ロボット姿勢 (Base->Gripper): X={x:.2f} Y={y:.2f} Z={z:.2f} mm")
        print(f"マーカー姿勢 (Camera->Marker): T_x={tm[0]:.4f} T_y={tm[1]:.4f} T_z={tm[2]:.4f} m")

        # 初回は記録だけ
        if prev_Tg is None:
            prev_Tg = Tg
            prev_Tm = T_camera_marker
            print("✓ 1枚目（基準）を登録しました。次の姿勢へ移動してください。")
            sample_count += 1
            continue

        # 相対姿勢を計算
        # A = T_g1_g2 = T_g1_base * T_base_g2 = (T_base_g1)^-1 * T_base_g2
        A = invert(prev_Tg) @ Tg            # ロボット（ベース座標系から）の相対移動
        # B = T_m1_m2 = T_m1_cam * T_cam_m2 = (T_cam_m1)^-1 * T_cam_m2
        B = invert(prev_Tm) @ T_camera_marker # マーカー（カメラ座標系から）の相対移動
        
        A_list.append(A)
        B_list.append(B)

        print(f"✓ 相対姿勢データを追加（現在 {len(A_list)} 組）")

        # 次の基準として現在の姿勢を保存
        prev_Tg = Tg
        prev_Tm = T_camera_marker
        sample_count += 1

    cv2.destroyAllWindows()


    # ====================
    # Solve AX = XB
    # ====================
    if len(A_list) < 2:
        print("\n=== キャリブレーション失敗 ===")
        print("最低2組（3枚の画像）のデータが必要です。処理を終了します。")
        pipeline.stop()
        return

    print("\n=== Solving AX = XB (Eye-to-Hand) ===")
    # X: T_base_camera を求める
    R_cam, t_cam = solve_hand_eye(A_list, B_list)

    T_base_camera = rt_to_matrix(R_cam, t_cam)
    
    # データをmmに変換して表示
    T_base_camera_mm = T_base_camera.copy()
    T_base_camera_mm[:3, 3] *= 1000 # m -> mm

    print("\n===== RESULT: T_base_camera (カメラ基準座標系) =====")
    print("単位: 回転行列（無単位）、並進ベクトル（mm）")
    print(T_base_camera_mm)

    print(f"\n✅ キャリブレーション完了。カメラ中心位置（ベース座標系）：")
    print(f"X: {T_base_camera_mm[0, 3]:.3f} mm")
    print(f"Y: {T_base_camera_mm[1, 3]:.3f} mm")
    print(f"Z: {T_base_camera_mm[2, 3]:.3f} mm")


    # オイラー角への変換
    r = R.from_matrix(R_cam)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    print(f"\n回転（ロール, ピッチ, ヨー）: {roll:.2f}, {pitch:.2f}, {yaw:.2f} 度")

    pipeline.stop()


if __name__ == "__main__":
    main()