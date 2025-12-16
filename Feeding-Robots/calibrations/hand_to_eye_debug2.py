import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R
import time

# ================================
# 1. 設定 & 定数
# ================================
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.076# 単位: メートル (10mm)

# RealSense intrinsics（公式から得た値を使用）
fx, fy = 608.54150390625, 607.1893920898438
cx, cy = 309.4483947753906, 264.0105285644531
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)

# ================================
# 2. ユーティリティ(回転行列と座標行列を結合)
# ================================
def rt_to_matrix(R_mat, tvec):
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.reshape(3)
    return T

# ================================
# 3. RealSense・ArUco処理（カメラとマーカーの処理）
# ================================
def capture_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())
    return img
#　カメラ座標系に対するマーカーの位置と回転を算出、Marker -> Camera
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
    print("qを入力するとキャリブレーションを開始します。\n")

    # OpenCV calibrateHandEye に渡すためのリスト
    # Base -> Gripper
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_base2gripper_list = []
    t_base2gripper_list = []
    # Target(Marker) -> Camera (OpenCVの仕様上、Camera->MarkerではなくTarget->Camの形式で扱う場合があるが、
    # calibrateHandEyeは通常 Camera coord での Marker pose を入力する)
    R_cam2target_list = []
    t_cam2target_list = []

    sample_count = 0

    while True:
        #user_input = input(f"\n[{sample_count + 1}枚目] Enterで撮影 (qで終了): ").strip().lower()
        #if user_input == 'q':
        #    if sample_count < 3:
        #        print("⚠ 最低3枚のデータが必要です。")
        #        continue
        #    break

        # 1. 画像取得 & マーカー検出
        img = capture_frame(pipeline)
        Rm, tm = detect_marker_pose(img) # Camera -> Marker

        key=0
        if Rm is not None:
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, 
                              np.array([cv2.Rodrigues(Rm)[0]]), tm, MARKER_LENGTH * 2)
            cv2.imshow("Frame", img)
            key=cv2.waitKey(10)
        else:
            print("❌ マーカーが見つかりません。")
            cv2.imshow("Frame", img)
            cv2.waitKey(1)
            continue
        if key==ord('q'):
            break
        if key!=ord('t'):
            continue
        
        # 2. ロボット姿勢取得 (Base -> Gripper)
        # xArmは mm, degree で返ってくる
        code, pose = arm.get_position(is_radian=False)
        if code != 0:
            print("❌ ロボット通信エラー")
            continue
        # Ufactory Studioと同じようにTCPの座標が出る    
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose
        print("Gripper Pose (Base->Gripper):")
        print(pose)        
        # mm -> m 変換
        t_g = np.array([x_mm/1000.0, y_mm/1000.0, z_mm/1000.0])
        # degree -> Rotation Matrix
        R_g = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()

        print(f"  Robot (m): {t_g}")
        print(f"  Marker(m): {tm}")

        # リストに追加 (絶対姿勢をそのまま保存)
        R_gripper2base_list.append(R_g)
        t_gripper2base_list.append(t_g)
        R_base2gripper_list.append(np.transpose(R_g))
        t_base2gripper_list.append(-np.matmul(np.transpose(R_g),t_g))
        
        # 3. マーカー姿勢取得 (Target -> Camera)
        #invert this, because marker is on the robot hand
        R_c2t = np.transpose(Rm)
        t_c2t = -np.matmul(np.transpose(Rm),tm)

        R_cam2target_list.append(R_c2t)
        t_cam2target_list.append(t_c2t)

        sample_count += 1
        print(f"✓ データ追加 (計 {sample_count} 枚)")

    cv2.destroyAllWindows()
    pipeline.stop()

    if sample_count < 3:
        return

    print("\n=== 計算中 (Eye-to-Hand) ===")
    
    # OpenCVによるキャリブレーション (AX=XB)
    # Eye-to-Handの場合、以下の入力を想定:
    # 1. R_gripper2base, t_gripper2base: ロボットベースからグリッパーへの変換
    # 2. R_target2cam, t_target2cam:     カメラから見たマーカー(Target)の変換
    # 出力: R_cam2base, t_cam2base (カメラ座標系からベース座標系への変換、あるいはその逆)
    
    try:
        # R_marker2gripper, t_marker2gripper = cv2.calibrateHandEye(
        #     R_gripper2base_list,
        #     t_gripper2base_list,
        #     R_cam2target_list,
        #     t_cam2target_list,
        #     method=cv2.CALIB_HAND_EYE_TSAI
        # )
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

    # 結果: T_base_cam (ベース座標系におけるカメラの位置・姿勢)
    # calibrateHandEyeの戻り値は、Eye-to-Handの場合通常 "Base -> Camera" への変換行列
    # つまり、 Camera_Coordinates = R * Base_Coordinates + t ではなく
    # Base_Coordinates = R * Camera_Coordinates + t (またはその逆) の定義を確認する必要がある。
    # OpenCVのドキュメントでは、Eye-to-Handの出力は "rotation and translation of the camera in the robot base frame" とある。
    # つまり T_base_camera を出力する。

    #
    T_gripper2marker = np.eye(4)
#    T_base_camera[:3, :3] = R_cam
#    T_base_camera[:3, 3] = t_cam.flatten()
    T_gripper2marker[:3, :3] = R_gripper2marker
    T_gripper2marker[:3, 3] = t_gripper2marker.flatten()

    
    T_base2camera = np.eye(4)
#    T_base_camera[:3, :3] = R_cam
#    T_base_camera[:3, 3] = t_cam.flatten()
    T_base2camera[:3, :3] = R_base2camera
    T_base2camera[:3, 3] = t_base2camera.flatten()

    print("\n===== CALIBRATION RESULT =====")
    print("T_base_camera (メートル単位):")
    print(np.round(T_base2camera, 6))
    print(np.round(T_gripper2marker,6))
    print("\n【確認用】カメラの位置 (Base座標系, mm):")
    print(f"X: {T_base2camera[0,3]*1000:.2f} mm")
    print(f"Y: {T_base2camera[1,3]*1000:.2f} mm")
    print(f"Z: {T_base2camera[2,3]*1000:.2f} mm")
    print(f"Xm: {T_gripper2marker[0,3]*1000:.2f} mm")
    print(f"Ym: {T_gripper2marker[1,3]*1000:.2f} mm")
    print(f"Zm: {T_gripper2marker[2,3]*1000:.2f} mm")


    # === 検証コード ===
    print("\n=== 精度検証 ===")
    print("取得した最後のデータを使って検証します。")
    
    # 最後のデータ


    for i in range(R_gripper2base_list.len()):
        T_base2gripper = np.eye(4)
        T_base2gripper[:3, :3] = R_base2gripper_list[i]
        T_base2gripper[:3, 3] = t_base2gripper_list[i].flatten()
        T_cam2marker = np.eye(4)
        T_cam2marker[:3, :3] = R_cam2target_list[i]
        T_cam2marker[:3, 3] = t_cam2target_list[i].flatten()

        samePoint=np.matmul(np.linalg.inv(np.matmul(T_cam2marker,T_base2camera)),np.matmul(T_gripper2marker,T_base2gripper))

        #transform
        
    #T_gripper2base = rt_to_matrix(R_gripper2base_list[-1], t_gripper2base_list[-1])
    #T_camera2marker = rt_to_matrix(R_cam2target_list[-1], t_cam2target_list[-1])
    
    # 計算上のマーカー位置 (Base座標系)
    # P_base = T_base_camera * P_camera (マーカー位置)
    # T_base_marker_calc = T_base_camera @ T_camera_marker
    #T_marker2base_calc = T_base_camera @ T_camera_marker
    
    #pos_marker_in_base = T_base_marker_calc[:3, 3]
    #pos_gripper_in_base = T_base_gripper[:3, 3]
    
    #print(f"ロボット手先位置 (Base系): {pos_gripper_in_base}")
    #print(f"カメラから計算したマーカー位置 (Base系): {pos_marker_in_base}")
    
    #diff = pos_marker_in_base - pos_gripper_in_base
    #dist_error = np.linalg.norm(diff)
    
    # print(f"\nズレ (Marker - Gripper): {diff}")
    # print(f"距離誤差: {dist_error*1000:.2f} mm")
    # print("※ Eye-to-Handの場合、グリッパー位置とマーカー貼り付け位置には物理的なオフセットがあるため、")
    # print("   この「距離誤差」は『グリッパー中心からマーカーまでの距離』に近い値になるはずです。")
    # print("   （数メートルにならなければ成功です）")

if __name__ == "__main__":
    main()