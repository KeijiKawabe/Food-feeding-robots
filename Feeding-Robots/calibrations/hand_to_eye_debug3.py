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
MARKER_LENGTH = 0.076  # 単位: メートル

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
    t = tvec[0][0]  # shape: (3,)
    return R_mat, t

# ================================
# 4. メイン処理
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
    print("単位はすべて [メートル] で計算します。")
    print("\n【使い方】")
    print("1. ロボットを手動で異なる位置・姿勢に動かす")
    print("2. マーカーがカメラに映る位置で Enter キーを押してデータ取得")
    print("3. 最低3箇所、推奨5箇所以上のデータを取得")
    print("4. 'q' を入力してキャリブレーション計算を実行\n")
    
    # OpenCV calibrateHandEye に渡すためのリスト
    # Gripper -> Base (^gT_b) ※OpenCVの入力名は R_gripper2base
    R_gripper2base_list = []
    t_gripper2base_list = []
    
    # Target -> Camera (^tT_c) ※OpenCVの入力名は R_target2cam
    R_target2cam_list = []
    t_target2cam_list = []
    
    sample_count = 0
    
    # リアルタイムプレビュー用のループ
    print("📹 カメラプレビュー開始...")
    print("   マーカーが見える位置にロボットを動かしてください\n")
    
    while True:
        # リアルタイムでフレームを表示
        img = capture_frame(pipeline)
        Rm_preview, tm_preview = detect_marker_pose(img)
        
        if Rm_preview is not None:
            # マーカーが検出されている場合は軸を描画
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs,
                             np.array([cv2.Rodrigues(Rm_preview)[0]]), 
                             tm_preview, MARKER_LENGTH * 2)
            cv2.putText(img, "Marker Detected - Ready!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img, "No Marker Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(img, f"Samples: {sample_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Frame", img)
        
        # キー入力待機（1ms）
        key = cv2.waitKey(1) & 0xFF
        
        # Enter キーでデータ取得
        if key == 10: # Enter キー
            if Rm_preview is None:
                print("❌ マーカーが見つかりません。ロボットを動かしてマーカーをカメラに映してください。")
                continue
            
            # マーカー検出成功
            Rm, tm = Rm_preview, tm_preview  # これは Camera -> Marker (^cT_t)
            
            # ロボット姿勢取得 (Base -> Gripper)
            code, pose = arm.get_position(is_radian=False)
            if code != 0:
                print("❌ ロボット通信エラー")
                continue
            
            x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose
            
            # mm -> m 変換
            t_base2gripper = np.array([x_mm/1000.0, y_mm/1000.0, z_mm/1000.0])
            # degree -> Rotation Matrix (Base -> Gripper)
            R_base2gripper = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
            
            # OpenCVの入力形式に合わせて変換
            # 1. Base -> Gripper を Gripper -> Base に変換
            R_gripper2base = R_base2gripper.T
            t_gripper2base = -R_base2gripper.T @ t_base2gripper
            
            # 2. Camera -> Target を Target -> Camera に変換
            R_target2cam = Rm.T
            t_target2cam = -Rm.T @ tm
            
            print(f"\n[{sample_count + 1}枚目] データ取得成功!")
            print(f"  ロボット位置 (Base->Gripper, m): X={t_base2gripper[0]:.3f}, Y={t_base2gripper[1]:.3f}, Z={t_base2gripper[2]:.3f}")
            print(f"  マーカー位置 (Camera->Target, m): X={tm[0]:.3f}, Y={tm[1]:.3f}, Z={tm[2]:.3f}")
            
            # リストに追加
            R_gripper2base_list.append(R_gripper2base)
            t_gripper2base_list.append(t_gripper2base)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(t_target2cam)
            
            sample_count += 1
            print(f"✓ データ追加完了 (計 {sample_count} 枚)")
            
            if sample_count >= 3:
                print("\n→ 次の位置にロボットを動かして Enter、または 'q' で計算開始")
            else:
                print(f"\n→ あと {3 - sample_count} 枚必要です。次の位置にロボットを動かして Enter")
        
        # 'q' キーでキャリブレーション開始
        elif key == ord('q'):
            if sample_count < 3:
                print(f"\n⚠ 最低3枚のデータが必要です。現在 {sample_count} 枚")
                continue
            print("\n🔄 キャリブレーション計算を開始します...")
            break
    
    cv2.destroyAllWindows()
    pipeline.stop()
    
    print("\n=== 計算中 (Eye-to-Hand) ===")
    
    # OpenCVによるキャリブレーション
    # Eye-to-Hand: Camera -> Base (^cT_b) を求める
    # 出力名は R_cam2gripper だが、Eye-to-Handでは実質 Camera -> Base
    try:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base_list,
            t_gripper2base_list,
            R_target2cam_list,
            t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
    except cv2.error as e:
        print(f"❌ 計算エラー: {e}")
        return
    
    # 結果: T_cam2base (Camera -> Base の変換)
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()
    
    # 実用上は Base -> Camera が欲しい場合が多いので逆変換も計算
    T_base2cam = np.linalg.inv(T_cam2base)
    
    print("\n" + "="*50)
    print("  CALIBRATION RESULT")
    print("="*50)
    print("\nT_cam2base (Camera -> Base) [メートル単位]:")
    print(np.round(T_cam2base, 6))
    
    print("\nT_base2cam (Base -> Camera) [メートル単位]:")
    print(np.round(T_base2cam, 6))
    
    print("\n【確認用】カメラの位置 (Base座標系):")
    print(f"  X: {T_base2cam[0,3]*1000:.2f} mm")
    print(f"  Y: {T_base2cam[1,3]*1000:.2f} mm")
    print(f"  Z: {T_base2cam[2,3]*1000:.2f} mm")
    
    # === 検証コード ===
    print("\n" + "="*50)
    print("  精度検証")
    print("="*50)
    print("取得した全データで精度を確認します。\n")
    
    total_error = 0.0
    for i in range(len(R_gripper2base_list)):
        # Base -> Gripper に戻す
        R_base2gripper_i = R_gripper2base_list[i].T
        t_base2gripper_i = -R_gripper2base_list[i].T @ t_gripper2base_list[i]
        T_base2gripper_i = rt_to_matrix(R_base2gripper_i, t_base2gripper_i)
        
        # Target -> Camera に戻す (Camera -> Target)
        R_cam2target_i = R_target2cam_list[i].T
        t_cam2target_i = -R_target2cam_list[i].T @ t_target2cam_list[i]
        T_cam2target_i = rt_to_matrix(R_cam2target_i, t_cam2target_i)
        
        # Base -> Marker を計算
        # Base -> Marker = Base -> Camera @ Camera -> Marker
        T_base2marker = T_base2cam @ T_cam2target_i
        
        pos_marker_in_base = T_base2marker[:3, 3]
        pos_gripper_in_base = T_base2gripper_i[:3, 3]
        
        diff = pos_marker_in_base - pos_gripper_in_base
        dist_error = np.linalg.norm(diff)
        total_error += dist_error
        
        print(f"[サンプル {i+1}]")
        print(f"  ロボット手先位置 (Base系): {pos_gripper_in_base}")
        print(f"  マーカー位置推定値 (Base系): {pos_marker_in_base}")
        print(f"  距離誤差: {dist_error*1000:.2f} mm\n")
    
    avg_error = total_error / len(R_gripper2base_list)
    print("="*50)
    print(f"平均距離誤差: {avg_error*1000:.2f} mm")
    print("="*50)
    print("\n※ Eye-to-Handの場合、グリッパー位置とマーカー貼り付け位置には")
    print("   物理的なオフセットがあるため、この距離誤差は")
    print("   『グリッパー中心からマーカーまでの距離』に近い値になります。")
    print("   （数十mm〜数百mm程度が正常です）")

if __name__ == "__main__":
    main()