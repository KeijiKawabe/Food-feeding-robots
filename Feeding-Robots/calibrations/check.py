import numpy as np
import os

# キャリブレーションファイルのパス
CALIB_PATH = "T_Base_rgb.npy"

if not os.path.exists(CALIB_PATH):
    print(f"❌ ファイルが見つかりません: {CALIB_PATH}")
    exit(1)

# 読み込み
arr = np.load(CALIB_PATH, allow_pickle=True)

print("=== キャリブレーションファイルの内容 ===")
print(f"Type: {type(arr)}")
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")

if arr.shape == (4, 4):
    T = arr
    print("\n4x4 行列として保存されています:")
    print(T)
    
    # 分解して確認
    R = T[:3, :3]
    t = T[:3, 3]
    
    print("\n回転行列 R:")
    print(R)
    print("\n並進ベクトル t (カメラの位置):")
    print(f"  X: {t[0]:.4f} m ({t[0]*1000:.1f} mm)")
    print(f"  Y: {t[1]:.4f} m ({t[1]*1000:.1f} mm)")
    print(f"  Z: {t[2]:.4f} m ({t[2]*1000:.1f} mm)")
    
    print("\n期待値との比較:")
    print("  X ≈ 0.0m (ほぼ同じ)")
    print("  Y ≈ 0.25m (カメラがY方向に20-30cm)")
    print("  Z ≈ 0.80m (カメラが80cm上)")
    
    # 逆行列も確認
    T_inv = np.linalg.inv(T)
    print("\n逆行列 (Camera → Base) の並進ベクトル:")
    t_inv = T_inv[:3, 3]
    print(f"  X: {t_inv[0]:.4f} m")
    print(f"  Y: {t_inv[1]:.4f} m")
    print(f"  Z: {t_inv[2]:.4f} m")
    
    # 回転行列の確認（正規直交性）
    det = np.linalg.det(R)
    print(f"\n回転行列の行列式: {det:.6f} (1.0 に近いべき)")
    
    RTR = R.T @ R
    print("R^T @ R (単位行列に近いべき):")
    print(RTR)
    
    # テスト: カメラ座標の点を変換
    print("\n=== テスト変換 ===")
    P_cam = np.array([-0.075, 0.227, 1.033, 1.0])
    print(f"カメラ座標の点: {P_cam[:3]}")
    
    # Base → Camera なので、Camera → Base には逆行列
    P_base = T_inv @ P_cam
    print(f"Base座標の点 (逆行列使用): {P_base[:3]}")
    
elif arr.shape == ():
    obj = arr.item()
    print("\nDict として保存されています:")
    print(f"Keys: {obj.keys()}")
    for key, val in obj.items():
        print(f"\n{key}:")
        print(val)
else:
    print(f"\n不明な形式: {arr}")