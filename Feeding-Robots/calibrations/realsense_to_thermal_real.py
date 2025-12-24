#外部パラメータを確かめるコード
import os
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# プロジェクトルート
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.thermal.pi160_controller import PI160Controller

# ========= 設定 =========
EXTR_NPZ = "./out/pi160_to_realsense_extrinsics.npz"   # R_th2rgb, t_th2rgb が入ってるやつ
TH_INTR_NPZ = "./out/pi160_intrinsics_live.npz"        # K, dist が入ってるやつ

RS_W, RS_H, RS_FPS = 640, 480, 30
THERMAL_SCALE = 4          # 表示拡大
USE_THERMAL_RAW = True    # True: raw表示(要変換) / False: palette表示
RAW_TO_TEMP = True         # raw->℃変換（表示用）
# =======================


def invert_extrinsic(R_th2rgb: np.ndarray, t_th2rgb: np.ndarray):
    """
    X_rgb = R_th2rgb * X_th + t_th2rgb
    -> X_th = R_rgb2th * X_rgb + t_rgb2th
    """
    R_rgb2th = R_th2rgb.T
    t_rgb2th = -R_rgb2th @ t_th2rgb
    return R_rgb2th, t_rgb2th


def capture_thermal_once(cam: PI160Controller):
    # バッファクリア
    cam.get_palette_image()
    cam.get_thermal_data()
    time.sleep(0.03)
    data = cam.get_thermal_data()       # uint16 (120,160)
    pal  = cam.get_palette_image()      # uint8 (120,160,3)
    return data, pal

def apply_th_uv_transform(u, v, W=160, H=120, mode=0):
        """
        mode:
        0: none
        1: flip_x (左右反転)
        2: flip_y (上下反転)
        3: rot180
        4: rot90_cw
        5: rot90_ccw
        6: transpose (x<->y)
        7: transverse (transpose+rot180)
        """
        u = float(u); v = float(v)

        if mode == 0:
            return u, v
        if mode == 1:  # flip x
            return (W - 1 - u), v
        if mode == 2:  # flip y
            return u, (H - 1 - v)
        if mode == 3:  # rot180
            return (W - 1 - u), (H - 1 - v)
        if mode == 4:  # rot90 cw: (u,v)->(H-1-v, u)  ※出力画像サイズは (H,W) になる点に注意
            return (H - 1 - v), u
        if mode == 5:  # rot90 ccw: (u,v)->(v, W-1-u)
            return v, (W - 1 - u)
        if mode == 6:  # transpose
            return v, u
        if mode == 7:  # transverse
            return (H - 1 - v), (W - 1 - u)

        return u, v


def thermal_to_gray8(img: np.ndarray):
    """表示/簡易検証用（8bitへ正規化）"""
    if img is None:
        return None
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()
    if g.dtype == np.uint16:
        g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif g.dtype in (np.float32, np.float64):
        g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g8 = g.astype(np.uint8)
    return g8


class ClickState:
    def __init__(self):
        self.u = None
        self.v = None

    def set(self, u, v):
        self.u = int(u)
        self.v = int(v)

    def clear(self):
        self.u = None
        self.v = None

    def has(self):
        return self.u is not None and self.v is not None
    




def main():
    # ---- load extrinsics ----
    if not os.path.exists(EXTR_NPZ):
        print(f"✗ extrinsics npz not found: {EXTR_NPZ}")
        return
    ex = np.load(EXTR_NPZ, allow_pickle=True)
    R_th2rgb = ex["R_th2rgb"].astype(np.float64)
    t_th2rgb = ex["t_th2rgb"].astype(np.float64).reshape(3, 1)

    R_rgb2th, t_rgb2th = invert_extrinsic(R_th2rgb, t_th2rgb)

    print("Loaded extrinsics:")
    print("R_th2rgb:\n", R_th2rgb)
    print("t_th2rgb:\n", t_th2rgb.reshape(3))
    print("Using inverse for projection: RGB -> Thermal")

    # ---- load thermal intrinsics ----
    if not os.path.exists(TH_INTR_NPZ):
        print(f"✗ thermal intrinsics npz not found: {TH_INTR_NPZ}")
        return
    th = np.load(TH_INTR_NPZ, allow_pickle=True)
    K_th = th["K"].astype(np.float64)
    dist_th = th["dist"].astype(np.float64).reshape(-1, 1)

    # ---- init PI160 ----
    cam_th = PI160Controller()
    if (not getattr(cam_th, "lib", None)) or (not getattr(cam_th, "handle", None)):
        print("✗ Thermal camera not initialized.")
        return

    # ---- init RealSense (color+depth aligned to color) ----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
    config.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    # warmup
    for _ in range(15):
        pipeline.wait_for_frames()

    # color intrinsics（クリック座標と一致させる）
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr_color = color_stream.get_intrinsics()

    click = ClickState()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click.set(x, y)
            print(f"[click] RGB pixel = ({x}, {y})")

    cv2.namedWindow("RGB")
    cv2.setMouseCallback("RGB", on_mouse)

    print("\n=== RealSense -> PI160 projection test ===")
    print("操作:")
    print("  RGBウィンドウを左クリック: その点をPI160へ投影")
    print("  r: クリック解除")
    print("  q/ESC: 終了\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            rgb = np.asanyarray(color.get_data())  # (H,W,3) BGR

            # Thermal 1枚取る
            # data, pal = capture_thermal_once(cam_th)
            # if USE_THERMAL_RAW or pal is None:
            #     src = data
            #     if RAW_TO_TEMP and (src is not None) and (src.dtype == np.uint16):
            #         src = src.astype(np.float32) * 0.1 - 100.0
            #     th_vis = thermal_to_gray8(src)
            #     th_vis = cv2.cvtColor(th_vis, cv2.COLOR_GRAY2BGR) if th_vis is not None else np.zeros((120, 160, 3), np.uint8)
            # else:
            #     th_vis = pal.copy()  # (120,160,3)
            data, pal = capture_thermal_once(cam_th)

            # --- raw固定で表示（座標系統一） ---
            src = data  # uint16 (120,160)

            if src is not None and src.dtype == np.uint16:
                if RAW_TO_TEMP:
                    # ℃に変換（表示用）
                    src = src.astype(np.float32) * 0.1 - 100.0

            th8 = thermal_to_gray8(src)  # 8bit gray
            if th8 is None:
                th_vis = np.zeros((120, 160, 3), np.uint8)
            else:
                th_vis = cv2.cvtColor(th8, cv2.COLOR_GRAY2BGR)  # 表示はBGRに揃える


            # 表示用に拡大
            th_show = cv2.resize(th_vis, (160 * THERMAL_SCALE, 120 * THERMAL_SCALE), interpolation=cv2.INTER_NEAREST)

            # クリック点があるなら、depth->3D->Thermal->投影
            if click.has():
                u, v = click.u, click.v

                # depth(m)
                z = float(depth.get_distance(u, v))
                if z <= 0.0:
                    cv2.putText(rgb, "depth=0 (invalid) - move target", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # 1) RGBピクセル -> RGBカメラ座標 3D (m)
                    # rs2_deproject_pixel_to_point は intrinsics と depth(m) で [x,y,z] を返す
                    X_rgb = np.array(rs.rs2_deproject_pixel_to_point(intr_color, [u, v], z), dtype=np.float64).reshape(3, 1)

                    # 2) RGB座標 -> Thermal座標（逆変換）
                    X_th = (R_rgb2th @ X_rgb) + t_rgb2th

                    # 3) Thermal座標(3D) -> Thermal画像へ投影
                    # objectPoints をカメラ座標そのものとして扱うため rvec=tvec=0
                    obj = X_th.reshape(1, 1, 3).astype(np.float64)
                    rvec = np.zeros((3, 1), dtype=np.float64)
                    tvec = np.zeros((3, 1), dtype=np.float64)

                    imgpts, _ = cv2.projectPoints(obj, rvec, tvec, K_th, dist_th)
                    u_th, v_th = imgpts.reshape(2)

                    # 描画
                    cv2.circle(rgb, (u, v), 6, (0, 255, 255), 2)
                    cv2.putText(rgb, f"z={z:.3f}m", (u + 10, v - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Thermal側（拡大表示に合わせてスケール）
                    uth_i = int(round(u_th))
                    vth_i = int(round(v_th))

                    # 画面内なら描画
                    W_th, H_th = 160, 120
                    uv_mode = 0  # ←まず左右反転を試す（0〜3が回転なしで簡単）

                    u_th_fix, v_th_fix = apply_th_uv_transform(u_th, v_th, W=W_th, H=H_th, mode=uv_mode)

                    uth_i = int(round(u_th_fix))
                    vth_i = int(round(v_th_fix))

                    if 0 <= uth_i < W_th and 0 <= vth_i < H_th:
                        cv2.circle(th_show, (uth_i * THERMAL_SCALE, vth_i * THERMAL_SCALE),
                                6, (0, 255, 255), 2)
                        cv2.putText(th_show, f"th=({u_th_fix:.1f},{v_th_fix:.1f})", (8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        cv2.putText(th_show, f"OUT th=({u_th_fix:.1f},{v_th_fix:.1f})", (8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 追加デバッグ表示（任意）
                    cv2.putText(rgb, f"X_rgb=[{X_rgb[0,0]:.3f},{X_rgb[1,0]:.3f},{X_rgb[2,0]:.3f}]", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgb, f"X_th =[{X_th[0,0]:.3f},{X_th[1,0]:.3f},{X_th[2,0]:.3f}]", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RGB", rgb)
            cv2.imshow("PI160", th_show)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                click.clear()
                print("[click] cleared")

    finally:
        pipeline.stop()
        cam_th.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
