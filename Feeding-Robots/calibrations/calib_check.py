import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time

# =======================================
# 1. 手元の Hand–Eye 結果 T_base_camera を貼る
# =======================================
T_base_camera = np.array([
[-0.56870572, -0.81759112, -0.09010307,  0.85268819],
 [ 0.50069114, -0.25718523, -0.82653744,  0.40449154],
 [ 0.6525965,  -0.51517038,  0.55562334,  0.07127631],
 [ 0.,          0.,          0.,          1.        ],
])#出力をコピペする

# =======================================
# 2. TCP → Marker オフセット行列（28cm, X軸方向）
# =======================================
T_gm = np.eye(4)
T_gm[0, 3] = -0.28   # TCP の X 軸方向に +0.28 m


# =======================================
# Utility
# =======================================
def pose_to_matrix(p):
    """xArm の TCP pose（mm & rad）→ 4×4 行列(base→gripper)"""
    x, y, z, rx, ry, rz = p
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T


# =======================================
# 3. RealSense 初期設定
# =======================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

camera_matrix = np.array([
    [389.846, 0,      321.177],
    [0,       389.846, 235.201],
    [0,       0,        1     ]
])
dist_coeffs = np.zeros(5)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()


# =======================================
# 4. xArm 初期設定
# =======================================
arm = XArmAPI("192.168.1.199")
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)
time.sleep(1)


# =======================================
# 5. Verification Loop
# =======================================
while True:

    # ---- Robot side: base→marker ----
    p = arm.get_position(is_radian=True)[1]   # [x,y,z,rx,ry,rz]
    T_bg = pose_to_matrix(p)                  # base→gripper
    T_bm_robot = T_bg @ T_gm                  # base→marker（ロボットでの推定）

    # ---- Camera side: base→marker ----
    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()
    img = np.asanyarray(frame.get_data())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        print("ArUco missing")
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    # marker→camera
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.035, camera_matrix, dist_coeffs)
    R_mc, _ = cv2.Rodrigues(rvec[0])
    T_mc = np.eye(4)
    T_mc[:3, :3] = R_mc
    T_mc[:3, 3] = tvec[0].reshape(3)

    # base→marker（カメラから）
    T_bm_camera = T_base_camera @ T_mc

    # ---- Difference ----
    diff = T_bm_camera[:3, 3] - T_bm_robot[:3, 3]
    error = np.linalg.norm(diff)

    print("\n===== Verification =====")
    print("marker_from_camera:", T_bm_camera[:3, 3])
    print("marker_from_robot :", T_bm_robot[:3, 3])
    print("difference (m):    ", diff)
    print("error norm (m):    ", error)

    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break
