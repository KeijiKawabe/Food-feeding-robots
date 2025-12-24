import cv2
import numpy as np
import pyrealsense2 as rs

# ========= 設定 =========
# ここはあなたのボードに合わせて増やしてOK（(cols, rows)）
PATTERN_CANDIDATES = [
    (11, 4), (4, 11),
    (11, 5), (5, 11),
    (9, 6),  (6, 9),
]
# =======================


def make_blob_detector(blob_color=255, min_area=80, max_area=30000, min_dist=12,
                       min_circularity=0.5, min_inertia=0.1):
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 5
    p.maxThreshold = 250
    p.thresholdStep = 10

    p.filterByColor = True
    p.blobColor = int(blob_color)  # 255: bright circles, 0: dark circles

    p.filterByArea = True
    p.minArea = float(min_area)
    p.maxArea = float(max_area)

    p.filterByCircularity = True
    p.minCircularity = float(min_circularity)

    p.filterByInertia = True
    p.minInertiaRatio = float(min_inertia)

    p.filterByConvexity = False
    p.minDistBetweenBlobs = float(min_dist)

    return cv2.SimpleBlobDetector_create(p)


def preprocess(gray, mode="adaptive", invert=False):
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    if mode == "adaptive":
        th = cv2.adaptiveThreshold(
            g, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )
        out = th
    elif mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(g)
    else:
        out = g

    if invert:
        out = 255 - out
    return out


def debug_blobs(img8, blob):
    kps = blob.detect(img8)
    vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(vis, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return len(kps), vis


def try_find(img8, blob):
    results = []
    flag_sets = [
        cv2.CALIB_CB_ASYMMETRIC_GRID,
        cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
        cv2.CALIB_CB_SYMMETRIC_GRID,
        cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
    ]
    for psz in PATTERN_CANDIDATES:
        for flags in flag_sets:
            ok, centers = cv2.findCirclesGrid(img8, psz, flags=flags, blobDetector=blob)
            if ok:
                results.append((True, psz, flags, centers))
                return results  # 最初に見つかったやつを返す（十分）
    return results


def main():
    # RealSense start
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # warmup
    for _ in range(15):
        pipeline.wait_for_frames()

    # パラメータ（キーでいじれる）
    invert = False
    prep_mode = "adaptive"   # "adaptive" or "clahe" or "raw"
    blob_color = 255         # 255=白い円, 0=黒い円
    min_area = 80
    max_area = 30000
    min_dist = 12
    min_circ = 0.5
    min_inertia = 0.1

    print("=== RealSense CircleGrid Test ===")
    print("keys:")
    print("  i : invert")
    print("  p : preprocess mode (adaptive -> clahe -> raw)")
    print("  b : blobColor toggle (255 <-> 0)")
    print("  [ / ] : minArea -/+")
    print("  - / = : minDistBetweenBlobs -/+")
    print("  q / ESC : quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            bgr = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            img8 = preprocess(gray, mode=prep_mode, invert=invert)

            blob = make_blob_detector(
                blob_color=blob_color,
                min_area=min_area,
                max_area=max_area,
                min_dist=min_dist,
                min_circularity=min_circ,
                min_inertia=min_inertia,
            )

            n_kp, blob_vis = debug_blobs(img8, blob)
            found = try_find(img8, blob)

            vis = bgr.copy()
            status = f"kp={n_kp} invert={invert} prep={prep_mode} blobColor={blob_color} minArea={min_area} minDist={min_dist}"
            cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if found:
                ok, psz, flags, centers = found[0]
                cv2.drawChessboardCorners(vis, psz, centers, True)
                cv2.putText(vis, f"FOUND {psz} flags={flags}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("rgb", vis)
            cv2.imshow("preprocess", img8)
            cv2.imshow("blob_debug", blob_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('i'):
                invert = not invert
            if key == ord('p'):
                prep_mode = "clahe" if prep_mode == "adaptive" else ("raw" if prep_mode == "clahe" else "adaptive")
            if key == ord('b'):
                blob_color = 0 if blob_color == 255 else 255
            if key == ord('['):
                min_area = max(5, min_area - 10)
            if key == ord(']'):
                min_area = min_area + 10
            if key == ord('-'):
                min_dist = max(1, min_dist - 1)
            if key == ord('='):
                min_dist = min_dist + 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
