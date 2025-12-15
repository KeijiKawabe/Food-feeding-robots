import pyrealsense2 as rs

# パイプラインの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ストリーミング開始
profile = pipeline.start(config)

# RGBカメラのストリームプロファイルを取得
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# 値の確認（これをキャリブレーションに使います）
print("=== RealSense Internal Intrinsics ===")
print(f"Width: {intrinsics.width}")
print(f"Height: {intrinsics.height}")
print(f"fx: {intrinsics.fx}")
print(f"fy: {intrinsics.fy}")
print(f"cx: {intrinsics.ppx}") # ppxがcxのこと
print(f"cy: {intrinsics.ppy}") # ppyがcyのこと
print(f"Distortion Model: {intrinsics.model}")
print(f"Distortion Coeffs: {intrinsics.coeffs}") # これが歪み係数

pipeline.stop()