# test_scripts/test_pipeline_on_image.py

import os
import sys
import cv2

# make `src` importable so `src.pipeline` and `src.thermal` resolve
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.pipeline import PerceptionPipeline


def main():
    print("\n=== Testing Perception Pipeline with Depth ===")

    SAM2_CFG = "../../sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    SAM2_CKPT = "../../sam2/checkpoints/sam2.1_hiera_base_plus.pt"

    DEPTH_CKPT = "../Depth-Anything-v2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth"

    # Get OpenAI API key from environment if available (used by thermal GPT analysis)
    api_key = os.getenv("OPENAI_API_KEY")

    pipe = PerceptionPipeline(
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        enable_depth=True,
        depth_ckpt=DEPTH_CKPT,
        device="cuda",
        enable_thermal=True,
        thermal_api_key=api_key,
        thermal_save_image=False
    )

    img_path = "test_image.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ cannot load image: {img_path}")
        return

    out = pipe.process_frame(img)

    print("\n=== Pipeline Output ===")
    for k, v in out.items():
        print(f"{k}: {v}")

    # 可視化
    if out["bbox"] is not None:
        x1, y1, x2, y2 = out["bbox"]
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Depth-Aware Pipeline", vis)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    # --- Thermal test (one-shot) ---
    print("\n=== Thermal test (one-shot) ===")
    if getattr(pipe, 'thermal', None) is None:
        print("Thermal system not initialized or not available")
        return

    # Run the pipeline's thermal analysis routine (captures and runs GPT analysis)
    therm_res = pipe.run_thermal_analysis()
    print("Thermal result:", therm_res)

    # Also capture and display the palette image if available
    try:
        tdata, timg = pipe.thermal.capture_thermal_image()
        if timg is not None:
            cv2.imshow('Thermal Palette', timg)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Failed to capture/display thermal image: {e}")


if __name__ == "__main__":
    main()
