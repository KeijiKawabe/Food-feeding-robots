import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))

import clip, torch
print("[CLIP] ✅", clip.available_models())

import depth_anything_v2
print("[DepthAnything] ✅ Loaded")

from sam2.sam2_image_predictor import SAM2ImagePredictor
print("[SAM2] ✅ Ready")

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
