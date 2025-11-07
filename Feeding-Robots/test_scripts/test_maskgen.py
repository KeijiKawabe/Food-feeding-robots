import os, sys, cv2
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from perception.sam2_wrapper import SAM2Engine
from utils.misc import to_rgb, draw_mask_on_image, filter_masks_by_area
ROOT = os.path.dirname(__file__)


IMG  = os.path.join(ROOT, "..", "data", "test_image.jpg")
CFG  = os.path.join(ROOT, "..", "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml")
CKPT = os.path.join(ROOT, "..", "..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")

def main():
    if not os.path.exists(IMG):  raise SystemExit("test_image not found")
    if not os.path.exists(CFG) or not os.path.exists(CKPT):
        raise SystemExit("SAM2 config/ckpt not found. Please set paths.")

    bgr = cv2.imread(IMG); rgb = to_rgb(bgr)
    H, W = rgb.shape[:2]

    sam = SAM2Engine(CFG, CKPT, device="cuda", points_per_side=8, min_mask_region_area=500)
    masks = sam.generate_masks(rgb)
    print(f"[SAM2] raw masks: {len(masks)}")

    masks = filter_masks_by_area(masks, H, W, min_area=1000, max_area_frac=0.5)
    print(f"[SAM2] filtered masks: {len(masks)}")

    if not masks:
        raise SystemExit("No masks after filtering. Try relaxing thresholds.")

    # 可視化（最大領域だけ塗る）
    biggest = max(masks, key=lambda m: m["area"])
    vis = draw_mask_on_image(bgr.copy(), biggest["segmentation"].astype("uint8"))
    cv2.imshow("maskgen", vis)
    cv2.waitKey(10000)  # 1秒だけ表示
    cv2.destroyAllWindows()
    print("OK: SAM2 mask generation works.")

if __name__ == "__main__":
    main()
