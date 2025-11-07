import os, sys, cv2, numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from perception.sam2_wrapper import SAM2Engine
from utils.misc import to_rgb, draw_mask_on_image, filter_masks_by_area

ROOT = os.path.dirname(__file__)
IMG  = os.path.join(ROOT, "..", "data", "test_image.jpg")
CFG  = os.path.join(ROOT, "..", "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml")
CKPT = os.path.join(ROOT, "..","..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")

def main():
    if not os.path.exists(IMG):  raise SystemExit("test_image not found")
    if not os.path.exists(CFG) or not os.path.exists(CKPT):
        raise SystemExit("SAM2 config/ckpt not found.")

    bgr = cv2.imread(IMG); rgb = to_rgb(bgr); H,W = rgb.shape[:2]

    sam = SAM2Engine(CFG, CKPT, device="cuda", points_per_side=8, min_mask_region_area=500)
    sam.set_image(rgb)  # 画像セット（高コスト）

    masks = sam.generate_masks(rgb)
    masks = filter_masks_by_area(masks, H, W, min_area=1000, max_area_frac=0.5)
    if not masks: raise SystemExit("no masks")

    # 最大領域のbboxを取得して、predict_by_bboxで精密マスクへ
    m = max(masks, key=lambda x: x["area"])
    x,y,w,h = m["bbox"]
    bbox = [int(x), int(y), int(x+w), int(y+h)]
    refined = sam.predict_by_bbox(bbox)
    assert refined.shape[:2] == (H, W)

    vis = bgr.copy()
    cv2.rectangle(vis, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,180,0), 2)
    vis = draw_mask_on_image(vis, refined)
    cv2.imshow("bbox refine", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("OK: SAM2 bbox->mask works.")

if __name__ == "__main__":
    main()
