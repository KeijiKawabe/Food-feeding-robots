import cv2, numpy as np
from typing import List, Tuple

def to_rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def draw_mask_on_image(img_bgr, mask, color=(0,255,0), alpha=0.45):
    if mask is None: return img_bgr
    overlay = img_bgr.copy()
    m = mask > 0
    overlay[m] = (overlay[m]*(1-alpha) + np.array(color)*alpha).astype(np.uint8)
    return overlay

def crop_by_bbox(img, bbox):
    x1,y1,x2,y2 = map(int, bbox)
    h,w = img.shape[:2]
    x1=max(0,min(x1,w-1)); x2=max(0,min(x2,w-1))
    y1=max(0,min(y1,h-1)); y2=max(0,min(y2,h-1))
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2]

def filter_masks_by_area(masks, H, W, min_area=1000, max_area_frac=0.5):
    if not masks: return []
    max_area = int(max_area_frac*H*W)
    return [m for m in masks if min_area <= int(m.get("area",0)) <= max_area]

def masks_to_crops_and_bboxes(image_rgb, masks):
    crops, bboxes = [], []
    for m in masks:
        x,y,w,h = m["bbox"]
        x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)
        crop = crop_by_bbox(image_rgb, [x1,y1,x2,y2])
        if crop is not None and crop.size>0:
            crops.append(crop); bboxes.append([x1,y1,x2,y2])
    return crops, bboxes
