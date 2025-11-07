# feeding_robot/src/match_clip.py
import torch
import clip
from PIL import Image
import numpy as np
import cv2

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _preprocess = clip.load("ViT-B/32", device=_device)

def score_masks_with_clip(bgr_img, masks, text_prompt):
    """SAM2マスクごとにCLIPでスコアを計算"""
    text_tokens = clip.tokenize([text_prompt]).to(_device)
    text_features = _model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    scores = []
    for m in masks:
        crop = _crop_with_mask(bgr_img, m["mask"])
        if crop is None:
            scores.append(0.0)
            continue
        img_tensor = _preprocess(Image.fromarray(crop)).unsqueeze(0).to(_device)
        img_features = _model.encode_image(img_tensor)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        sim = (img_features @ text_features.T).item()
        scores.append(sim)
    return scores

def _crop_with_mask(bgr, mask):
    ys, xs = mask.nonzero()
    if len(xs) == 0: return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    crop = bgr[y1:y2, x1:x2]
    # 背景を黒にする（ノイズ除去）
    m = mask[y1:y2, x1:x2][:, :, None]
    crop = crop * m + (1 - m) * 0
    return cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2RGB)
