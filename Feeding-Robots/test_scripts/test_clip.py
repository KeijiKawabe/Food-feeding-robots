import os, sys, cv2, numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from perception.clip_scorer import ClipScorer
from utils.misc import to_rgb

IMG = os.path.join(os.path.dirname(__file__), "..", "data", "test_image.jpg")

def main():
    if not os.path.exists(IMG):
        raise SystemExit(f"test image not found: {IMG}")

    bgr = cv2.imread(IMG)
    rgb = to_rgb(bgr)

    # シンプルに "画像全体" を1つのcropとしてCLIPへ
    crops = [rgb]
    scorer = ClipScorer(device="cuda", model_name="ViT-B/32", prompts={
        "food": ["a photo of cooked food", "mashed baby food on a spoon"],
        "non_food": ["empty plate", "wooden tabletop without food"]
    })
    use_fp16 = False

    scores = scorer.score_crops(crops)
    print("[CLIP] keys:", list(scores.keys()))
    for k, v in scores.items():
        print(f"[CLIP] {k} scores:", v)

    # pick_best（しきい値ゆるめ）
    best = scorer.pick_best(crops, thresholds={"food": 0.0})
    print("[CLIP] best:", best)
    assert "food" in scores and len(scores["food"]) == 1, "score shape mismatch"
    print("OK: CLIP scorer works.")

if __name__ == "__main__":
    main()
