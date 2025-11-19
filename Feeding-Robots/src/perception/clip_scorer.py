# perception/clip_scorer.py
import numpy as np
import torch
import clip
from PIL import Image
from typing import Dict, List, Optional

class ClipScorer:
    def __init__(self, device="cuda", model_name="ViT-B/32",
                 prompts: Optional[Dict[str, List[str]]] = None,
                 use_fp16: bool = True):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # ✨ 重要：モデル全体は float32 のまま（.half() しない）
        self.use_image_amp = (self.device == "cuda" and use_fp16)

        self.prompts = prompts or {
            "rice": ["a photo of rice"],
            "non_food": ["empty plate"]
        }
        self.text_feat = self._encode_prompts(self.prompts)

    def _encode_prompts(self, prompts: Dict[str, List[str]]):
        text_feat = {}
        with torch.no_grad():
            # ✨ テキスト側は常に FP32。autocast も無効化。
            autocast_ctx = (torch.cuda.amp.autocast(enabled=False)
                            if self.device == "cuda" else torch.no_grad())
            with autocast_ctx:
                for cls, phrases in prompts.items():
                    toks = clip.tokenize(phrases).to(self.device)
                    feats = self.model.encode_text(toks)          # FP32
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    text_feat[cls] = feats.mean(dim=0, keepdim=True)  # [1, D], FP32
        return text_feat

    def update_prompts(self, prompts: Dict[str, List[str]]):
        self.prompts = prompts
        self.text_feat = self._encode_prompts(prompts)

    @torch.no_grad()
    def score_crops(self, crops_rgb: List[np.ndarray]) -> Dict[str, np.ndarray]:
        if not crops_rgb:
            return {k: np.zeros((0,), dtype=np.float32) for k in self.prompts.keys()}

        imgs = [self.preprocess(Image.fromarray(im)).to(self.device)
                for im in crops_rgb]
        batch = torch.stack(imgs)  # [N,3,H,W], FP32テンソル

        # ✨ 画像側だけ AMP を使う（あくまで入力と一部演算のみ FP16 に）
        if self.use_image_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                img_feat = self.model.encode_image(batch)
        else:
            img_feat = self.model.encode_image(batch)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # ここでFP32に戻ることも多い
        scores = {}
        for cls, tfeat in self.text_feat.items():  # tfeat は FP32
            sim = (img_feat @ tfeat.T) * 100.0
            scores[cls] = sim.squeeze(1).float().cpu().numpy()
        return scores

    def pick_best(self, crops_rgb: List[np.ndarray], thresholds: Optional[Dict[str, float]] = None):
        thresholds = thresholds or {"rice": 23.0}
        scores = self.score_crops(crops_rgb)
        best = None
        for cls, arr in scores.items():
            if arr.size == 0: continue
            j = int(np.argmax(arr)); s = float(arr[j])
            if s >= thresholds.get(cls, 1e9):
                if best is None or s > best["score"]:
                    best = {"cls": cls, "index": j, "score": s}
        return best
