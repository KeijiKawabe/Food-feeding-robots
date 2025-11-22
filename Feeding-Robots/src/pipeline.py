# src/pipeline.py

import time
from typing import Dict, Any, Optional

import numpy as np

from .utils.misc import (
    to_rgb,
    filter_masks_by_area,
    masks_to_crops_and_bboxes,
)
from .perception.sam2_wrapper import SAM2Engine
from .perception.clip_scorer import ClipScorer


class PerceptionPipeline:
    """
    RGB + SAM2 + CLIP (+ 任意で Depth) をまとめた認識パイプライン。

    想定する入力:
        - frame_bgr: OpenCV形式の BGR 画像 (H, W, 3), dtype=uint8
                     RealSense の color フレームから取得したものを想定
        - depth_frame: 任意。frame_bgr と同じ解像度の深度画像 (H, W)
                       RealSense の aligned depth (color に揃えたもの) を想定
                       単位は mm (標準的な RealSense の出力) を想定

    出力:
        dict で以下のキーを返す:
            - mask:      選択された食材領域の2値マスク (H, W) or None
            - bbox:      [x1, y1, x2, y2] 形式のバウンディングボックス or None
            - label:     CLIPで選ばれたクラス名 (str) or None
            - score:     CLIPスコア (float) or None
            - center_px: (cx, cy) 画素座標 (bbox の中心) or None
            - depth_m:   center_px における深度 [m] (float) or None
            - fps:       このフレームの処理の指数移動平均 FPS (float)

    備考:
        - DepthAnything は使わず、深度は RealSense などの実センサから渡す前提。
        - 「何を食べさせるか」の意思決定 (Task Planning) は別モジュールに任せる。
          ここでは純粋に「どの食材がどこにあり、どのくらいの距離にあるか」
          までを推定する。
    """

    def __init__(
        self,
        sam2_cfg: str,
        sam2_ckpt: str,
        device: str = "cuda",
        maskgen_interval: int = 10,
        min_area: int = 1000,
        max_area_frac: float = 0.5,
        clip_model: str = "ViT-B/32",
        clip_prompts=None,
        enable_depth: bool = True,
    ) -> None:
        """
        Args:
            sam2_cfg:   SAM2 の config YAML へのパス
            sam2_ckpt:  SAM2 の checkpoint (.pt) へのパス
            device:     "cuda" or "cpu"
            maskgen_interval:
                何フレームごとに SAM2 のマスク生成を行うか。
                単発テストなら 1, 動画なら 5〜10 などにして負荷を下げる。
            min_area:
                小さすぎるマスクを捨てるための画素数のしきい値。
            max_area_frac:
                画像全体に対するマスク面積の最大割合。
                (例) 0.5 なら画像の 50% を超える巨大なマスクは除外。
            clip_model:
                使用する CLIP モデル名。
            clip_prompts:
                ClipScorer に渡すプロンプト辞書。
                例: {"rice": ["rice", "boiled rice"], "curry": ["curry", "stew"], ...}
                None の場合は ClipScorer 側のデフォルトに任せる。
            enable_depth:
                True の場合、depth_frame が渡されていれば中心深度 depth_m を計算する。
        """
        # --- SAM2 エンジン ---
        self.sam = SAM2Engine(
            sam2_cfg,
            sam2_ckpt,
            device=device,
            points_per_side=8,
            min_mask_region_area=min_area,
        )

        # --- CLIP スコアラー ---
        self.clip = ClipScorer(
            device=device,
            model_name=clip_model,
            prompts=clip_prompts,
        )

        self.maskgen_interval = maskgen_interval
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.enable_depth = enable_depth

        self.frame_count = 0
        self.ema_fps: Optional[float] = None

        # 前回の結果をキャッシュ（maskgen_interval > 1 のときに利用）
        self.last: Dict[str, Any] = {
            "mask": None,
            "bbox": None,
            "label": None,
            "score": None,
            "center_px": None,
            "depth_m": None,
        }

    # --- CLIP プロンプトを途中で変えたい場合用 ---
    def update_clip_prompts(self, prompts) -> None:
        """
        ClipScorer 側のプロンプトを更新したい場合に使用。
        prompts のフォーマットは ClipScorer の実装に従う。
        例: {"rice": ["rice", "boiled rice"], "curry": ["curry"], ...}
        """
        self.clip.update_prompts(prompts)

    # --------------------------------------------------
    # メイン処理
    # --------------------------------------------------
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        1フレーム分の RGB (+ 任意で Depth) を入力として処理し、
        食材マスク / BBox / CLIP ラベル / 深度などを返す。

        Args:
            frame_bgr:  BGR 画像 (H, W, 3), dtype=uint8
            depth_frame:
                RealSense などの深度画像 (H, W), 単位 mm を想定。
                color とすでにアラインしてある前提。
                None の場合は depth_m, center_px は None のまま。

        Returns:
            Dict[str, Any]:
                "mask", "bbox", "label", "score",
                "center_px", "depth_m", "fps" を含む辞書。
        """
        t0 = time.time()

        # BGR → RGB
        rgb = to_rgb(frame_bgr)
        H, W = rgb.shape[:2]

        # マスク生成が必要かどうか判定
        need_new_masks = (
            self.frame_count % self.maskgen_interval == 0
            or self.last["mask"] is None
        )

        if need_new_masks:
            # --- SAM2 で「全マスク」を生成 ---
            self.sam.set_image(rgb)  # ここが高コスト
            masks = self.sam.generate_masks(rgb)

            # --- 小さすぎる/大きすぎるマスクをフィルタ ---
            masks = filter_masks_by_area(
                masks,
                H,
                W,
                self.min_area,
                self.max_area_frac,
            )

            # --- マスクごとに crop & bbox を作成 ---
            crops, bboxes = masks_to_crops_and_bboxes(rgb, masks)

            if crops:
                # --- CLIP で最もそれっぽい食材マスクを1つ選ぶ ---
                # 閾値は ClipScorer 側の実装に任せる or 必要ならここで dict を渡す
                pick = self.clip.pick_best(crops, thresholds=None)

                if pick is not None:
                    idx = pick["index"]
                    cls = pick["cls"]
                    score = float(pick["score"])
                    bbox = bboxes[idx]

                    # bbox で SAM2 のマスクを再度 refine してもいいし、
                    # 既存の masks[idx] をそのまま使っても良い。
                    # ここでは refine して精度を上げる。
                    refined_mask = self.sam.predict_by_bbox(bbox)

                    # bbox 中心ピクセル
                    x1, y1, x2, y2 = bbox
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    center_px = (cx, cy)

                    # 深度が使える場合は中心の depth を m で取得
                    depth_m = None
                    if self.enable_depth and depth_frame is not None:
                        if (
                            depth_frame.shape[0] == H
                            and depth_frame.shape[1] == W
                        ):
                            raw_depth = float(depth_frame[cy, cx])  # mm 想定
                            if raw_depth > 0:
                                depth_m = raw_depth / 1000.0  # m に変換
                        # 解像度が合っていない場合は depth_m は None のまま

                    self.last.update(
                        mask=refined_mask,
                        bbox=bbox,
                        label=cls,
                        score=score,
                        center_px=center_px,
                        depth_m=depth_m,
                    )
                else:
                    # CLIP で有力な候補が見つからなかった場合
                    self.last.update(
                        mask=None,
                        bbox=None,
                        label=None,
                        score=None,
                        center_px=None,
                        depth_m=None,
                    )
            else:
                # 有効マスクが1つもなかった場合
                self.last.update(
                    mask=None,
                    bbox=None,
                    label=None,
                    score=None,
                    center_px=None,
                    depth_m=None,
                )

        # FPS の計算（指数移動平均）
        self.frame_count += 1
        dt = max(time.time() - t0, 1e-6)
        fps = 1.0 / dt
        if self.ema_fps is None:
            self.ema_fps = fps
        else:
            self.ema_fps = 0.9 * self.ema_fps + 0.1 * fps

        # 出力をまとめて返す
        return {
            "mask": self.last["mask"],
            "bbox": self.last["bbox"],
            "label": self.last["label"],
            "score": self.last["score"],
            "center_px": self.last["center_px"],
            "depth_m": self.last["depth_m"],
            "fps": self.ema_fps,
        }
