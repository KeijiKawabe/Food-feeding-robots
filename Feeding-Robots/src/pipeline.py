# src/pipeline.py

import time
import os
import cv2
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
    def process_frame_multi(self, image_bgr):
        """
        ROI限定 Segment Everything + CLIP により food candidate を検出。
        デバッグ用に SAM2 の全 mask / overlay / summary を保存する。
        """

        import os
        import time
        import numpy as np
        import cv2

        results = []

        # ============================
        # 0) ROI 設定
        # ============================
        ROI_X1, ROI_X2 = 120, 480
        ROI_Y1, ROI_Y2 = 200, 480

        roi_img = image_bgr[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        if roi_img.size == 0:
            print("⚠ ROI image is empty")
            return results

        # ============================
        # 1) デバッグフォルダ準備
        # ============================
        DEBUG_ROOT = "debug_sam2"
        os.makedirs(DEBUG_ROOT, exist_ok=True)

        frame_id = int(time.time() * 1000)
        frame_dir = os.path.join(DEBUG_ROOT, f"frame_{frame_id}")
        os.makedirs(frame_dir, exist_ok=True)

        # 保存：元画像 & ROI
        cv2.imwrite(os.path.join(frame_dir, "rgb.png"), image_bgr)
        cv2.imwrite(os.path.join(frame_dir, "roi.png"), roi_img)

        # 可視化用（元画像）
        vis_img = image_bgr.copy()

        # ROI 枠を描画（参考用）
        cv2.rectangle(
            vis_img,
            (ROI_X1, ROI_Y1),
            (ROI_X2, ROI_Y2),
            (255, 0, 0),
            2
        )

        # ============================
        # 2) SAM2: ROI で Segment Everything
        # ============================
        masks = self.sam.generate_masks(roi_img)

        if masks is None or len(masks) == 0:
            print("⚠ SAM2 returned no masks (ROI)")
            return results

        print(f"[SAM2] ROI mask count: {len(masks)}")

        # ============================
        # 3) 各 mask 処理
        # ============================
        for idx, m in enumerate(masks):

            # ---------- mask 取得 ----------
            mask = m.get("segmentation", None)
            if mask is None:
                print(f"[SKIP] idx={idx} mask is None keys={list(m.keys())}")
                continue

            # ---------- mask 保存（ROI座標） ----------
            mask_u8 = (mask.astype(np.uint8) * 255)
            cv2.imwrite(
                os.path.join(frame_dir, f"mask_{idx:02d}.png"),
                mask_u8
            )

            overlay = roi_img.copy()
            overlay[mask] = [0, 0, 255]
            cv2.imwrite(
                os.path.join(frame_dir, f"mask_{idx:02d}_overlay.png"),
                overlay
            )

            # ---------- bbox（ROI座標） ----------
            # ---------- bboxをsegmentationから再計算（ROI座標） ----------
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                print(f"[SKIP] idx={idx} empty segmentation")
                continue

            x0 = int(xs.min())
            x1 = int(xs.max()) + 1   # +1 重要（スライスで幅0を防ぐ）
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1

            # 念のためROI範囲にクリップ
            h, w = roi_img.shape[:2]
            x0 = max(0, min(x0, w-1))
            x1 = max(0, min(x1, w))
            y0 = max(0, min(y0, h-1))
            y1 = max(0, min(y1, h))

            # ここで初めてcrop
            crop = roi_img[y0:y1, x0:x1]
            if crop.size == 0:
                print(f"[SKIP] idx={idx} empty crop after recompute bbox: {(x0,y0,x1,y1)}")
                continue


            # ---------- CLIP ----------
            score_dict = self.clip.score_single(crop)
            if score_dict is None:
                print(f"[SKIP] idx={idx} CLIP returned None")
                continue

            best_label = max(score_dict, key=score_dict.get)
            best_score = score_dict[best_label]

            # ---------- 座標を元画像に戻す ----------
            gx0 = x0 + ROI_X1
            gy0 = y0 + ROI_Y1
            gx1 = x1 + ROI_X1
            gy1 = y1 + ROI_Y1

            cx = int((gx0 + gx1) / 2)
            cy = int((gy0 + gy1) / 2)

            # ---------- summary 可視化 ----------
            cv2.rectangle(vis_img, (gx0, gy0), (gx1, gy1), (0, 255, 0), 2)
            cv2.circle(vis_img, (cx, cy), 4, (0, 0, 255), -1)

            text = f"{best_label} ({best_score:.2f})"
            cv2.putText(
                vis_img,
                text,
                (gx0, max(gy0 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            results.append({
                "label": best_label,
                "score": float(best_score),
                "clip_scores": score_dict,
                "bbox": (gx0, gy0, gx1, gy1),
                "center_px": (cx, cy),
                "crop": crop
            })

        # ============================
        # 4) summary 保存
        # ============================
        cv2.imwrite(os.path.join(frame_dir, "summary.png"), vis_img)
        print(f"🖼 Saved SAM2 ROI debug images to: {os.path.abspath(frame_dir)}")

        # ============================
        # 5) console log
        # ============================
        print("\n=== Multi-food detection result (ROI) ===")
        for r in results:
            print(
                f"Label={r['label']} "
                f"score={r['score']:.2f} "
                f"center={r['center_px']}"
            )

        return results


    # def process_frame_multi(self, image_bgr):
    #     """
    #     SAM2 + CLIP により、画像中の全 food candidate を返す関数。

    #     出力例:
    #     [
    #         {"label": "Yogurt", "bbox": [...], "center_px": (cx, cy), "score": 22.4},
    #         {"label": "curry",  "bbox": [...], "center_px": (cx, cy), "score": 21.0},
    #         ...
    #     ]
    #     """

    #     results = []
    #     vis_img = image_bgr.copy()

    #     # ---------- 1) SAM2 mask generation ----------
    #     masks = self.sam.generate_masks(image_bgr)


    #     # masks は list[dict] 形式を想定（mask, bbox, score など）

    #     if masks is None or len(masks) == 0:
    #         print("⚠ SAM2 returned no masks")
    #         return results

    #     # ---------- 2) Each mask → CLIP scoring ----------
    #     for idx, m in enumerate(masks):
    #         bbox = m["bbox"]  # (x0, y0, x1, y1)
    #         x0, y0, x1, y1 = map(int, bbox)
    #         crop = image_bgr[y0:y1, x0:x1]

    #         if crop.size == 0:
    #             continue

    #         # CLIP scoring (returns dict: {"Yogurt": score, "curry": score, ...})
    #         score_dict = self.clip.score_single(crop)
    #         if score_dict is None:
    #             continue

    #         # best label
    #         best_label = max(score_dict, key=score_dict.get)
    #         best_score = score_dict[best_label]

    #         # center of bbox
    #         cx = int((x0 + x1)/2)
    #         cy = int((y0 + y1)/2)
    #                 # ===== 可視化 =====
    #         color = (0, 255, 0)  # bbox: green
    #         cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
    #         cv2.circle(vis_img, (cx, cy), 4, (0, 0, 255), -1)
    #         text = f"{best_label} ({best_score:.2f})"
    #         cv2.putText(
    #             vis_img,
    #             text,
    #             (x0, max(y0 - 5, 15)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.45,
    #             (255, 255, 255),
    #             2,
    #             cv2.LINE_AA
    #         )

    #         results.append({
    #             "label": best_label,
    #             "score": float(best_score),
    #             "clip scores": score_dict,
    #             "bbox": bbox,
    #             "center_px": (cx, cy),
    #         })
    #         ts = int(time.time() * 1000)
    #         out_path = f"debug_multi_food_{ts}.png"
    #         cv2.imwrite(out_path, vis_img)
    #         print(f"🖼 Saved detection visualization: {out_path}")

    #     # ---------- 3) DEBUG: show recognized instances ----------
    #     print("\n=== Multi-food detection result ===")
    #     for r in results:
    #         print(f"Label={r['label']} score={r['score']:.2f} center={r['center_px']}")

    #     return results

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        target_label: Optional[str] = None,
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
            print(f"DEBUG: SAM2 generated {len(masks)} raw masks.") # マスク候補の数

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
            print(f"DEBUG: After area filter, {len(masks)} masks remain.")
            # =============================
            # DEBUG: Save all crops & CLIP scores
            # =============================
            debug_dir = "debug_crops"
            os.makedirs(debug_dir, exist_ok=True)

            print("\n--- DEBUG: CLIP crop & score list ---")

            for i, crop in enumerate(crops):
                # crop 保存
                crop_path = os.path.join(debug_dir, f"crop_{i:02d}.png")
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                # 全ラベルに対するスコアを計算
                score_dict = self.clip.score_single(crop)

                print(f"[Crop {i:02d}] {crop_path}")
                for label, score in score_dict.items():
                    print(f"    {label:10s} : {score:.4f}")


            if crops:
                if target_label is None:
                    # 従来の方式：CLIPスコア最大採用
                    pick = self.clip.pick_best(crops)
                else:
                    # ★ 新方式：LLM の next_food でフィルタした中で最大スコア
                    pick = self.clip.pick_target(crops, target_label)
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
    



