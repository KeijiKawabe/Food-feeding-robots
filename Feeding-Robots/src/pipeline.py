# src/pipeline.py

import time
import os
import cv2
from typing import Dict, Any, Optional
import sys
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

        # __init__ の最後あたりに追加
        self.last_instances: Dict[str, Any] = {}   # label -> best instance
        self.last_candidates: list = []            # デバッグ用（任意）


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
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            score_dict = self.clip.score_single(crop_rgb)
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
                "crop": crop_rgb
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

 
    def cripping(self, rgb, H, W):
        y1, y2 = int(H * 0.5), int(H * 1.0)
        x1, x2 = int(W * 0.3), int(W * 0.9)
        
        cropped_rgb = rgb[y1:y2, x1:x2]
        
        # 可視化
        args = sys.argv
        if len(args) > 1:
            if self.frame_count % self.maskgen_interval == 0:
                cv2.imshow("debug_roi_input", cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        # 1. クロップ画像, 2. yの開始位置, 3. xの開始位置 を返す
        return cropped_rgb, y1, x1

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
        rgb_before_clop  = to_rgb(frame_bgr)
        H_before_crop, W_before_crop = rgb_before_clop.shape[:2]

        # --- 1. クロップとオフセット取得 ---
        # rgb: クロップ後画像, offset_y/x: 元画像での開始座標
        rgb, offset_y, offset_x = self.cripping(rgb_before_clop, H_before_crop, W_before_crop)
        cH, cW = rgb.shape[:2]  # クロップ後の高さ・幅

        # マスク生成判定
        need_new_masks = (
            self.frame_count % self.maskgen_interval == 0
            or self.last["mask"] is None
        )

        if need_new_masks:
            self.sam.set_image(rgb)
            masks = self.sam.generate_masks(rgb)

            # --- ここで「クロップ世界のマスク」を「元の世界のサイズ」に変換する ---
            for m in masks:
                # 1. BBoxの座標をオフセット分ずらす
                m['bbox'][0] += offset_x
                m['bbox'][1] += offset_y

                # 2. マスク(240, W) を元の (480, W) に戻す
                # 元の画像と同じサイズの「すべてFalse（黒）」の配列を作る
                full_mask = np.zeros((H_before_crop, W_before_crop), dtype=bool)
                # 指定した位置（クロップした範囲）にだけ、SAMの結果を貼り付ける
                full_mask[offset_y : offset_y + rgb.shape[0], 
                          offset_x : offset_x + rgb.shape[1]] = m['segmentation']
                
                # 上書きする
                m['segmentation'] = full_mask

            # この後、フィルタリングやCLIP処理を続行すれば、
            # 全ての座標とサイズが H=480 の世界で統一されます。

            # --- 4. フィルタリング (ここでの H, W はオリジナルサイズを渡す) ---
            masks = filter_masks_by_area(
                masks,
                H_before_crop, 
                W_before_crop,
                self.min_area,
                self.max_area_frac,
            )

            # --- 5. CLIP用の Crop作成 ---
            # ここでは「元画像(rgb_before_clop)」から「復元後のBBox」を使って切り抜く
            crops, bboxes = masks_to_crops_and_bboxes(rgb_before_clop, masks)
            print(f"DEBUG: After area filter, {len(masks)} masks remain.")
            # =============================
            # DEBUG: Save all crops & CLIP scores
            # =============================
            debug_dir = "debug_crops"
            os.makedirs(debug_dir, exist_ok=True)

            print("\n--- DEBUG: CLIP crop & score list ---")

                # =============================
            # CLIP: labelごとに best crop を選ぶ
            # =============================


            best_per_label = {}
            for i, crop in enumerate(crops):
                score_dict = self.clip.score_single(crop)
                if score_dict is None:
                    continue
                
                for label, score in score_dict.items():
                    if (label not in best_per_label or score > best_per_label[label]["score"]):
                        bbox = bboxes[i]

                        # 中心座標の計算
                        x1, y1, x2, y2 = bbox
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # 距離の取得
                        depth_m = None
                        if self.enable_depth and depth_frame is not None:
                            d = float(depth_frame[cy, cx])
                            if d > 0:
                                depth_m = d / 1000.0

                        # --- 【修正】SAMによる再予測とサイズ復元 ---
                        # 1. SAM用のローカル座標BBox
                        bbox_for_sam = [
                            bbox[0] - offset_x, 
                            bbox[1] - offset_y, 
                            bbox[2] - offset_x, 
                            bbox[3] - offset_y
                        ]
                        
                        # 2. クロップ画像に対して再予測
                        refined_mask_small = self.sam.predict_by_bbox(bbox_for_sam)

                        # 3. 元のサイズ (480x640) に戻すためのパディング処理
                        refined_full_mask = np.zeros((H_before_crop, W_before_crop), dtype=bool)
                        refined_full_mask[offset_y : offset_y + cH, 
                                          offset_x : offset_x + cW] = refined_mask_small
                        
                        # 4. 辞書に格納 (修正したフルサイズマスクを入れる)
                        best_per_label[label] = {
                            "mask": refined_full_mask, 
                            "bbox": bbox,
                            "center_px": (cx, cy),
                            "score": float(score),
                            "depth_m": depth_m,
                        }


        self.frame_count += 1
        dt = max(time.time() - t0, 1e-6)
        fps = 1.0 / dt
        self.ema_fps = fps if self.ema_fps is None else 0.9*self.ema_fps + 0.1*fps


        # === 可視化コードの追加 (return の直前に挿入) ===

       
        args = sys.argv
        if len(args) > 1:
            # ループの前に、全候補を格納するリストを用意
            all_candidates = []

            for i, crop in enumerate(crops):
                score_dict = self.clip.score_single(crop)
                if score_dict is None:
                    continue
                
                # そのクロップで最も高いスコアを持つラベルを取得
                top_label = max(score_dict, key=score_dict.get)
                top_score = score_dict[top_label]

                # 情報を辞書にまとめてリストに追加
                all_candidates.append({
                    "crop": crop,
                    "label": top_label,
                    "score": float(top_score),
                    "bbox": bboxes[i]
                })
            all_candidates.sort(key=lambda x: x["score"], reverse=True)
            if all_candidates:
                import matplotlib.pyplot as plt
                import math

                num_imgs = len(all_candidates)
                cols = 5
                rows = math.ceil(num_imgs / cols)

                plt.figure(figsize=(cols * 3, rows * 3))
                for i, cand in enumerate(all_candidates):
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(cand["crop"])
                    plt.title(f"idx:{i} {cand['label']}\n({cand['score']:.2f})")
                    plt.axis('off')
                
                plt.tight_layout()
                plt.show()
           
        # ============================================
        # === サイズ・座標の整合性チェックログ ===
        print(f"\n--- Output Data Consistency Check ---")
        print(f"Original Image Size : {H_before_crop}x{W_before_crop}")
        print(f"Cropped ROI Size    : {cH}x{cW} (Offset: y={offset_y}, x={offset_x})")
        
        for label, inst in best_per_label.items():
            mask_shape = inst["mask"].shape
            bbox = inst["bbox"]
            center = inst["center_px"]
            print(f"[{label}]:")
            print(f"  - Mask Shape  : {mask_shape}  {'[OK]' if mask_shape[:2] == (H_before_crop, W_before_crop) else '[ERROR: Size Mismatch]'}")
            print(f"  - BBox        : {bbox}")
            print(f"  - Center Pixel: {center}")
            # Centerが画像範囲内かチェック
            if not (0 <= center[0] < W_before_crop and 0 <= center[1] < H_before_crop):
                print(f"  - [WARNING]: Center pixel is OUTSIDE the original image dimensions!")
        print(f"--------------------------------------\n")

        # ★ 必ず Dict を返す
        return {
            "instances": best_per_label,
            "fps": self.ema_fps,
        }
    def process_frame2(
        self,
        frame_bgr: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        target_label: Optional[str] = None,
        save_best_crops: bool = True,
        debug_dir: str = "debug_crops",
    ) -> Dict[str, Any]:
        """
        process_frame 改良版（ラベル別ベストを返す + キャッシュ対応）

        - 各cropについて CLIPを全ラベルでスコア → labelごとのargmaxを保持
        - maskgen_interval の間は前回の結果を返す（空にならない）
        - target_label が指定されたら、そのラベルだけ返す（LLM連携用）
        - best候補の crop/masked crop を保存可能（検証用）

        Returns:
        {
            "instances": {label: {"mask","bbox","center_px","score","depth_m","crop_path","masked_crop_path"}},
            "fps": float,
            "candidates": [...]  # 任意のデバッグ
        }
        """
        t0 = time.time()

        rgb = to_rgb(frame_bgr)
        H, W = rgb.shape[:2]

        need_new_masks = (
            self.frame_count % self.maskgen_interval == 0
            or (self.last_instances is None)
            or (len(self.last_instances) == 0)
        )

        # --- helper: depth median in mask (頑健) ---
        def depth_m_from_mask(mask: np.ndarray) -> Optional[float]:
            if (not self.enable_depth) or (depth_frame is None) or (mask is None):
                return None
            if depth_frame.shape[:2] != (H, W):
                return None
            vals = depth_frame[mask > 0]
            vals = vals[vals > 0]
            if vals.size == 0:
                return None
            return float(np.median(vals)) / 1000.0  # mm -> m

        # --- helper: save crop & masked crop for BEST only ---
        def save_best_crop(label: str, score: float, bbox, mask_full: Optional[np.ndarray]):
            if not save_best_crops or bbox is None:
                return None, None

            os.makedirs(debug_dir, exist_ok=True)
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))
            if (x2 - x1) <= 2 or (y2 - y1) <= 2:
                return None, None

            crop = rgb[y1:y2, x1:x2].copy()
            ts = int(time.time() * 1000)
            base = os.path.join(debug_dir, f"best_{label}_s{score:.3f}_{ts}")

            crop_path = base + ".png"
            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            masked_path = None
            if mask_full is not None:
                m = mask_full[y1:y2, x1:x2]
                m = (m > 0).astype(np.uint8) * 255
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                masked = cv2.bitwise_and(crop_bgr, crop_bgr, mask=m)
                masked_path = base + "_masked.png"
                cv2.imwrite(masked_path, masked)

            return crop_path, masked_path

        # =========================
        # 1) 新規にSAM2+CLIPを回す
        # =========================
        if need_new_masks:
            self.sam.set_image(rgb)
            masks = self.sam.generate_masks(rgb)
            print(f"DEBUG: SAM2 generated {len(masks)} raw masks.")

            masks = filter_masks_by_area(
                masks,
                H, W,
                self.min_area,
                self.max_area_frac,
            )

            crops, bboxes = masks_to_crops_and_bboxes(rgb, masks)
            print(f"DEBUG: After area filter, {len(masks)} masks remain.")
            print("\n--- DEBUG: CLIP crop & score list ---")

            best_per_label: Dict[str, Dict[str, Any]] = {}
            candidates = []  # 任意：あとで分析したい人向け

            for i, crop in enumerate(crops):
                score_dict = self.clip.score_single(crop)
                if score_dict is None:
                    continue

                bbox = bboxes[i]
                x1, y1, x2, y2 = bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # （任意）候補ログ
                candidates.append({
                    "index": i,
                    "bbox": bbox,
                    "center_px": (cx, cy),
                    "scores": {k: float(v) for k, v in score_dict.items()},
                })

                # labelごとのargmax更新
                for label, score in score_dict.items():
                    prev = best_per_label.get(label)
                    if (prev is None) or (score > prev["score"]):
                        # bboxベースで refine（あなたの現方式踏襲）
                        refined_mask = self.sam.predict_by_bbox(bbox)

                        # depth：中心点より mask中央値が安定（推奨）
                        depth_m = depth_m_from_mask(refined_mask)
                        if depth_m is None and self.enable_depth and depth_frame is not None:
                            # 保険：中心点深度
                            if depth_frame.shape[:2] == (H, W):
                                d = float(depth_frame[cy, cx])
                                if d > 0:
                                    depth_m = d / 1000.0

                        crop_path, masked_path = save_best_crop(label, float(score), bbox, refined_mask)

                        best_per_label[label] = {
                            "mask": refined_mask,
                            "bbox": bbox,
                            "center_px": (cx, cy),
                            "score": float(score),
                            "depth_m": depth_m,
                            "crop_path": crop_path,
                            "masked_crop_path": masked_path,
                        }

            # キャッシュ更新（重要）
            self.last_instances = best_per_label
            self.last_candidates = candidates

        # =========================
        # 2) maskgen_interval中はキャッシュを返す
        # =========================
        instances = self.last_instances if self.last_instances is not None else {}
        candidates = self.last_candidates if self.last_candidates is not None else []

        # target_label 指定があるならフィルタ（LLMが決めたラベルだけ欲しいとき用）
        if target_label is not None:
            if target_label in instances:
                instances = {target_label: instances[target_label]}
            else:
                # 無い場合は空を返す（main側でフォールバックしてもOK）
                instances = {}

        # FPS
        self.frame_count += 1
        dt = max(time.time() - t0, 1e-6)
        fps = 1.0 / dt
        self.ema_fps = fps if self.ema_fps is None else 0.9 * self.ema_fps + 0.1 * fps

        return {
            "instances": instances,
            "fps": self.ema_fps,
            "candidates": candidates,
        }





