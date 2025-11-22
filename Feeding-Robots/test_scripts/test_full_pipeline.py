# test_scripts/test_full_pipeline.py

import os
import sys
import cv2

# -----------------------------
# Python パス設定
# -----------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.pipeline import PerceptionPipeline
from src.planner.task_planner import TaskPlanner
from src.thermal.thermal_gpt_system import ThermalGPTSystem
from src.utils.misc import draw_mask_on_image


def main():
    print("Project root:", PROJECT_ROOT)

    # -----------------------------
    # 1. OpenAI APIキー取得（Thermal + GPT 用）
    # -----------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY が環境変数として設定されていません。")
        print("   Windows の『環境変数の編集』から OPENAI_API_KEY を設定してください。")
        return

    # -----------------------------
    # 2. SAM2 の設定ファイル / 重み
    # -----------------------------
    CFG = os.path.join(PROJECT_ROOT, "..", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml")
    CKPT = os.path.join(PROJECT_ROOT, "..", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")

    if not os.path.exists(CFG):
        print("❌ SAM2 config が見つかりません:", CFG)
        return
    if not os.path.exists(CKPT):
        print("❌ SAM2 checkpoint が見つかりません:", CKPT)
        return

    # -----------------------------
    # 3. PerceptionPipeline (RGB: SAM2 + CLIP)
    # -----------------------------
    clip_prompts = {
        "rice": ["a photo of rice"],
        "curry": ["a plate of curry", "a bowl of curry"],
        "salad": ["a bowl of salad", "vegetable salad"],
        # 必要に応じて追加
    }

    pipe = PerceptionPipeline(
        sam2_cfg=CFG,
        sam2_ckpt=CKPT,
        device="cuda",
        maskgen_interval=1,        # 静止画1枚なので 1 でOK
        min_area=1000,
        max_area_frac=0.5,
        clip_model="ViT-B/32",
        clip_prompts=clip_prompts,
    )

    # -----------------------------
    # 4. Thermal + GPT (温度ベースのタスクプランナー)
    # -----------------------------
    thermal_system = ThermalGPTSystem(openai_api_key=api_key, target_temp=65)

    # -----------------------------
    # 5. TaskPlanner (統合意思決定)
    # -----------------------------
    planner = TaskPlanner()

    # -----------------------------
    # 6. RGB テスト画像の読み込み
    # -----------------------------
    img_path = os.path.join(THIS_DIR, "..", "data", "test_image.jpg")
    img_path = os.path.abspath(img_path)

    print("\n=== Full Pipeline Test ===")
    print("RGB image:", img_path)

    bgr = cv2.imread(img_path)
    if bgr is None:
        print("❌ 画像を読み込めません:", img_path)
        return

    # -----------------------------
    # 7. RGB パイプライン実行 (SAM2 + CLIP)
    # -----------------------------
    rgb_result = pipe.process_frame(bgr, depth_frame=None)

    print("\n--- RGB Perception Output ---")
    print("label    :", rgb_result.get("label"))
    print("score    :", rgb_result.get("score"))
    print("bbox     :", rgb_result.get("bbox"))
    print("center_px:", rgb_result.get("center_px"))
    print("depth_m  :", rgb_result.get("depth_m"))
    print("fps(EMA) :", rgb_result.get("fps"))

    # -----------------------------
    # 8. Thermal + GPT による温度評価 & 食材選択
    # -----------------------------
    print("\n--- Thermal + GPT Decision ---")
    # 食事履歴（とりあえず空 or ダミーでOK）
    history = []  # 例: ["rice", "soup"]

    thermal_decision = thermal_system.process(history=history)
    if thermal_decision is None:
        print("❌ Thermal/GPT の処理に失敗しました。")
        thermal_system.cleanup()
        return

    print("next_food:", thermal_decision.get("next_food"))
    print("too_hot :", thermal_decision.get("too_hot"))
    print("reason  :", thermal_decision.get("reason"))
    print("stats   :", thermal_decision.get("stats"))

    # -----------------------------
    # 9. LLM (Thermal) の候補で RGB を再評価してから Planner に渡す
    # -----------------------------
    print("\n--- Re-evaluate RGB using LLM-suggested food ---")
    llm_label = thermal_decision.get("next_food")
    rgb_llm = None
    if llm_label:
        print(f"LLM suggested: {llm_label} -> asking CLIP to find that label on RGB")
        rgb_llm = pipe.process_frame(bgr, depth_frame=None, target_label=llm_label)
        print("RGB (LLM prompted) -> label:", rgb_llm.get("label"), "score:", rgb_llm.get("score"), "bbox:", rgb_llm.get("bbox"))

    # Decide which RGB result to give the planner: prefer LLM-guided result if it found a match
    if rgb_llm is not None and rgb_llm.get("label") is not None:
        rgb_for_planner = rgb_llm
    else:
        rgb_for_planner = rgb_result

    print("\n--- TaskPlanner Output ---")
    plan = planner.plan(rgb_for_planner, thermal_decision)
    print(plan)

    # -----------------------------
    # 10. 結果に応じた処理（ここが将来 xArm 制御に繋がる）
    # -----------------------------
    if not plan["allowed"]:
        print("\n⚠ プラン却下:", plan["reason"])
        print("   → ロボットは動作せず待機（安全側）")
    else:
        print("\n✅ プラン承認:", plan["reason"])
        print("   次に掬う食材:", plan["food"])
        print("   画素座標 center_px:", plan["center_px"])
        print("   → ここから RealSense Depth → xArm 座標変換につなげる")

    # -----------------------------
    # 11. RGB 画像にマスクとBBoxを可視化（デバッグ用）
    # -----------------------------
    mask = rgb_for_planner.get("mask")
    bbox = rgb_for_planner.get("bbox")

    vis = bgr.copy()
    if mask is not None:
        vis = draw_mask_on_image(vis, mask)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 2)

    cv2.imshow("Full Pipeline (RGB + SAM2 + CLIP)", vis)
    print("\nウィンドウに RGB 認識結果を表示しました。何かキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # -----------------------------
    # 12. Thermal カメラのクリーンアップ
    # -----------------------------
    thermal_system.cleanup()


if __name__ == "__main__":
    main()
