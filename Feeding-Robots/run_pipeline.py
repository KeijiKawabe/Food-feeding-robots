# run_pipeline.py
# このスクリプトは、リアルタイムの食品認識のために PerceptionPipeline を初期化して実行します。
# OpenAI の API を使用して音声を文字起こしし、LLM ベースのプロンプト生成を行います。
# パイプラインはカメラフィードからのビデオフレームを処理し、マスクやバウンディングボックスをオーバーレイします。

import os, cv2, argparse, json
from openai import OpenAI
from Feeding_Robots.src.pipeline import PerceptionPipeline
from Feeding_Robots.src.audio.asr_llm_prompts import transcribe_audio, llm_prompts_from_text
from Feeding_Robots.src.utils.misc import draw_mask_on_image

def main():
    # コマンドライン引数を解析
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2_cfg", required=True)
    ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--min_area", type=int, default=1000)
    ap.add_argument("--max_area_frac", type=float, default=0.5)
    ap.add_argument("--clip_model", default="ViT-B/32")
    ap.add_argument("--openai_api_key", default=None,
                    help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    ap.add_argument("--asr_model", default="gpt-4o-transcribe")
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--voice_path", default="")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    # PerceptionPipeline を初期化
    pipe = PerceptionPipeline(
        sam2_cfg=args.sam2_cfg, sam2_ckpt=args.sam2_ckpt, device=args.device,
        maskgen_interval=args.interval, min_area=args.min_area,
        max_area_frac=args.max_area_frac, clip_model=args.clip_model
    )

  # 管理用の変数として API キーを取得（CLI 引数があればそれを優先し、なければ環境変数を使う）
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened(): raise SystemExit("Camera open failed")

    # カメラフィードを開く
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened(): raise SystemExit("カメラのオープンに失敗しました")

    print("[INFO] q:終了 / u:音声からプロンプトを更新 / t:テキストを入力してプロンプトを更新")
    while True:
        # カメラからフレームを読み取る
        ok, frame = cap.read()
        if not ok: break

        # フレームをパイプラインで処理
        out = pipe.process_frame(frame)

        # 結果を可視化
        vis = draw_mask_on_image(frame.copy(), out["mask"], (0,200,0), 0.45)
        if out["bbox"] is not None:
            x1,y1,x2,y2 = out["bbox"]; cv2.rectangle(vis,(x1,y1),(x2,y2),(0,180,0),2)
        txt = f"FPS:{out['fps']:.1f}"
        if out["label"] and out["score"] is not None: txt += f" | {out['label']} {out['score']:.1f}"
        cv2.putText(vis, txt, (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,220,30), 2)
        cv2.imshow("食品認識", vis)

        # ユーザー入力を処理
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('u'):
            # 音声文字起こしからプロンプトを更新
            if not client or not args.voice_path or not os.path.exists(args.voice_path):
                print("[ERR] OpenAI キーと --voice_path が必要です"); continue
            text = transcribe_audio(args.voice_path, client, args.asr_model)
            prompts = llm_prompts_from_text(text, client, args.llm_model)
            pipe.update_clip_prompts(prompts); print("[OK] プロンプトが更新されました")
        elif k == ord('t'):
            # ユーザー入力のテキストからプロンプトを更新
            if not client: print("[ERR] OpenAI キーが必要です"); continue
            print("プロンプトテキストを入力してください:"); text = input().strip()
            prompts = llm_prompts_from_text(text, client, args.llm_model)
            pipe.update_clip_prompts(prompts); print("[OK] プロンプトが更新されました")

    # リソースを解放
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
