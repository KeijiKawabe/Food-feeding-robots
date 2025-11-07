import os, sys, json, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from audio.asr_llm_prompts import transcribe_audio, llm_prompts_from_text
from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", type=str, default="")
    ap.add_argument("--asr_model", type=str, default="gpt-4o-transcribe")
    ap.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    args = ap.parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[SKIP] OPENAI_API_KEY not set. Falling back to fake text.")
        voice_text = "今日は麺類を食べたい。スープは澄んでいて、麺は細めで黄色っぽい。"
        prompts = llm_prompts_from_text(voice_text, client=ClientStub(), model=args.llm_model)
    else:
        client = OpenAI(api_key=api_key)
        if args.voice and os.path.exists(args.voice):
            text = transcribe_audio(args.voice, client=client, model=args.asr_model)
            print("[ASR]", text)
            prompts = llm_prompts_from_text(text, client=client, model=args.llm_model)
        else:
            print("[SKIP] no --voice. Using manual text.")
            text = "お粥のような白っぽいとろみのある食べ物を探して。器は白。"
            prompts = llm_prompts_from_text(text, client=client, model=args.llm_model)

    print(json.dumps(prompts, ensure_ascii=False, indent=2))
    for k in ["noodles","porridge","soup","non_food"]:
        assert k in prompts and isinstance(prompts[k], list)
    print("OK: ASR/LLM prompt generation works (or fallback).")

# 簡易スタブ（OPENAI未設定時用）
class ClientStub:
    @property
    def responses(self):
        class ResponsesStub:
            def create(self, *a, **k):
                return {
                    "output_text": json.dumps({
                        "noodles": ["thin yellow noodles in clear broth"],
                        "porridge": ["smooth pale porridge in shallow bowl"],
                        "soup": ["golden soup with small oil drops"],
                        "non_food": ["empty ceramic plate"]
                    })
                }
        return ResponsesStub()
    def __getattr__(self, name): return self
    def create(self, *a, **k):
        class R: 
            output_text = json.dumps({
                "noodles":["thin yellow noodles in clear broth"],
                "porridge":["smooth pale porridge in shallow bowl"],
                "soup":["golden soup with small oil drops"],
                "non_food":["empty ceramic plate"]
            })
        return R()

if __name__ == "__main__":
    main()
