# asr_llm_prompts.py
# このスクリプトは、音声文字起こしと LLM を使用したプロンプト生成を行います。
# Feeding Robot のための CLIP プロンプトを生成するためのユーティリティ関数を提供します。

import json, os
from typing import Dict, List
from openai import OpenAI

# システムプロンプト: CLIP プロンプトを生成するための基本的な指示
PROMPT_SYSTEM = (
    "You build CLIP prompts for a feeding robot. "
    "Return JSON with keys: noodles, porridge, soup, non_food; "
    "values: arrays of 3-8 short, visual-only phrases."
)

# ユーザープロンプトテンプレート: 音声テキストを基にプロンプトを生成するためのテンプレート
PROMPT_USER_TMPL = (
    "Voice intent:\n"
    "\"\"\"{voice_text}\"\"\"\n"
    "Constraints:\n"
    "- Short, visual descriptors (color/shape/texture/plating)\n"
    "- Keep one language inside a class\n"
    "- No extra keys\n"
    "Return pure JSON only.\n"
)

# 音声ファイルを文字起こしする関数
# 引数:
# - path: 音声ファイルのパス
# - client: OpenAI クライアント
# - model: 使用する音声文字起こしモデル (デフォルト: gpt-4o-transcribe)
# 戻り値:
# - 文字起こしされたテキスト

def transcribe_audio(path: str, client: OpenAI, model="gpt-4o-transcribe") -> str:
    with open(path, "rb") as f:
        r = client.audio.transcriptions.create(model=model, file=f)
    return r.text

# テキストから CLIP プロンプトを生成する関数
# 引数:
# - text: 入力テキスト
# - client: OpenAI クライアント
# - model: 使用する LLM モデル (デフォルト: gpt-4o-mini)
# 戻り値:
# - CLIP プロンプトの辞書 (キー: noodles, porridge, soup, non_food)

def llm_prompts_from_text(text: str, client: OpenAI, model="gpt-4o-mini") -> Dict[str, List[str]]:
    r = client.responses.create(
        model=model,
        input=[{"role":"system","content":PROMPT_SYSTEM},
               {"role":"user","content":PROMPT_USER_TMPL.format(voice_text=text)}]
    )
    try:
        # 応答を JSON として解析
        data = json.loads(r.output_text)
        for k in ["noodles","porridge","soup","non_food"]:
            assert k in data and isinstance(data[k], list)
        return data
    except Exception:
        # エラー時のデフォルト値
        return {
            "noodles":  ["thin yellow noodles in clear broth"],
            "porridge": ["smooth pale porridge in shallow bowl"],
            "soup":     ["golden soup with small oil drops"],
            "non_food": ["empty ceramic plate"]
        }
