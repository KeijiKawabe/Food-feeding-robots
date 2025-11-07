import json, os
from typing import Dict, List
from openai import OpenAI

PROMPT_SYSTEM = (
    "You build CLIP prompts for a feeding robot. "
    "Return JSON with keys: noodles, porridge, soup, non_food; "
    "values: arrays of 3-8 short, visual-only phrases."
)
PROMPT_USER_TMPL = """\
Voice intent:
\"\"\"{voice_text}\"\"\"
Constraints:
- Short, visual descriptors (color/shape/texture/plating)
- Keep one language inside a class
- No extra keys
Return pure JSON only.
"""

def transcribe_audio(path: str, client: OpenAI, model="gpt-4o-transcribe") -> str:
    with open(path, "rb") as f:
        r = client.audio.transcriptions.create(model=model, file=f)
    return r.text

def llm_prompts_from_text(text: str, client: OpenAI, model="gpt-4o-mini") -> Dict[str, List[str]]:
    r = client.responses.create(
        model=model,
        input=[{"role":"system","content":PROMPT_SYSTEM},
               {"role":"user","content":PROMPT_USER_TMPL.format(voice_text=text)}]
    )
    try:
        data = json.loads(r.output_text)
        for k in ["noodles","porridge","soup","non_food"]:
            assert k in data and isinstance(data[k], list)
        return data
    except Exception:
        return {
            "noodles":  ["thin yellow noodles in clear broth"],
            "porridge": ["smooth pale porridge in shallow bowl"],
            "soup":     ["golden soup with small oil drops"],
            "non_food": ["empty ceramic plate"]
        }
