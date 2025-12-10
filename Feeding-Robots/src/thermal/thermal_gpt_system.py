# src/thermal/thermal_gpt_system.py

import numpy as np
import cv2
import time
import json
from openai import OpenAI
from .pi160_controller import PI160Controller


# -----------------------------
# 区域定義（好きなだけ細かく定義可能）
# Thermal画像が (H=120, W=160) の想定
# -----------------------------
FOOD_AREAS = {
    "curry":  {"x1": 0,  "x2": 40,  "y1": 0,  "y2": 30},
    "rice":       {"x1": 40, "x2": 90,  "y1": 0,  "y2": 120},
    "salad":      {"x1": 90, "x2": 160, "y1": 0,  "y2": 120},
}


class ThermalGPTSystem:
    """
    Thermal-only Task Planner
    -------------------------
    食材分類はしない。
    Thermal上の区域ごとに split し、その温度情報から
    LLM に「次に食べるべき区域」を判断させる。
    """

    def __init__(self, openai_api_key, target_temp=65):
        self.client = OpenAI(api_key=openai_api_key)
        self.camera = PI160Controller()
        self.target_temp = target_temp

    # -----------------------------
    # Thermal capture
    # -----------------------------
    def capture(self):
        if not self.camera.lib or not self.camera.handle:
            print("✗ Thermal camera not initialized.")
            return None

        # バッファクリア
        self.camera.get_palette_image()
        self.camera.get_thermal_data()
        time.sleep(0.1)

        data = self.camera.get_thermal_data()
        img = self.camera.get_palette_image()

        if data is None or img is None:
            return None

        return data, img

    # -----------------------------
    # 区域ごとの温度計算
    # -----------------------------
    def compute_area_temps(self, data):
        """
        各区域ごとに:
        - max 温度
        - mean 温度
        を計算し dict を返す。
        """
        results = {}

        for name, rect in FOOD_AREAS.items():
            x1, x2 = rect["x1"], rect["x2"]
            y1, y2 = rect["y1"], rect["y2"]

            sub = data[y1:y2, x1:x2]

            if sub.size == 0:
                results[name] = {"max": None, "mean": None}
                continue

            results[name] = {
                "max": float(np.max(sub)),
                "mean": float(np.mean(sub)),
            }

        return results

    # -----------------------------
    # GPT に判断させる
    # -----------------------------
    def ask_gpt(self, area_temps, history=None):
        if history is None:
            history = []

        # 区域名と温度情報を整形
        text_areas = "\n".join(
            f"{name}: max={v['max']:.1f}℃, mean={v['mean']:.1f}℃"
            for name, v in area_temps.items()
        )

        prompt = f"""
あなたはロボットの食事介助タスクプランナーです。

食品の種類は Thermal カメラによる区域分けで管理しています。
RGB は使用しません。

以下は各区域の温度データです（max = 最も熱い点を示します）。

--- 区域温度 ---
{text_areas}

安全上限温度: {self.target_temp}℃
前回食べた区域: {history}

目的：
1. 各区域が「熱すぎる」かどうか評価
2. 次にどの区域から食べるべきかを温度や三角食べの考え方をもとに決定
3. 理由を簡潔に英語で説明
出力に関しては以下の JSON フォーマットに厳密に従ってください。

出力形式（JSONのみ）：
{{
  "next_food": "<食品ラベル（curry, rice, salad のいずれか）>",
  "too_hot": true/false,
  "reason": "<理由>"
}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )

        txt = response.choices[0].message.content

        try:
            return json.loads(txt)
        except:
            print("⚠ JSON パース失敗:", txt)
            return {
                "next_area": None,
                "too_hot": None,
                "reason": "parse_error"
            }

    # -----------------------------
    # 外部から呼ぶ関数
    # -----------------------------
    def process(self, history=None):
        thermal = self.capture()
        if thermal is None:
            return None

        data, _ = thermal

        # 区域ごとの温度を計算
        area_temps = self.compute_area_temps(data)

        # GPT に意思決定してもらう
        decision = self.ask_gpt(area_temps, history)

        # デバッグのため温度情報も返す
        decision["area_temps"] = area_temps

        return decision
    
    def run_test(self, target_temp=65, save_image=True):
        """
        Thermal camera test + GPT safety check.
        Returns True if safe, False if unsafe.
        """

        # ---- 1. Thermal capture ----
        palette, raw = self.camera.capture_frame()

        if raw is None:
            print("✗ Thermalカメラから画像を取得できませんでした")
            return False

        # 温度変換
        temperature = raw.astype("float32") * 0.1 - 100.0
        temp_max = float(temperature.max())

        print(f"取得温度: max={temp_max:.2f}℃")

        # ---- 2. Save image ----
        # ---- 3. GPT safety check ----
        result = self.ask_gpt_temperature_judgement(temp_max, target_temp)

        print("GPT 判定:", result)

        return ("OK" in result)
    
    def ask_gpt_temperature_judgement(self, temp, target):
        prompt = f"""
食品の最大温度は {temp:.1f}℃ です。
安全に食べられる温度（上限）は {target}℃ とします。

この食べ物は安全に食べられますか？
回答は「OK」または「NG」のみ返してください。
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a strict food safety checker."},
                      {"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()




    # -----------------------------
    # Cleanup
    # -----------------------------
    def cleanup(self):
        self.camera.disconnect()
    
        # -----------------------------
    # RGB 認識より前に「次に食べる食材」を決める関数
    # -----------------------------
    def decide_next_food(self, history=None):
        """
        Thermal 画像 → 区域温度の計算 → GPT 判定
        という1セットの処理を行い、
        PerceptionPipeline に渡すべき next_food を返す。

        出力例：
        {
            "next_food": "curry",
            "too_hot": false,
            "reason": "...",
            "area_temps": {...}
        }
        """
        # 1) Thermal capture
        thermal = self.capture()
        if thermal is None:
            return {
                "next_food": None,
                "too_hot": None,
                "reason": "thermal_capture_failed",
                "area_temps": None
            }

        data, _ = thermal

        # 2) 区域温度の計算
        area_temps = self.compute_area_temps(data)

        # 3) GPT に意思決定させる
        decision = self.ask_gpt(area_temps, history)

        # 4) デバッグ用途で温度も返す
        decision["area_temps"] = area_temps

        return decision

