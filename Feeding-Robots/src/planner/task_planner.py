# src/planner/task_planner.py

class TaskPlanner:
    """
    RGB（SAM2 + CLIP）と Thermal（GPT）の結果を統合し、
    次にロボットがすくう食材の位置を決定する。
    """

    def __init__(self):
        pass

    def plan(self, rgb_result, thermal_decision):
        """
        rgb_result: PerceptionPipeline.process_frame() の出力
        thermal_decision: ThermalGPTSystem.process() の JSON

        返回値:
        {
            "food": "curry",
            "bbox": [...],
            "center_px": (...),
            "allowed": True/False,
            "reason": "..."
        }
        """

        # --- Thermal による食材選定 ---
        next_food = thermal_decision.get("next_food")
        too_hot = thermal_decision.get("too_hot")
        reason = thermal_decision.get("reason")

        # --- RGB の認識結果 ---
        detected_label = rgb_result.get("label")
        bbox = rgb_result.get("bbox")
        center = rgb_result.get("center_px")

        # --- 1) 温度が危険なら停止 ---
        if too_hot:
            return {
                "food": next_food,
                "bbox": None,
                "center_px": None,
                "allowed": False,
                "reason": f"{next_food} is too hot: {reason}"
            }

        # --- 2) LLMが選んだ食材とRGBが一致しない ---
        if detected_label != next_food:
            return {
                "food": next_food,
                "bbox": None,
                "center_px": None,
                "allowed": False,
                "reason": f"Mismatch: LLM wants {next_food}, but RGB sees {detected_label}"
            }

        # --- 3) すべてOK：ロボットの行動計画OK ---
        return {
            "food": next_food,
            "bbox": bbox,
            "center_px": center,
            "allowed": True,
            "reason": "OK"
        }
    def decide_next_bite_with_plates_llm(
        client: OpenAI,
        plates_info: list,
        safe_temp_max: float,
        history: list,
    ) -> dict:
        """
        複数の皿情報（plates_info = [{plate_id, label, temp{min,max,mean}, times_eaten}, ...]）
        を LLM に渡して、「どの皿から次の一口を取るか（または何も取らないか）」を決めてもらう。

        戻り値の例:
        {
        "choose": true/false,          # 今回一口を取るかどうか
        "plate_id": "A",               # 選んだ皿ID（choose=falseのときは null でもよい）
        "label": "rice",               # その皿のラベル
        "reason": "short English reason"
        }
        """
        plates_json = json.dumps(plates_info, ensure_ascii=False)

        prompt = f"""
    You are a task planner for a meal-assistance robot.

    The robot has several plates in front of the user.
    For each plate, we have:
    - plate_id: an identifier such as "A", "B", "C"
    - label: which food is on this plate (e.g., "rice", "curry")
    - temp: statistics of the temperature on that plate's region
    - min: minimum temperature in °C
    - max: maximum temperature in °C
    - mean: mean temperature in °C
    - times_eaten: how many bites have already been taken from this food.

    Safety rule:
    - The safety upper bound for eating is {safe_temp_max:.1f} °C.
    If the temperature is clearly above this bound, the robot should NOT feed from that plate.

    Here is the current plate info in JSON:
    {plates_json}

    Eating history (sequence of food labels) so far:
    {history}

    Task:
    Decide whether the robot should feed one bite from one of these plates now.
    - If you think it is safe and reasonable, choose exactly one plate_id.
    - If you think no plate is safe or appropriate, choose not to feed this time.

    Output **strict JSON only** in this format:
    {{
    "choose": true or false,
    "plate_id": "<plate id or null>",
    "label": "<food label or null>",
    "reason": "<short English reason>"
    }}
    """

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            txt = resp.choices[0].message.content
            data = json.loads(txt)
            return data
        except Exception as e:
            print("⚠ Plate-based LLM 判定に失敗:", e)
            # 失敗した場合は「今回は feed しない」という安全側のデフォルト
            return {
                "choose": False,
                "plate_id": None,
                "label": None,
                "reason": "LLM_error",
            }
