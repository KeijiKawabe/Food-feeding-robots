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
