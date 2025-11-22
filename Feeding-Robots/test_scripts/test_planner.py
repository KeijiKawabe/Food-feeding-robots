# test_scripts/test_planner.py

import os
import sys

# プロジェクトの src をパスに追加
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.planner.task_planner import TaskPlanner


def test_case_ok():
    """
    Thermal も RGB も "curry" で一致していて、
    温度も安全なケース → allowed = True になるはず
    """
    rgb_result = {
        "label": "curry",
        "bbox": [100, 50, 200, 160],
        "center_px": (150, 100),
    }

    thermal_decision = {
        "next_food": "curry",
        "too_hot": False,
        "reason": "temperature is safe",
    }

    planner = TaskPlanner()
    plan = planner.plan(rgb_result, thermal_decision)

    print("=== TEST: OK case ===")
    print(plan)
    print()


def test_case_too_hot():
    """
    Thermal 側で「curry は熱すぎる」と判断されたケース
    → ロボットは動かない（allowed = False）
    """
    rgb_result = {
        "label": "curry",
        "bbox": [100, 50, 200, 160],
        "center_px": (150, 100),
    }

    thermal_decision = {
        "next_food": "curry",
        "too_hot": True,
        "reason": "curry max temp 75C (too hot)",
    }

    planner = TaskPlanner()
    plan = planner.plan(rgb_result, thermal_decision)

    print("=== TEST: Too hot case ===")
    print(plan)
    print()


def test_case_mismatch():
    """
    Thermal は "curry" を食べたいと言っているが、
    RGB 側の認識ラベルが "rice" になっているケース
    → allowed = False（セーフティのため停止）
    """
    rgb_result = {
        "label": "rice",
        "bbox": [50, 60, 160, 170],
        "center_px": (105, 115),
    }

    thermal_decision = {
        "next_food": "curry",
        "too_hot": False,
        "reason": "curry and rice both safe, choose curry",
    }

    planner = TaskPlanner()
    plan = planner.plan(rgb_result, thermal_decision)

    print("=== TEST: Mismatch case ===")
    print(plan)
    print()


def main():
    print("Project root:", PROJECT_ROOT)
    print("Python path includes src?:", PROJECT_ROOT in sys.path)
    print()

    test_case_ok()
    test_case_too_hot()
    test_case_mismatch()


if __name__ == "__main__":
    main()
