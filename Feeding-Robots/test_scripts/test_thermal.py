# test_scripts/test_thermal.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.thermal.thermal_gpt_system import ThermalGPTSystem


def main():
    # すでに Windows の環境変数に設定されている API_KEY を取得
    API_KEY = os.getenv("OPENAI_API_KEY")

    if not API_KEY:
        print("❌ OPENAI_API_KEY が環境変数として設定されていません")
        print("   Windows の「環境変数の編集」で設定してください")
        return

    system = ThermalGPTSystem(openai_api_key=API_KEY)

    print("\n=== Thermal Camera + GPT Safety Test ===")

    ok = system.run_test(target_temp=65, save_image=True)

    if ok:
        print("✓ テスト成功")
    else:
        print("✗ テスト失敗")

    system.cleanup()


if __name__ == "__main__":
    main()
