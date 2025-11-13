
from .pi160_controller import PI160Controller
from openai import OpenAI
import numpy as np
import cv2
import time

class ThermalGPTSystem:
    """PI160 + GPT統合システム"""

    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.camera = PI160Controller()

    def capture_thermal_image(self):
            """温度データと画像を取得"""
            if not self.camera.lib or not self.camera.handle:
                print("✗ カメラが初期化されていません")
                return None, None
            
            # --- ↓↓↓ ここから修正箇所 ↓↓↓ ---

            # 【重要】バッファをクリアするため、一度フレームを空読みして捨てる
            # これにより、次に取得するデータが最新のものになる
            print("  バッファクリアのためフレームを空読みします...")
            self.camera.get_palette_image()  # パレット画像も同様に空読み
            self.camera.get_thermal_data()   # 温度データも空読み
            
            time.sleep(0.1)  # 新しいフレームが生成されるのを少し待つ

            # --- ↑↑↑ ここまで修正箇所 ↑↑↑ ---
            
            # ここからが実際に使用するデータの取得
            thermal_data = self.camera.get_thermal_data()
            if thermal_data is None:
                print("✗ 温度データ取得失敗")
                return None, None
            
            thermal_image = self.camera.get_palette_image()
            if thermal_image is None:
                print("✗ 画像取得失敗")
                return None, None
                
            return thermal_data, thermal_image

    def get_temp_stats(self, data):
        """温度統計"""
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data))
        }

    def analyze_with_gpt(self, image, stats, thermal_data, target_temp=65):
        """
        GPTに温度行列のサンプリングデータと統計値を送信して分析させる
        """
        # --- 1. 温度行列をサンプリングしてテキスト化 ---
        # 例：120x160 → 6x8 に圧縮して送信（トークン節約）
        sample = thermal_data[::20, ::20].round(1)
        matrix_text = "\n".join([" ".join(map(str, row)) for row in sample])

        # --- 2. 分析プロンプト作成 ---
        prompt = f"""
    あなたは熱画像の温度分布を分析するアシスタントです。

    次の温度行列は各位置の平均温度（摂氏）を表しています。
    各数値は1ピクセルではなく約20ピクセル分を平均したものです。

    温度行列（°C）:
    {matrix_text}

    統計情報:
    - 平均温度: {stats['mean']:.1f}℃
    - 最大温度: {stats['max']:.1f}℃
    - 最低温度: {stats['min']:.1f}℃
    - 目標温度: {target_temp}℃ 以下

    質問:
    1. 目標温度を超えている領域が全体のどの位置（上・中央・下など）にありますか？
    2. 平均温度と比較して特に高温な領域は全体のどの位置（上下左右)にありますか？
    3. この物体を安全に触れそうですか？（YES/NO）
    """

        # --- 3. GPTへ送信 ---
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # 軽量モデルでコスト低減
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0
        )

        return response.choices[0].message.content


    def run_test(self, target_temp=65, save_image=False):
        """テスト実行"""
        print("\n" + "="*60)
        print("PI160 + GPT分析開始")
        print("="*60)
        
        print("\n[1] サーマル画像取得中...")
        thermal_data, image = self.capture_thermal_image()
        
        if thermal_data is None:
            print("✗ テスト失敗")
            return False
        
        print(f"✓ 画像取得成功: {thermal_data.shape}")
        
        print("\n[2] 温度データ分析...")
        stats = self.get_temp_stats(thermal_data)
        print(f"  平均温度: {stats['mean']:.1f}℃")
        print(f"  最高温度: {stats['max']:.1f}℃")
        print(f"  最低温度: {stats['min']:.1f}℃")
        # サンプリングして簡易マップを表示（20ピクセルごと）
        sample = thermal_data[::20, ::20].round(1)
        print("\n[DEBUG] 温度行列サンプル:")
        for row in sample:
            print(" ".join(f"{v:5.1f}" for v in row))

        if save_image:
            filename = f"thermal_{int(time.time())}.jpg"
            cv2.imwrite(filename, image)
            print(f"  画像保存: {filename}")
        
        print("\n[3] GPT-4で分析中...")
        start = time.time()
        
        try:
            analysis = self.analyze_with_gpt(image, stats, thermal_data, target_temp)

            elapsed = time.time() - start
            
            print(f"✓ 分析完了 ({elapsed:.1f}秒)")
            print("\n--- GPT-4 分析結果 ---")
            print(analysis)
            print("--- 終了 ---\n")
            
            return True
        except Exception as e:
            print(f"✗ GPT分析エラー: {e}")
            return False

    def cleanup(self):
        """クリーンアップ"""
        self.camera.disconnect()