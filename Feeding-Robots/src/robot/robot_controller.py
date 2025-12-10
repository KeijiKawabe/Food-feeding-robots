from xarm.wrapper import XArmAPI
import os
import time
import csv
from typing import Dict, Any
class Robotcontroller:
    @staticmethod
    def init_xarm(ip: str) -> XArmAPI:
        """
        xArm 本体と接続して「動ける状態」にする初期化関数。
        """
        arm = XArmAPI(ip)
        print(f"[xArm] 接続中... IP={ip}")

        arm.motion_enable(True)
        arm.set_mode(0)   # position mode
        arm.set_state(0)  # ready
        time.sleep(1.0)

        err, warn = arm.get_err_warn_code()
        print(f"[xArm] err={err}, warn={warn}")
        if err != 0:
            print("⚠ xArm にエラーが残っています。GUI で一度クリアしておくと安心です。")

        return arm
    
    def play_traj_file(arm: XArmAPI, traj_path: str):
        """
        xArm 用の traj ファイルを読み込んで再生する簡易関数。

        ここでは例として、
        - CSV 形式
        - ヘッダに joint1, joint2, ..., joint7
        - 各行に rad 単位の関節角

        を想定している。
        実際のファイル形式や API 名は、使っている SDK / エクスポート形式に合わせて調整してください。
        """
        if not os.path.exists(traj_path):
            print(f"⚠ traj ファイルが見つかりません: {traj_path}")
            return

        print(f"[xArm] traj 再生開始: {traj_path}")

        try:
            with open(traj_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        joints = [float(row[f"joint{i}"]) for i in range(1, 8)]
                    except KeyError:
                        print("⚠ CSV の列名が想定と違います。joint1..joint7 を想定しています。")
                        break

                    # xArm SDK の関節角コマンド
                    # 実際の API 名・引数はインストールしている SDK に合わせて調整してください。
                    # 例: arm.set_servo_angle_j(joints, is_radian=True, wait=True)
                    code = arm.set_servo_angle(joints, is_radian=True, wait=True)
                    if code != 0:
                        print(f"⚠ xArm コマンドエラー: code={code}")
                        break

            print("[xArm] traj 再生終了")
        except Exception as e:
            print("⚠ traj 再生中に例外が発生しました:", e)


    def move_robot_to_food(
        arm: XArmAPI,
        center_px,
        depth_m: float,
        calib: Dict[str, Any],
        label: str,
    ):
        TRAJ_MAP: Dict[str, str] = {
        "rice": "traj/rice_scoop.csv",
        }
        TRAJ_TO_MOUTH = "traj/to_mouth.csv"
        """
        「次の一口」の食材ラベルをもとに、
        あらかじめティーチングしておいた traj を再生する入口関数。

        v0 では hand-eye や Base 座標は一切使わず、
        皿固定 & traj 再生のみで対応する前提。
        """
        print("")
        print("========== [ROBOT ACTION] ==========")
        print(f"  target food  : {label}")
        print(f"  image center : {center_px}")
        print(f"  depth (m)    : {depth_m}")
        print("  ※ hand-eye を使わず、traj 再生のみで動作します。")
        print("====================================")

        # 1. 食材ごとの「すくい」traj
        traj_scoop = TRAJ_MAP.get(label, None)
        if traj_scoop is None:
            print(f"⚠ ラベル {label} に対応する traj が定義されていません。")
            return

        # 2. 皿から食材をすくう動作
        Robotcontroller.play_traj_file(arm, traj_scoop)

        # 3. 口元に運ぶ動作（共通）
        if os.path.exists(TRAJ_TO_MOUTH):
            Robotcontroller.play_traj_file(arm, TRAJ_TO_MOUTH)
        else:
            print("⚠ 口元に運ぶ traj (TRAJ_TO_MOUTH) が見つかりません。")

        