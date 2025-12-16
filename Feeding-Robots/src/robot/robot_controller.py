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
    
    def play_traj_file(arm: XArmAPI, traj_path: str, is_radian=True):
        if not os.path.exists(traj_path):
            print(f"⚠ traj ファイルが見つかりません: {traj_path}")
            return

        print(f"[xArm] traj 再生開始: {traj_path}")

        try:
            with open(traj_path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) == 0:
                print("⚠ traj ファイルが空です")
                return

            header = rows[0]
            data_rows = rows[1:]

            # ----------------------------
            # 1. ヘッダ有りか判定
            # ----------------------------
            has_header = any(
                "joint" in h.lower() or h.lower().startswith("j")
                for h in header
            )

            if has_header:
                # joint列を自動抽出
                joint_indices = []
                for i, h in enumerate(header):
                    hl = h.lower()
                    if "joint" in hl or hl.startswith("j"):
                        joint_indices.append((i, h))

                if len(joint_indices) != 7:
                    print(f"⚠ joint列が7個見つかりません: {[h for _,h in joint_indices]}")
                    return

                # joint1..7 順に並び替え
                joint_indices.sort(
                    key=lambda x: int("".join(filter(str.isdigit, x[1])))
                )

                indices = [i for i, _ in joint_indices]

            else:
                # ヘッダ無し（数値のみ）
                if len(header) < 7:
                    print(f"⚠ 列数が不足しています: {len(header)} 列")
                    return

                # ★ 先頭7列を joint とみなす
                indices = list(range(7))
                data_rows = rows

            # ----------------------------
            # 2. 再生
            # ----------------------------
            for row in data_rows:
                joints = [float(row[i]) for i in indices]
                code = arm.set_servo_angle(
                    angles=joints,
                    is_radian=True,
                    speed=0.5,
                    mvacc=1.0,
                    radius=0.0,   # ★ これが決定打
                    wait=True
                )



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

        