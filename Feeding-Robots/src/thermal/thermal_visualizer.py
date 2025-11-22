# src/thermal/thermal_visualizer.py

import cv2
from .thermal_gpt_system import FOOD_AREAS

def draw_thermal_areas(img_bgr, data, area_temps):
    vis = img_bgr.copy()

    for name, rect in FOOD_AREAS.items():
        x1, x2 = rect["x1"], rect["x2"]
        y1, y2 = rect["y1"], rect["y2"]

        # 温度
        t_max = area_temps[name]["max"]
        t_mean = area_temps[name]["mean"]

        # 色設定
        color = (0, 200, 0)
        if t_max is not None:
            if t_max > 65: color = (0, 0, 255)
            elif t_max < 30: color = (255, 0, 0)

        # 矩形
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # ラベル
        text = f"{name}: {t_max:.1f}C" if t_max is not None else name
        cv2.putText(vis, text, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis
