#!/usr/bin/env python3
"""Запись одного эпизода обученной политики в симуляторе на видеофайл.

Прогоняет политику (по умолчанию BC warm-start, runs/sac_bc/final.zip) в
SortingPickEnv, перебирает сиды до успешной маршрутизации и сохраняет этот эпизод
(обзорная камера cam_global) с подписью товар/зона/результат.

Пример:
    python scripts/record_policy.py --policy runs/sac_bc/final.zip --out policy_run.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from robozone.rl_env import SortingPickEnv  # noqa: E402

_FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(sz):
    for p in _FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


ZONE_NAME = {"B": "B — сортировщик", "C": "C — негабарит", "D": "D — доупаковка"}


def caption(frame, obj, zone, cur, done, ok, f_big, f_sm):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    d.rectangle([0, 0, W, 52], fill=(15, 20, 35, 200))
    d.text((20, 12), "RoboZone · обученная политика (BC warm-start)",
           fill=(225, 235, 250), font=f_big)
    d.rectangle([0, H - 46, W, H], fill=(15, 20, 35, 200))
    d.text((20, H - 38), f"товар: {obj}   цель: {ZONE_NAME[zone]}",
           fill=(255, 255, 255), font=f_sm)
    if done:
        col = (90, 220, 120) if ok else (240, 120, 120)
        d.text((W - 250, H - 38), "ДОСТАВЛЕНО ✓" if ok else "промах",
               fill=col, font=f_sm)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="runs/sac_bc/final.zip")
    ap.add_argument("--out", default="policy_run.mp4")
    ap.add_argument("--max-tries", type=int, default=25)
    ap.add_argument("--scenarios", action="store_true")
    ap.add_argument("--object", default=None)
    args = ap.parse_args()

    from stable_baselines3 import SAC
    model = SAC.load(args.policy)
    f_big, f_sm = font(26), font(24)

    for attempt in range(args.max_tries):
        env = SortingPickEnv(seed=100 + attempt, render_mode="rgb_array",
                             inject_scenarios=args.scenarios,
                             pose_mode="tilt", rand_level=1)
        opts = {"object": args.object} if args.object else None
        obs, info = env.reset(options=opts)
        obj, zone = info["object"], info["target_zone"]
        frames, done, ok = [], False, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            ok = info["current_zone"] == zone
            frames.append(caption(env.render(), obj, zone,
                                   info["current_zone"], done, ok, f_big, f_sm))
        env.close()
        print(f"попытка {attempt}: товар={obj} зона={zone} -> "
              f"{'успех' if ok else 'промах'} ({len(frames)} кадров)")
        if ok:
            # добавим стоп-кадр результата
            frames += [frames[-1]] * 20
            w = imageio.get_writer(args.out, fps=20, codec="libx264",
                                   quality=8, macro_block_size=None)
            for fr in frames:
                w.append_data(fr)
            w.close()
            print(f"\nвидео сохранено: {args.out}  "
                  f"({len(frames)} кадров, {len(frames)/20:.1f} с)")
            return
    print("не удалось получить успешный эпизод за отведённые попытки")


if __name__ == "__main__":
    main()
