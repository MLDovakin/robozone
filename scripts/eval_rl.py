#!/usr/bin/env python3
"""Оценка политики в среде SortingPickEnv.

Прогоняет N эпизодов и считает долю успешной маршрутизации по зонам.
Работает и без обученной модели (--policy random) — для проверки контура.

Примеры:
    python scripts/eval_rl.py --policy random --episodes 20
    python scripts/eval_rl.py --policy runs/sac/final.zip --algo sac --video eval.mp4
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from robozone.rl_env import SortingPickEnv  # noqa: E402


def load_policy(spec, algo):
    if spec == "random":
        return None
    from stable_baselines3 import SAC, PPO
    cls = {"sac": SAC, "ppo": PPO}[algo]
    return cls.load(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="random")
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--episodes", type=int, default=22)
    ap.add_argument("--reset-mode", default="accumulator")
    ap.add_argument("--video", default="")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    render = bool(args.video)
    env = SortingPickEnv(reset_mode=args.reset_mode,
                         render_mode="rgb_array" if render else None,
                         seed=args.seed)
    model = load_policy(args.policy, args.algo)

    writer = None
    if render:
        import imageio
        writer = imageio.get_writer(args.video, fps=20)

    per_zone = defaultdict(lambda: [0, 0])
    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        zone = info["target_zone"]
        done = False
        ep_ret = 0.0
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            if writer is not None:
                writer.append_data(env.render())
            done = term or trunc
        ok = info["current_zone"] == zone and not info["held"]
        per_zone[zone][1] += 1
        per_zone[zone][0] += int(ok)
        successes += int(ok)
        print(f"эп {ep:2d} [{info['object']:16s}] цель={zone} "
              f"итог={info['current_zone']} R={ep_ret:6.1f} "
              f"{'OK' if ok else '--'}")

    if writer is not None:
        writer.close()
        print(f"видео: {args.video}")

    print("\n== по зонам ==")
    for z in ("B", "C", "D"):
        s, n = per_zone[z]
        if n:
            print(f"  {z}: {s}/{n} ({100*s/n:.0f}%)")
    print(f"итого: {successes}/{args.episodes} "
          f"({100*successes/args.episodes:.0f}%)")
    env.close()


if __name__ == "__main__":
    main()
