#!/usr/bin/env python3
"""Сбор демонстраций скриптовым экспертом в SortingPickEnv (для warm-start RL).

Прогоняет ScriptedExpert через env.step, записывает переходы
(obs, action, reward, next_obs, done) успешных эпизодов в .npz и печатает
долю успеха эксперта. Демонстрации нужны для BC-предобучения актёра SAC и
заполнения replay buffer.

Пример:
    python scripts/collect_demos.py --episodes 400 --out runs/demos.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from robozone.rl_env import SortingPickEnv  # noqa: E402
from robozone.expert import ScriptedExpert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--out", default="runs/demos.npz")
    ap.add_argument("--reset-mode", default="accumulator")
    ap.add_argument("--pose-mode", default="tilt")
    ap.add_argument("--rand-level", type=int, default=1)
    ap.add_argument("--inject-scenarios", action="store_true",
                    help="включить сценарии отказов (по умолчанию выкл для чистых демо)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-success", action="store_true", default=True)
    args = ap.parse_args()

    env = SortingPickEnv(reset_mode=args.reset_mode, pose_mode=args.pose_mode,
                         rand_level=args.rand_level,
                         inject_scenarios=args.inject_scenarios, seed=args.seed)
    expert = ScriptedExpert(env)

    obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
    n_success = 0
    per_zone = {"B": [0, 0], "C": [0, 0], "D": [0, 0]}

    for ep in range(args.episodes):
        obs, info = env.reset()
        expert.reset()
        zone = info["target_zone"]
        per_zone[zone][1] += 1
        buf, done, success = [], False, False
        while not done:
            a = expert.act()
            nobs, r, term, trunc, info = env.step(a)
            buf.append((obs, a, r, nobs, float(term)))
            obs = nobs
            done = term or trunc
            if term and info["current_zone"] == zone:
                success = True
        if success:
            n_success += 1
            per_zone[zone][0] += 1
        if success or not args.only_success:
            for o, a, r, no, d in buf:
                obs_l.append(o); act_l.append(a); rew_l.append(r)
                nobs_l.append(no); done_l.append(d)
        if (ep + 1) % 50 == 0:
            print(f"эп {ep+1}/{args.episodes}: успех эксперта "
                  f"{n_success}/{ep+1} ({100*n_success/(ep+1):.0f}%), "
                  f"переходов {len(obs_l)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=np.array(obs_l, np.float32), actions=np.array(act_l, np.float32),
        rewards=np.array(rew_l, np.float32), next_obs=np.array(nobs_l, np.float32),
        dones=np.array(done_l, np.float32))
    print(f"\nсохранено: {args.out}  ({len(obs_l)} переходов)")
    print(f"успех эксперта: {n_success}/{args.episodes} "
          f"({100*n_success/args.episodes:.0f}%)")
    for z in ("B", "C", "D"):
        s, n = per_zone[z]
        if n:
            print(f"  зона {z}: {s}/{n} ({100*s/n:.0f}%)")
    env.close()


if __name__ == "__main__":
    main()
