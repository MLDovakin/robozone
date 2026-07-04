#!/usr/bin/env python3
"""Сравнение политик (BC vs AWAC) на сценариях отказов.

Оценивает обе политики на одних и тех же сценариях: normal, shift, rotate,
tcp_offset, friction, vacuum_delay, vacuum_weak, и mixed (все вместе).
Выводит таблицу успехов + анализ, какие сценарии сложные для каждой.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from robozone.rl_env import SortingPickEnv

SCENARIOS = [None, "normal", "shift", "rotate", "tcp_offset",
             "friction", "vacuum_delay", "vacuum_weak"]
SCENARIO_NAMES = {
    None: "mixed (все вместе)",
    "normal": "normal (без возмущений)",
    "shift": "shift (смещение 2–10 мм)",
    "rotate": "rotate (неправильный угол)",
    "tcp_offset": "tcp_offset (смещение TCP 2–8 мм)",
    "friction": "friction (коэф. трения ×[0.4,1.6])",
    "vacuum_delay": "vacuum_delay (задержка 3–8 шагов)",
    "vacuum_weak": "vacuum_weak (слабый захват 30–110 Н)",
}


def load_policy(policy_path):
    """Загружает политику: либо SB3 SAC, либо AWAC PyTorch."""
    if policy_path.endswith(".zip"):
        from stable_baselines3 import SAC
        model = SAC.load(policy_path)
        return lambda obs: model.predict(obs, deterministic=True)[0]
    elif policy_path.endswith(".pt"):
        import torch
        ckpt = torch.load(policy_path, map_location="cpu")

        class Actor(torch.nn.Module):
            def __init__(self, obs_dim, act_dim):
                super().__init__()
                self.body = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 256), torch.nn.ReLU(),
                    torch.nn.Linear(256, 256), torch.nn.ReLU())
                self.mu = torch.nn.Linear(256, act_dim)
                self.log_std = torch.nn.Linear(256, act_dim)
            def forward(self, obs):
                h = self.body(obs)
                return torch.tanh(self.mu(h))

        actor = Actor(46, 7)
        actor.load_state_dict(ckpt["actor"])
        om = torch.from_numpy(ckpt["om"]).float()
        os = torch.from_numpy(ckpt["os"]).float()

        def policy_fn(obs):
            o = torch.from_numpy(obs).float().unsqueeze(0)
            o = (o - om) / os
            with torch.no_grad():
                a = actor(o)[0].numpy()
            return a

        return policy_fn
    else:
        raise ValueError(f"unknown format: {policy_path}")


def evaluate_policy(policy, policy_name, scenarios, N=50, seed=777):
    """Оценивает политику на каждом сценарии."""
    print(f"\n{'='*70}")
    print(f"Оценка политики: {policy_name}")
    print(f"{'='*70}")

    results = {}
    for scen in scenarios:
        env = SortingPickEnv(seed=seed, inject_scenarios=(scen is None),
                             pose_mode="tilt", rand_level=1)
        successes = 0
        per_zone = defaultdict(lambda: [0, 0])  # [success, total]

        for ep in range(N):
            opts = None if scen is None else {"scenario": scen}
            obs, info = env.reset(options=opts)
            target = info["target_zone"]
            done = False

            while not done:
                action = policy(obs)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc

            success = info["current_zone"] == target
            successes += success
            per_zone[target][1] += 1
            if success:
                per_zone[target][0] += 1

        env.close()
        success_rate = 100.0 * successes / N
        results[scen if scen else "mixed"] = {
            "rate": success_rate,
            "per_zone": dict(per_zone)
        }

        scen_name = SCENARIO_NAMES.get(scen, str(scen))
        print(f"  {scen_name:45s} {success_rate:5.1f}% ({successes}/{N})")
        for zone in ["B", "C", "D"]:
            if zone in per_zone:
                s, t = per_zone[zone]
                print(f"      {zone}: {s}/{t} ({100*s/t:.0f}%)")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-policy", default="runs/sac_bc/final.zip")
    ap.add_argument("--awac-policy", default="runs/awac/final.pt")
    ap.add_argument("--n-episodes", type=int, default=50)
    args = ap.parse_args()

    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║       AWAC (offline→online) vs BC warm-start: сравнение политик      ║")
    print("║                     Перенос обучения на сложные сценарии              ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")

    # Загружаем обе политики
    print("\n[1] Загружаю BC политику...", end="")
    bc_policy = load_policy(args.bc_policy)
    print(" OK")

    print("[2] Загружаю AWAC политику...", end="")
    if Path(args.awac_policy).exists():
        awac_policy = load_policy(args.awac_policy)
        print(" OK")
    else:
        print(" NOT FOUND (пропускаю, ждём обучения)")
        awac_policy = None

    # Оцениваем BC
    bc_res = evaluate_policy(bc_policy, "BC warm-start (baseline)",
                             SCENARIOS, N=args.n_episodes)

    # Оцениваем AWAC если доступна
    awac_res = None
    if awac_policy:
        awac_res = evaluate_policy(awac_policy, "AWAC (offline→online)",
                                   SCENARIOS, N=args.n_episodes)

        # Сравнение
        print(f"\n{'='*70}")
        print("СРАВНЕНИЕ: AWAC vs BC (разница в процентах, + = AWAC лучше)")
        print(f"{'='*70}")
        for scen in SCENARIOS:
            key = scen if scen else "mixed"
            bc_r = bc_res[key]["rate"]
            awac_r = awac_res[key]["rate"]
            diff = awac_r - bc_r
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {SCENARIO_NAMES.get(scen, str(scen)):45s} "
                  f"BC {bc_r:5.1f}% → AWAC {awac_r:5.1f}% {arrow} ({diff:+.1f}%)")

        # Вывод итогов
        bc_avg = np.mean([bc_res[k]["rate"] for k in bc_res])
        awac_avg = np.mean([awac_res[k]["rate"] for k in awac_res])
        print(f"\n  Средний успех: BC {bc_avg:.1f}% → AWAC {awac_avg:.1f}% "
              f"({awac_avg - bc_avg:+.1f}%)")

    # Анализ слабых мест
    print(f"\n{'='*70}")
    print("АНАЛИЗ СЛОЖНЫХ СЦЕНАРИЕВ")
    print(f"{'='*70}")

    def analyze(res, name):
        worst = min((s, k) for k, v in res.items() if k != "mixed"
                    for s in [v["rate"]])
        best = max((s, k) for k, v in res.items() if k != "mixed"
                   for s in [v["rate"]])
        print(f"\n{name}:")
        print(f"  Самый сложный: {worst[1]} ({worst[0]:.0f}%)")
        print(f"  Самый простой: {best[1]} ({best[0]:.0f}%)")

    analyze(bc_res, "BC warm-start")
    if awac_res:
        analyze(awac_res, "AWAC")


if __name__ == "__main__":
    main()
