#!/usr/bin/env python3
"""Финальный отчёт AWAC: сравнение BC vs AWAC на сложных сценариях.

Оценивает обе политики (BC и AWAC) на 8 сценариях (normal, shift, rotate, etc.)
с детализацией по зонам (B/C/D), выводит таблицу, разницу, анализ.
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from robozone.rl_env import SortingPickEnv

SCENARIOS = [None, "normal", "shift", "rotate", "tcp_offset",
             "friction", "vacuum_delay", "vacuum_weak"]

SCENARIO_NAMES = {
    None: "mixed\t(все сценарии вместе)",
    "normal": "normal\t(без возмущений)",
    "shift": "shift\t(смещение 2–10 мм)",
    "rotate": "rotate\t(неправильный угол)",
    "tcp_offset": "tcp_offset\t(смещение TCP 2–8 мм)",
    "friction": "friction\t(трение ×[0.4,1.6])",
    "vacuum_delay": "vacuum_delay\t(задержка 3–8 шагов)",
    "vacuum_weak": "vacuum_weak\t(слабый 30–110 Н)",
}


def load_policy(policy_path):
    """Загружает политику (SB3 SAC .zip или AWAC PyTorch .pt)."""
    if policy_path.endswith(".zip"):
        from stable_baselines3 import SAC
        model = SAC.load(policy_path)
        return lambda obs: model.predict(obs, deterministic=True)[0]
    elif policy_path.endswith(".pt"):
        import torch
        ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)

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


def eval_policy(policy, scenarios, N=50, seed=888):
    """Оценивает политику на каждом сценарии."""
    results = {}
    for scen in scenarios:
        env = SortingPickEnv(seed=seed, inject_scenarios=(scen is None),
                             pose_mode="tilt", rand_level=1)
        successes = 0
        per_zone = defaultdict(lambda: [0, 0])

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
        results[scen if scen else "mixed"] = {
            "rate": 100.0 * successes / N,
            "per_zone": dict(per_zone)
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-policy", default="runs/sac_bc/final.zip")
    ap.add_argument("--awac-policy", default="runs/awac_v2/final.pt")
    ap.add_argument("--n-episodes", type=int, default=50)
    args = ap.parse_args()

    print("\n" + "=" * 90)
    print(" " * 20 + "AWAC (offline→online) vs BC warm-start")
    print(" " * 20 + "Сравнение на сложных сценариях отказов")
    print("=" * 90 + "\n")

    # Загружаем политики
    print("[1] Загружу BC политику...", end=" ", flush=True)
    bc_policy = load_policy(args.bc_policy)
    print("✓")

    print("[2] Загружу AWAC политику...", end=" ", flush=True)
    if Path(args.awac_policy).exists():
        awac_policy = load_policy(args.awac_policy)
        print("✓")
    else:
        print("NOT FOUND")
        return

    # Оцениваем
    print(f"[3] Оцениваю BC на {len(SCENARIOS)} сценариях ({args.n_episodes} эп. каждый)...")
    bc_res = eval_policy(bc_policy, SCENARIOS, N=args.n_episodes)

    print(f"[4] Оцениваю AWAC на {len(SCENARIOS)} сценариях ({args.n_episodes} эп. каждый)...")
    awac_res = eval_policy(awac_policy, SCENARIOS, N=args.n_episodes)

    # Таблица результатов
    print("\n" + "=" * 90)
    print("ИТОГОВАЯ ТАБЛИЦА (success rate %)")
    print("=" * 90)
    print(f"{'Сценарий':<30} {'BC':>8} {'AWAC':>8} {'Разница':>8} {'B':>8} {'C':>8} {'D':>8}")
    print("-" * 90)

    for scen in SCENARIOS:
        key = scen if scen else "mixed"
        name = SCENARIO_NAMES.get(scen, str(scen))[:30]
        bc_r = bc_res[key]["rate"]
        awac_r = awac_res[key]["rate"]
        diff = awac_r - bc_r

        # Per-zone AWAC
        zones_str = ""
        for z in ["B", "C", "D"]:
            if z in awac_res[key]["per_zone"]:
                s, t = awac_res[key]["per_zone"][z]
                zones_str += f"{100*s/t:>7.0f} "
            else:
                zones_str += "       - "

        print(f"{name:<30} {bc_r:>7.1f}% {awac_r:>7.1f}% {diff:>+7.1f}% {zones_str}")

    # Статистика
    bc_avg = np.mean([bc_res[k]["rate"] for k in bc_res])
    awac_avg = np.mean([awac_res[k]["rate"] for k in awac_res])
    improvement = awac_avg - bc_avg

    print("=" * 90)
    print(f"Средний успех:    BC {bc_avg:.1f}%  →  AWAC {awac_avg:.1f}%  ({improvement:+.1f}%)")
    print()

    # Анализ сложности
    print("=" * 90)
    print("АНАЛИЗ: Какие сценарии сложные / простые для AWAC")
    print("=" * 90)

    awac_by_diff = sorted(
        [(k, v["rate"]) for k, v in awac_res.items() if k != "mixed"],
        key=lambda x: x[1], reverse=True
    )

    print("\nЛучшие сценарии для AWAC (где преуспевает):")
    for scen, rate in awac_by_diff[:3]:
        bc_rate = bc_res[scen]["rate"]
        diff = rate - bc_rate
        print(f"  • {SCENARIO_NAMES.get(scen if scen else None, scen)[:40]:<40} "
              f"{rate:.0f}%  (vs BC {bc_rate:.0f}%, {diff:+.0f}%)")

    print("\nСложные сценарии для AWAC (где отстаёт):")
    for scen, rate in awac_by_diff[-3:]:
        bc_rate = bc_res[scen]["rate"]
        diff = rate - bc_rate
        status = "↓ хуже BC" if diff < -5 else "≈ как BC" if abs(diff) <= 5 else "↑ лучше BC"
        print(f"  • {SCENARIO_NAMES.get(scen if scen else None, scen)[:40]:<40} "
              f"{rate:.0f}%  (vs BC {bc_rate:.0f}%, {diff:+.0f}%) {status}")

    # Выводы
    print("\n" + "=" * 90)
    print("ВЫВОДЫ")
    print("=" * 90)

    if improvement > 5:
        print(f"✓ AWAC УЛУЧШИЛ политику на {improvement:.1f}% в среднем.")
        print("  Advantage-weighted регрессия + online AWAC успешно работает на")
        print("  сценариях отказов. Политика изучила recovery и адаптацию.")
    elif improvement > -5:
        print(f"~ AWAC примерно эквивалентен BC ({improvement:.1f}%).")
        print("  Offline-обучение слегка нарушало BC, но online восстановил.")
    else:
        print(f"✗ AWAC ДЕГРАДИРОВАЛ на {abs(improvement):.1f}% относительно BC.")
        print("  Критик не был достаточно откалиброван или λ был неправильный.")

    # Рекомендация
    best = "AWAC" if awac_avg > bc_avg else "BC"
    print(f"\nРекомендация для production: {best}")


if __name__ == "__main__":
    main()
