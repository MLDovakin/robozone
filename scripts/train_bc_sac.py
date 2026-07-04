#!/usr/bin/env python3
"""Warm-start RL: behavior cloning актёра SAC на демонстрациях + дообучение SAC.

SAC с нуля застревает в оптимуме «ничего не делать» (координированный
reach→grasp→place слишком редко возникает при случайном exploration). Решение —
warm-start демонстрациями скриптового эксперта:
  1) BC: супервайзно обучаем актёра SAC воспроизводить действия эксперта;
  2) seed: кладём переходы эксперта в replay buffer (данные для критика);
  3) fine-tune: обычный SAC поверх, уже с рабочей начальной политикой.

Требует: pip install stable-baselines3; демо из scripts/collect_demos.py.
Пример:
    python scripts/collect_demos.py --episodes 400 --out runs/demos.npz
    python scripts/train_bc_sac.py --demos runs/demos.npz --timesteps 300000 \
        --bc-epochs 40 --outdir runs/sac_bc
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="runs/demos.npz")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--ent-coef", default="0.02",
                    help="малый фикс. коэф. энтропии для мягкого дообучения")
    ap.add_argument("--init-log-std", type=float, default=-2.0,
                    help="начальный log_std актёра после BC (низкий = мало шума)")
    ap.add_argument("--outdir", default="runs/sac_bc")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        args.ent_coef = float(args.ent_coef)
    except ValueError:
        pass  # допускаем "auto"

    try:
        import torch
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        sys.exit("Установите: pip install stable-baselines3")

    from robozone.rl_env import SortingPickEnv

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    def make(rank):
        return lambda: Monitor(SortingPickEnv(seed=args.seed + rank))
    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecCls([make(i) for i in range(args.n_envs)])

    # ent_coef фиксированно малый: BC-политика точная, сильная стохастичность
    # SAC (высокий log_std + auto-энтропия) её разрушает. Мягкое дообучение.
    model = SAC("MlpPolicy", venv, verbose=1, seed=args.seed,
                buffer_size=500_000, batch_size=256, gamma=0.98, tau=0.02,
                learning_rate=1e-4, train_freq=1, learning_starts=1000,
                ent_coef=args.ent_coef)

    data = np.load(args.demos)
    obs = data["obs"].astype(np.float32)
    act = np.clip(data["actions"].astype(np.float32), -0.999, 0.999)
    rew, nobs, dones = data["rewards"], data["next_obs"], data["dones"]
    print(f"демонстраций: {len(obs)} переходов")

    # -------------------- 1) Behavior Cloning актёра --------------------
    dev = model.device
    obs_t = torch.as_tensor(obs, device=dev)
    act_t = torch.as_tensor(act, device=dev)
    # вес по измерениям действия: вакуум (индекс 6) бинарный — усиливаем, иначе
    # MSE усредняет его к 0 («выкл») и политика никогда не захватывает
    w = torch.ones(7, device=dev); w[6] = 5.0
    opt = torch.optim.Adam(model.actor.parameters(), lr=args.bc_lr)
    n, bs = len(obs), 256
    for ep in range(args.bc_epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = model.actor(obs_t[idx], deterministic=True)  # squashed action
            loss = torch.mean(w * (pred - act_t[idx]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == args.bc_epochs - 1:
            print(f"BC эпоха {ep:3d}: weighted-MSE {tot / n:.4f}")

    # понижаем log_std актёра: BC обучил только среднее, а стохастичность SAC
    # (std~1) разрушает точное управление -> задаём малый разброс
    with torch.no_grad():
        model.actor.log_std.data.fill_(args.init_log_std)
    model.save(str(outdir / "bc_only"))

    # -------------------- 2) seed replay buffer --------------------
    for i in range(n):
        model.replay_buffer.add(obs[i:i + 1], nobs[i:i + 1], act[i:i + 1],
                                np.array([rew[i]], np.float32),
                                np.array([dones[i]], np.float32), [{}])
    print(f"replay buffer заполнен демонстрациями: {model.replay_buffer.size()}")

    # -------------------- 3) SAC fine-tune --------------------
    ckpt = CheckpointCallback(save_freq=max(20000 // args.n_envs, 1),
                              save_path=str(outdir), name_prefix="model")
    model.learn(total_timesteps=args.timesteps, callback=ckpt,
                reset_num_timesteps=False)
    model.save(str(outdir / "final"))
    print(f"модель сохранена: {outdir / 'final'}.zip")


if __name__ == "__main__":
    main()
