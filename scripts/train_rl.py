#!/usr/bin/env python3
"""Обучение RL-политики захвата и маршрутизации (Stable-Baselines3 SAC).

Требует: pip install stable-baselines3
Пример:
    python scripts/train_rl.py --timesteps 300000 --n-envs 4
    python scripts/train_rl.py --algo ppo --timesteps 1000000

Чекпойнты и логи -> runs/. Оценка обученной политики — scripts/eval_rl.py.

Замечание по масштабу: полноценное обучение манипуляции требует миллионов
шагов и GPU. Скрипт задаёт корректный, воспроизводимый пайплайн; для защиты
достаточно короткого прогона, показывающего рост reward и работу контура.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--reset-mode", default="accumulator",
                    choices=["accumulator", "belt"])
    ap.add_argument("--outdir", default="runs/sac")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action="store_true",
                    help="прогресс-бар (нужны tqdm+rich)")
    args = ap.parse_args()

    try:
        from stable_baselines3 import SAC, PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        sys.exit("Установите stable-baselines3: pip install stable-baselines3")

    from robozone.rl_env import SortingPickEnv

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def make(rank):
        def _f():
            return Monitor(SortingPickEnv(reset_mode=args.reset_mode,
                                          seed=args.seed + rank))
        return _f

    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecCls([make(i) for i in range(args.n_envs)])

    ckpt = CheckpointCallback(save_freq=max(10000 // args.n_envs, 1),
                              save_path=str(outdir), name_prefix="model")

    # tensorboard — по желанию (если установлен)
    try:
        import tensorboard  # noqa: F401
        tb = str(outdir / "tb")
    except ImportError:
        tb = None

    if args.algo == "sac":
        model = SAC("MlpPolicy", venv, verbose=1, seed=args.seed,
                    buffer_size=300_000, batch_size=256, gamma=0.98,
                    tau=0.02, learning_rate=3e-4, train_freq=1,
                    tensorboard_log=tb)
    else:
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed,
                    n_steps=2048, batch_size=256, gamma=0.99,
                    tensorboard_log=tb)

    model.learn(total_timesteps=args.timesteps, callback=ckpt,
                progress_bar=args.progress)
    model.save(str(outdir / "final"))
    print(f"модель сохранена: {outdir / 'final'}.zip")


if __name__ == "__main__":
    main()
