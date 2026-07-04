#!/usr/bin/env python3
"""AWAC — корректный offline→online RL, который НЕ ломает BC-политику.

Проблема наивного SAC поверх BC: актёр максимизирует Q и уходит в действия «вне
данных» при ещё не откалиброванном критике → политика рушится. AWAC
(Advantage-Weighted Actor-Critic) обновляет актёра как **взвешенную регрессию к
действиям из данных**:
    L_actor = − E_(s,a)~buffer [ log π(a|s) · exp( A(s,a)/λ ) ],   A = Q(s,a) − V(s)
Актёр никогда не предлагает действия далеко от данных (веса лишь усиливают те,
что дали высокое преимущество) → offline-обучение стабильно, а online-дообучение
плавно улучшает политику, не разрушая захват.

Пайплайн: offline (только демонстрации) → online (сбор эпизодов средой со
сценариями отказов + продолжение обновлений). Реализация самостоятельная на
PyTorch (сети под obs=46, act=7), нормировка входа по статистике демонстраций.

Пример:
    python scripts/train_awac.py --demos runs/demos.npz \
        --offline-steps 30000 --online-steps 40000 --out runs/awac
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from robozone.rl_env import SortingPickEnv  # noqa: E402

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
EPS = 1e-6


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, h=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, act_dim)
        self.log_std = nn.Linear(h, act_dim)

    def _dist(self, obs):
        h = self.body(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std.exp()

    def sample(self, obs):
        mu, std = self._dist(obs)
        x = mu + std * torch.randn_like(std)
        a = torch.tanh(x)
        logp = (-0.5 * (((x - mu) / (std + EPS)) ** 2) - std.log()
                - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + EPS).sum(-1)
        return a, logp

    def log_prob(self, obs, a):
        mu, std = self._dist(obs)
        a = torch.clamp(a, -1 + 1e-4, 1 - 1e-4)
        x = torch.atanh(a)
        logp = (-0.5 * (((x - mu) / (std + EPS)) ** 2) - std.log()
                - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + EPS).sum(-1)
        return logp

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        mu, std = self._dist(obs)
        return torch.tanh(mu if deterministic else mu + std * torch.randn_like(std))


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, h=256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, h, 1)
        self.q2 = mlp(obs_dim + act_dim, h, 1)

    def forward(self, obs, a):
        x = torch.cat([obs, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, cap, obs_dim, act_dim):
        self.o = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros((cap, act_dim), np.float32)
        self.r = np.zeros(cap, np.float32)
        self.no = np.zeros((cap, obs_dim), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap, self.idx, self.full = cap, 0, False

    def add(self, o, a, r, no, d):
        i = self.idx
        self.o[i], self.a[i], self.r[i], self.no[i], self.d[i] = o, a, r, no, d
        self.idx = (i + 1) % self.cap
        self.full = self.full or self.idx == 0

    def add_batch(self, o, a, r, no, d):
        for i in range(len(o)):
            self.add(o[i], a[i], r[i], no[i], d[i])

    def __len__(self):
        return self.cap if self.full else self.idx

    def sample(self, n, dev):
        idx = np.random.randint(0, len(self), n)
        t = lambda x: torch.as_tensor(x[idx], device=dev)
        return t(self.o), t(self.a), t(self.r), t(self.no), t(self.d)


class AWAC:
    def __init__(self, obs_dim, act_dim, obs_mean, obs_std, dev,
                 gamma=0.98, tau=0.005, lam=1.0, lr=3e-4, wmax=20.0):
        self.dev, self.gamma, self.tau, self.lam, self.wmax = dev, gamma, tau, lam, wmax
        self.actor = Actor(obs_dim, act_dim).to(dev)
        self.critic = Critic(obs_dim, act_dim).to(dev)
        self.critic_t = Critic(obs_dim, act_dim).to(dev)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.om = torch.as_tensor(obs_mean, device=dev)
        self.os = torch.as_tensor(obs_std, device=dev)

    def norm(self, o):
        return (o - self.om) / self.os

    def update(self, buf, bs=256):
        o, a, r, no, d = buf.sample(bs, self.dev)
        o, no = self.norm(o), self.norm(no)
        # ---- критик: TD к цели Беллмана (без энтропии, AWAC) ----
        with torch.no_grad():
            a2, _ = self.actor.sample(no)
            q1t, q2t = self.critic_t(no, a2)
            y = r + self.gamma * (1 - d) * torch.min(q1t, q2t)
        q1, q2 = self.critic(o, a)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()

        # ---- актёр: advantage-weighted regression к действиям из данных ----
        with torch.no_grad():
            q1d, q2d = self.critic(o, a)
            q_data = torch.min(q1d, q2d)
            api, _ = self.actor.sample(o)
            v1, v2 = self.critic(o, api)
            v = torch.min(v1, v2)
            adv = q_data - v
            w = torch.clamp(torch.exp(adv / self.lam), max=self.wmax)
        logp = self.actor.log_prob(o, a)
        a_loss = -(w * logp).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()

        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return float(c_loss), float(a_loss), float(w.mean())

    @torch.no_grad()
    def act(self, obs_np, deterministic=True):
        o = self.norm(torch.as_tensor(obs_np, device=self.dev, dtype=torch.float32))
        return self.actor.act(o.unsqueeze(0), deterministic)[0].cpu().numpy()


def evaluate(agent, scenarios, N=40, seed=500):
    """Успех политики; при scenario!=None форсирует конкретный сценарий отказа."""
    from collections import defaultdict
    res = {}
    for scen in scenarios:
        env = SortingPickEnv(seed=seed, inject_scenarios=(scen is None),
                             pose_mode="tilt", rand_level=1)
        s = 0
        for e in range(N):
            opts = None if scen is None else {"scenario": scen}
            o, info = env.reset(options=opts)
            z = info["target_zone"]; done = False
            while not done:
                a = agent.act(o, deterministic=True)
                o, r, term, trunc, info = env.step(a); done = term or trunc
            s += info["current_zone"] == z
        env.close()
        res[scen if scen else "mixed"] = 100.0 * s / N
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="runs/demos.npz")
    ap.add_argument("--offline-steps", type=int, default=30000)
    ap.add_argument("--online-steps", type=int, default=40000)
    ap.add_argument("--updates-per-step", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="runs/awac")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.demos)
    obs = d["obs"].astype(np.float32)
    obs_mean = obs.mean(0); obs_std = obs.std(0) + 1e-3
    obs_dim, act_dim = obs.shape[1], d["actions"].shape[1]

    buf = ReplayBuffer(600_000, obs_dim, act_dim)
    buf.add_batch(obs, d["actions"].astype(np.float32), d["rewards"],
                  d["next_obs"].astype(np.float32), d["dones"])
    print(f"демонстраций в буфере: {len(buf)}")

    agent = AWAC(obs_dim, act_dim, obs_mean, obs_std, dev, lam=args.lam)

    # ---- Инициализация актёра: BC на демонстрациях ----
    print("=== BC ИНИЦИАЛИЗАЦИЯ актёра ===")
    o_bc = torch.as_tensor(obs, device=dev, dtype=torch.float32)
    a_bc = torch.as_tensor(d["actions"], device=dev, dtype=torch.float32)
    o_bc = agent.norm(o_bc)
    w_bc = torch.ones(act_dim, device=dev); w_bc[6] = 5.0  # вес для вакуума
    for bc_step in range(5000):
        mu, std = agent.actor._dist(o_bc)
        bc_loss = (w_bc * (torch.tanh(mu) - a_bc) ** 2).mean()
        agent.a_opt.zero_grad(); bc_loss.backward(); agent.a_opt.step()
        if (bc_step + 1) % 1000 == 0:
            print(f"  BC {bc_step+1}: loss {bc_loss:.4f}")
    print("BC инициализация завершена")

    # -------------------- OFFLINE --------------------
    print("=== OFFLINE (только демонстрации) ===")
    for step in range(args.offline_steps):
        c, a, w = agent.update(buf)
        if (step + 1) % 5000 == 0:
            print(f"  offline {step+1}: c_loss {c:.3f} a_loss {a:.3f} w {w:.2f}")
    torch.save({"actor": agent.actor.state_dict(), "om": obs_mean, "os": obs_std},
               outdir / "offline.pt")
    r0 = evaluate(agent, [None], N=40)
    print(f"  offline eval (mixed): {r0['mixed']:.0f}%")

    # -------------------- ONLINE (со сценариями отказов) --------------------
    print("=== ONLINE (среда + сценарии отказов) ===")
    env = SortingPickEnv(seed=args.seed + 7, inject_scenarios=True,
                         pose_mode="tilt", rand_level=1)
    o, info = env.reset()
    ep_ret, ep, done_cnt = 0.0, 0, 0
    for step in range(args.online_steps):
        a = agent.act(o, deterministic=False)
        no, r, term, trunc, info = env.step(a)
        buf.add(o, a, r, no, float(term))
        o = no; ep_ret += r
        for _ in range(args.updates_per_step):
            agent.update(buf)
        if term or trunc:
            ep += 1; done_cnt += int(info["current_zone"] == info["target_zone"])
            o, info = env.reset(); ep_ret = 0.0
            if ep % 25 == 0:
                print(f"  online step {step+1}: эпизодов {ep}, "
                      f"успех последних {done_cnt}/{ep}")
    env.close()
    torch.save({"actor": agent.actor.state_dict(), "om": obs_mean, "os": obs_std},
               outdir / "final.pt")
    print(f"сохранено: {outdir/'final.pt'}")

    # -------------------- ОТЧЁТ ПО СЦЕНАРИЯМ --------------------
    print("\n=== УСПЕХ ПО СЦЕНАРИЯМ (deterministic, 40 эп.) ===")
    scen = [None, "normal", "shift", "rotate", "tcp_offset",
            "friction", "vacuum_delay", "vacuum_weak"]
    res = evaluate(agent, scen, N=40)
    for k in scen:
        name = k if k else "mixed"
        print(f"  {name:14s}: {res[name if k else 'mixed']:4.0f}%")


if __name__ == "__main__":
    main()
