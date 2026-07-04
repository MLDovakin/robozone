#!/usr/bin/env python3
"""Сценарий проверки reward: внезапный срыв детали и требование повторного захвата.

Демонстрирует поведение среды из RL-задачи №2:
- деталь захватывается и поднимается;
- эмулируется ВНЕЗАПНЫЙ срыв захвата (в реальном обучении его порождает
  физика — резкое движение, лёгкая/скользкая деталь), при этом команда
  вакуума остаётся включённой;
- среда фиксирует срыв, выдаёт штраф drop_penalty и НЕ завершает эпизод, пока
  деталь на столе накопителя — политике нужно снова подвести присоску и
  поднять деталь.

Запуск:  PYTHONPATH=src python scripts/scenario_drop_repick.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from robozone.rl_env import SortingPickEnv, ACT_SCALE  # noqa: E402
from robozone.ik import servo_to, ik_step  # noqa: E402


def sim_grab(env, name):
    """Надёжный захват на уровне симуляции (как в скриптовой политике)."""
    s = env.sim
    top = s.object_top(name)
    servo_to(s, top + [0, 0, 0.18], tol=0.02, max_time=4.0)
    s.set_suction(1.0)
    target = top.copy()
    t0 = s.data.time
    while s.data.time - t0 < 3.0 and not s.is_held(name):
        if not s.suction_contact(name):
            target[2] -= 0.006
        servo_to(s, target, tol=0.004, max_time=0.25, suction=1.0, settle=True)
    # синхронизируем состояние среды с симуляцией после захвата на уровне sim
    env._target_q = s.arm_qpos().copy()
    env._was_held = s.is_held(name)
    env._grasp_pt = s.object_top(name)
    env._contacted = s.suction_contact(name)
    env._contact_conf = 1.0 if s.suction_contact(name) else 0.0
    env._recovery_mode = False
    env._prev_place_pot = env._place_potential()
    env._prev_approach_pot = -np.linalg.norm(s.ee_pos() - env._grasp_pt)
    env._prev_realign_pot = -np.linalg.norm(s.ee_pos() - env._grasp_pt)


def ik_action(env, target, suction):
    dq = ik_step(env.sim, target, down=True, max_dq=ACT_SCALE)
    a = np.zeros(7, np.float32)
    a[:6] = np.clip(dq / ACT_SCALE, -1, 1)
    a[6] = 1.0 if suction else -1.0
    return a


def main():
    env = SortingPickEnv(seed=3, pose_mode="upright", rand_level=0,
                         inject_scenarios=False, slip_penalty=3.0)
    _, info = env.reset(options={"object": "lunchbox"})
    name = info["object"]
    s = env.sim
    print(f"товар: {name}, цель: {info['target_zone']}")

    sim_grab(env, name)
    print(f"деталь захвачена (held={s.is_held(name)}), поднимаю")
    for _ in range(20):                       # подъём, вакуум включён
        env.step(ik_action(env, s.ee_pos() + [0, 0, 0.25], suction=True))
    print(f"высота детали: {s.object_pos(name)[2]:.2f} м, held={s.is_held(name)}")

    # --- ВНЕЗАПНЫЙ срыв: команда вакуума ON, но захват физически теряется ---
    print("\n>>> эмуляция внезапного срыва детали (команда вакуума ещё ВКЛ)")
    o = s.objects[name]
    drop_pos = np.array([s.object_pos(name)[0], s.object_pos(name)[1], 0.72])
    s._set_free_qpos(o, np.concatenate([drop_pos, [1, 0, 0, 0]]))
    _, r, term, trunc, info = env.step(ik_action(env, s.ee_pos(), suction=True))
    print(f"reward на шаге срыва: {r:.2f}  (ожидается штраф ≈ -{env.slip_penalty})")
    print(f"эпизод завершён? term={term} (деталь на столе -> продолжаем, нужно поднять)")

    # --- повторный захват после срыва ---
    print("\n>>> повторный захват детали")
    sim_grab(env, name)
    print(f"деталь снова поднята: held={s.is_held(name)} — восстановление после срыва")

    # доводим до зоны, чтобы показать успешное завершение эпизода
    for _ in range(20):
        env.step(ik_action(env, s.ee_pos() + [0, 0, 0.25], suction=True))
    print("сценарий завершён")
    env.close()


if __name__ == "__main__":
    main()
