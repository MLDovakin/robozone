"""Дифференциальная IK (damped least squares) для суставов UR10e.

Используется скриптовой политикой (scripts/demo_pick.py) и может служить
опорным контроллером при обучении RL (residual policy). Решает позицию
и, опционально, ориентацию оси присоски (Z сайта suction_tip вниз).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .sim_core import RobozoneSim, ARM_JOINTS


def _site_jac(sim: RobozoneSim, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    nv = sim.model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(sim.model, sim.data, jacp, jacr, site_id)
    return jacp[:, sim._arm_dofadr], jacr[:, sim._arm_dofadr]


def ik_step(sim: RobozoneSim, target_pos: np.ndarray,
            down: bool = True, damping: float = 0.1,
            max_dq: float = 0.2) -> np.ndarray:
    """Один шаг DLS-IK: приращение суставов к цели.

    target_pos — желаемая позиция кончика присоски (мир);
    down=True   — дополнительно ориентировать ось присоски (+Z сайта) вниз.
    Возвращает dq (6,) с ограничением шага max_dq.
    """
    site = sim._tip_site
    pos = sim.data.site_xpos[site]
    mat = sim.data.site_xmat[site].reshape(3, 3)

    jacp, jacr = _site_jac(sim, site)
    err_p = target_pos - pos

    if down:
        # хотим z-ось сайта = (0,0,-1): ошибка ориентации через векторное
        # произведение текущей и желаемой оси
        z_cur = mat[:, 2]
        z_des = np.array([0.0, 0.0, -1.0])
        err_r = np.cross(z_cur, z_des)
        jac = np.vstack([jacp, jacr])
        err = np.concatenate([err_p, 0.7 * err_r])
    else:
        jac = jacp
        err = err_p

    jjt = jac @ jac.T + (damping ** 2) * np.eye(jac.shape[0])
    dq = jac.T @ np.linalg.solve(jjt, err)
    n = np.abs(dq).max()
    if n > max_dq:
        dq *= max_dq / n
    return dq


def servo_to(sim: RobozoneSim, target_pos: np.ndarray, *,
             down: bool = True, tol: float = 0.01,
             max_time: float = 4.0, suction: float | None = None,
             max_dq: float = 0.08, settle: bool = False) -> bool:
    """Позиционное ведение кончика присоски к точке (для скриптовой политики).

    Каждые 50 мс пересчитывает IK-приращение в текущей конфигурации и
    интегрирует его в командную позу сервоприводов (не в измеренную —
    иначе низкоуровневый контроллер не успевает отработать и цикл буксует).
    Возвращает True, если цель достигнута с точностью tol.

    settle=True: даже при достижении цели продолжает шагать до конца max_time
    (нужно, когда вызывающий цикл ждёт стороннего события, напр. захвата, и
    полагается на продвижение времени — иначе ранний выход замораживает время).
    """
    steps_per_ctrl = max(1, int(0.05 / sim.dt))
    q_cmd = sim.arm_qpos().copy()
    t0 = sim.data.time
    while sim.data.time - t0 < max_time:
        if suction is not None:
            sim.set_suction(suction)
        if not settle and np.linalg.norm(target_pos - sim.ee_pos()) < tol:
            return True
        dq = ik_step(sim, target_pos, down=down, max_dq=max_dq)
        q_cmd = q_cmd + dq
        sim.set_arm_ctrl(q_cmd)
        sim.step(steps_per_ctrl)
    return np.linalg.norm(target_pos - sim.ee_pos()) < tol * 2
