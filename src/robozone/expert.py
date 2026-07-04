"""Скриптовый эксперт для сбора демонстраций в SortingPickEnv (warm-start RL).

Выдаёт действие в формате среды (6 приращений суставов в [-1,1] + вакуум) на
основе дифференциальной IK и конечного автомата фаз (подвод → захват → подъём →
перенос → укладка → отпускание). Работает поверх env.step, поэтому его действия
по формату и динамике идентичны обучаемой политике. Демонстрации эксперта
используются для behavior cloning актёра SAC и заполнения replay buffer.
"""
from __future__ import annotations

import numpy as np

from .rl_env import ACT_SCALE
from .sim_core import DROP_POINT, placement_tip_target
from .ik import ik_step


class ScriptedExpert:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.hold_dz = 0.12
        self._t_phase = 0

    def _action(self, target, suction, max_dq=ACT_SCALE):
        dq = ik_step(self.env.sim, target, down=True, max_dq=max_dq)
        a = np.zeros(7, np.float32)
        a[:6] = np.clip(dq / ACT_SCALE, -1.0, 1.0)
        a[6] = 1.0 if suction else -1.0
        return a

    def _to(self, phase):
        self.phase = phase
        self._t_phase = 0

    def act(self) -> np.ndarray:
        s = self.env.sim
        name = self.env._active
        zone = self.env._zone
        ee = s.ee_pos()
        held = s.is_held(name)
        self._t_phase += 1

        if self.phase == "approach":
            hover = s.object_top(name) + [0, 0, 0.16]
            if np.linalg.norm(ee - hover) < 0.03 or self._t_phase > 60:
                self._to("descend")
            return self._action(hover, suction=False)

        if self.phase == "descend":
            top = s.object_top(name)
            target = top if s.suction_contact(name) else top + [0, 0, -0.01]
            if held:
                self.hold_dz = float(ee[2] - s.object_pos(name)[2])
                self._to("lift")
            elif self._t_phase > 70:
                self._to("approach")
            return self._action(target, suction=True, max_dq=0.03)

        if self.phase == "lift":
            if not held:
                self._to("approach")
            elif ee[2] > 1.22 or self._t_phase > 40:
                self._to("transfer")
            return self._action([ee[0], ee[1], 1.30], suction=True)

        if self.phase == "transfer":
            above = np.array([DROP_POINT[zone][0], DROP_POINT[zone][1], 1.36])
            if not held:
                self._to("approach")
            elif np.linalg.norm(ee - above) < 0.05 or self._t_phase > 70:
                self._to("place")
            return self._action(above, suction=True)

        if self.phase == "place":
            tip = placement_tip_target(zone, self.hold_dz)
            if not held:
                self._to("approach")
            elif np.linalg.norm(ee - tip) < 0.04 or self._t_phase > 60:
                self._to("release")
            return self._action(tip, suction=True, max_dq=0.04)

        return self._action(ee, suction=False)      # release
