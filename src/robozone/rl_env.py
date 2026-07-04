"""Gymnasium-среда обучения RL для задачи «взять из накопителя и направить в зону».

НАБЛЮДЕНИЕ (Box, 46):
    q(6), qdot(6)                — состояние манипулятора
    ee_pos(3)                    — позиция кончика присоски (TCP)
    obj_pos(3), obj_quat(4)      — поза активного товара (с учётом obs-смещения)
    ee_to_obj(3)                 — вектор до товара
    ee_to_drop(3)                — вектор до точки зоны
    held(1), suction(1)          — реальный захват и состояние вакуума
    zone_onehot(3)               — целевая зона [B,C,D] (выход CV-части)
    extents(3)                   — габариты товара (масштаб)
    grasp(3), grasp_to_ee(3)     — размерозависимая точка захвата и вектор до неё
    contact(1)                   — тактильный сигнал: касание присоски (0/1)
    contacted_phase(1)           — был ли уже первый контакт (фаза задачи, монотонно)
    recovery(1)                  — режим восстановления после срыва (closed-loop)
    contact_conf(1)              — гладкая тактильная уверенность [0,1]

ДЕЙСТВИЕ (Box, 7):
    dq(6) in [-1,1]  — приращение целевых позиций суставов (масштаб ACT_SCALE)
    suction in [-1,1] — >0 включает вакуум

НАГРАДА — контактно-ориентированная (см. _reward): до первого касания ведёт
по расстоянию до точки захвата; ПОСЛЕ первого касания расстояние EE–товар
перестаёт влиять (чтобы толчок товара не читался как ошибка), и reward
складывается из бонуса за первый контакт, удержания захвата, штрафа за
проскальзывание/срыв и переноса товара в зону.

REGRASP (closed-loop recovery): при срыве фаза задачи НЕ откатывается в approach
(это ломало бы причинность). Вместо жёсткого сброса — гладкая contact_conf и
флаг recovery_mode: политика остаётся в той же фазе, а reward переключается на
восстановление (слабое сведение к товару + штраф за скорость товара + бонус за
восстановленный захват). Это соответствует reset-free / tactile-regrasp RL.

КОРНЕР-КЕЙСЫ (инъекция отказов, распределение SCENARIOS): смещение товара,
поворот, смещение TCP (калибровка), изменение трения, задержка и ослабление
вакуума — создают неудачные захваты и требуют повторного взятия.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # среда может использоваться и без gymnasium
    gym = None
    spaces = None

from .sim_core import RobozoneSim, ZONE_TARGET, HOME_QPOS, ZONE_AABB

ZONES = ("B", "C", "D")
ACT_SCALE = 0.05      # рад на шаг управления
CTRL_DT = 0.05        # период управления, с (20 Гц)

# Распределение сценариев корнер-кейсов (доли, сумма = 1.0).
# Создают неудачные захваты и ситуации повторного взятия детали.
SCENARIOS = (
    ("normal", 0.20),        # нормальный эпизод
    ("shift", 0.20),         # товар смещён на 2–10 мм от осевшей позы
    ("rotate", 0.15),        # товар слегка повёрнут (доп. наклон/рыскание)
    ("tcp_offset", 0.15),    # захват начинается со смещением TCP (калибровка 2–8 мм)
    ("friction", 0.10),      # изменён коэффициент трения товара
    ("vacuum_delay", 0.10),  # задержка включения вакуума (3–8 шагов)
    ("vacuum_weak", 0.10),   # ослабленный вакуум (сила прилипания 40–70%)
)
_SCEN_NAMES = [s for s, _ in SCENARIOS]
_SCEN_PROBS = np.array([p for _, p in SCENARIOS])


class SortingPickEnv(gym.Env if gym else object):
    """Один эпизод = один товар из накопителя в целевую зону."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, reset_mode: str = "accumulator",
                 object_names: Optional[list[str]] = None,
                 max_steps: int = 300, render_mode: Optional[str] = None,
                 seed: Optional[int] = None,
                 pose_mode: str = "tilt", rand_level: int = 1,
                 curriculum: bool = False,
                 inject_scenarios: bool = True,
                 # коэффициенты контактно-ориентированной награды
                 first_contact_bonus: float = 0.5,
                 stable_contact_bonus: float = 0.01,
                 slip_penalty: float = 2.0,
                 time_penalty: float = 0.01,
                 # closed-loop recovery (regrasp без отката фазы)
                 regrasp_bonus: float = 0.5,
                 realign_weight: float = 0.5,
                 obj_vel_penalty: float = 0.05,
                 drop_penalty: Optional[float] = None):
        if gym is None:
            raise ImportError("Требуется gymnasium: pip install gymnasium")
        super().__init__()
        self.sim = RobozoneSim(seed=seed)
        self.reset_mode = reset_mode
        self.object_names = object_names or sorted(self.sim.objects)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._renderer = None
        self._n_sub = max(1, int(CTRL_DT / self.sim.dt))

        # доменная рандомизация позы/позиции взятия
        self.pose_mode = pose_mode
        self.rand_level = rand_level
        self.curriculum = curriculum
        self.inject_scenarios = inject_scenarios
        self._episodes = 0
        self._recent_success = []

        # коэффициенты награды
        self.first_contact_bonus = first_contact_bonus
        self.stable_contact_bonus = stable_contact_bonus
        # drop_penalty оставлен как синоним slip_penalty для обратной совместимости
        self.slip_penalty = drop_penalty if drop_penalty is not None else slip_penalty
        self.time_penalty = time_penalty
        self.regrasp_bonus = regrasp_bonus
        self.realign_weight = realign_weight
        self.obj_vel_penalty = obj_vel_penalty

        # номиналы физики для восстановления после сценариев
        self._nominal_gain = float(self.sim.model.actuator_gainprm[self.sim._suction_act, 0])
        self._nominal_friction = {
            gid: self.sim.model.geom_friction[gid].copy()
            for o in self.sim.objects.values() for gid in o.geom_ids
        }

        # obs: q6 qd6 ee3 obj3 quat4 (ee-obj)3 (ee-drop)3 held1 suc1 zone3
        #      extents3 grasp3 (grasp-ee)3 contact1 phase1 recovery1 conf1 = 46
        obs_dim = 46
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)

        self._reset_state()

    def _reset_state(self):
        self._target_q = HOME_QPOS.copy()
        self._active = None
        self._zone = None
        self._grasp_pt = np.zeros(3)
        self._obs_bias = np.zeros(3)
        self._vacuum_delay = 0
        self._suction_buf: list[float] = []
        self._suction_cmd = 0.0
        self._scenario = "normal"
        self._steps = 0
        # фаза задачи (монотонна, без отката): контакт был -> манипуляция
        self._contacted = False
        self._was_held = False
        # closed-loop recovery: гладкая тактильная уверенность + флаг режима
        self._contact_conf = 0.0
        self._recovery_mode = False
        self._prev_approach_pot = 0.0
        self._prev_place_pot = 0.0
        self._prev_realign_pot = 0.0
        self._last_obs = np.zeros(self.observation_space.shape, np.float32)

    # ------------------------------------------------------------------ gym
    def _difficulty(self) -> tuple[str, int]:
        if not self.curriculum:
            return self.pose_mode, self.rand_level
        rate = np.mean(self._recent_success[-30:]) if self._recent_success else 0.0
        if rate < 0.4:
            return "upright", 0
        if rate < 0.7:
            return "tilt", 1
        return "tumble", 2

    def _restore_physics(self):
        self.sim.model.actuator_gainprm[self.sim._suction_act, 0] = self._nominal_gain
        for gid, fr in self._nominal_friction.items():
            self.sim.model.geom_friction[gid] = fr

    def _sample_scenario(self, override: Optional[str]) -> str:
        if override is not None:
            return override
        if not self.inject_scenarios:
            return "normal"
        return str(self.sim.rng.choice(_SCEN_NAMES, p=_SCEN_PROBS))

    def _apply_scenario(self):
        """Настраивает корнер-кейс текущего эпизода (после спавна и осадки)."""
        rng = self.sim.rng
        o = self.sim.objects[self._active]
        if self._scenario == "shift":
            d = rng.uniform(0.002, 0.010)          # 2–10 мм
            a = rng.uniform(-np.pi, np.pi)
            q = self.sim.object_qpos(self._active).copy()
            q[0] += d * np.cos(a); q[1] += d * np.sin(a)
            self.sim._set_free_qpos(o, q)
            mujoco.mj_forward(self.sim.model, self.sim.data)
        elif self._scenario == "tcp_offset":
            # смещение TCP/калибровки: товар в наблюдении сдвинут на 2–8 мм
            self._obs_bias = rng.uniform(-1, 1, 3) * np.array([0.008, 0.008, 0.004])
        elif self._scenario == "friction":
            f = rng.uniform(0.4, 1.6)
            for gid in o.geom_ids:
                self.sim.model.geom_friction[gid, 0] = self._nominal_friction[gid][0] * f
        elif self._scenario == "vacuum_delay":
            self._vacuum_delay = int(rng.integers(3, 9))
        elif self._scenario == "vacuum_weak":
            # маргинальная сила прилипания (~30–110 Н): лёгкие товары держатся,
            # тяжёлые срываются при подъёме/переносе -> нужен повторный захват
            self.sim.model.actuator_gainprm[self.sim._suction_act, 0] = \
                self._nominal_gain * rng.uniform(0.06, 0.22)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        self._restore_physics()
        self.sim.reset()
        self._reset_state()

        name = options.get("object") if options else None
        pose_mode, rand_level = self._difficulty()
        self._scenario = self._sample_scenario(
            options.get("scenario") if options else None)
        if self._scenario == "rotate":       # «слегка повёрнут» -> усиленный наклон
            pose_mode = "tumble" if pose_mode == "upright" else pose_mode
        if options:
            pose_mode = options.get("pose_mode", pose_mode)
            rand_level = options.get("rand_level", rand_level)

        name = self.sim.spawn(
            name, mode=self.reset_mode,
            settle_steps=150 if self.reset_mode == "accumulator" else 0,
            pose_mode=pose_mode, rand_level=rand_level)
        if self.reset_mode == "belt":
            for _ in range(600):
                self.sim.step(10)
                if self.sim.object_pos(name)[0] > -0.4 and \
                        np.linalg.norm(self.sim.object_vel(name)) < 0.05:
                    break
        self._active = name
        self._zone = self.sim.objects[name].zone
        self._apply_scenario()

        self._target_q = self.sim.arm_qpos().copy()
        self._grasp_pt = self.sim.object_top(name)
        self._prev_approach_pot = -np.linalg.norm(self.sim.ee_pos() - self._grasp_pt)
        self._prev_place_pot = self._place_potential()
        self._episodes += 1
        self._last_obs = self._obs()
        return self._last_obs, self._info()

    def step(self, action):
        action = np.asarray(action, np.float32)
        dq = action[:6] * ACT_SCALE
        self._target_q = np.clip(self._target_q + dq, -6.28, 6.28)
        self.sim.set_arm_ctrl(self._target_q)

        # задержка включения вакуума (сценарий vacuum_delay)
        raw = 1.0 if action[6] > 0 else 0.0
        self._suction_buf.append(raw)
        if len(self._suction_buf) > self._vacuum_delay:
            effective = self._suction_buf.pop(0)
        else:
            effective = 0.0
        self._suction_cmd = effective
        self.sim.set_suction(effective)

        self.sim.step(self._n_sub)
        self._steps += 1

        # защита от физической нестабильности (редкий blow-up контактов) —
        # завершаем эпизод, чтобы NaN не попал в политику/replay buffer
        if not np.all(np.isfinite(self.sim.data.qpos)):
            info = {"object": self._active, "target_zone": self._zone,
                    "current_zone": None, "held": False,
                    "contacted": self._contacted, "recovery": self._recovery_mode,
                    "scenario": self._scenario, "unstable": True}
            self._recent_success.append(0.0)
            return self._last_obs, -10.0, True, False, info

        if not self.sim.is_held(self._active):
            self._grasp_pt = self.sim.object_top(self._active)

        obs = self._obs()
        self._last_obs = obs
        reward, terminated = self._reward()
        truncated = self._steps >= self.max_steps
        if terminated or truncated:
            self._recent_success.append(
                1.0 if self._info()["current_zone"] == self._zone else 0.0)
        return obs, reward, terminated, truncated, self._info()

    # -------------------------------------------------------------- helpers
    def _place_potential(self) -> float:
        op = self.sim.object_pos(self._active)
        return 2.0 - np.linalg.norm(op - ZONE_TARGET[self._zone])

    def _obs(self) -> np.ndarray:
        s = self.sim
        q, qd = s.arm_qpos(), s.arm_qvel()
        ee = s.ee_pos()
        op = s.object_pos(self._active) + self._obs_bias   # смещение TCP/калибровки
        oq = s.object_qpos(self._active)[3:]
        drop = ZONE_TARGET[self._zone]
        held = 1.0 if s.is_held(self._active) else 0.0
        contact = 1.0 if s.suction_contact(self._active) else 0.0
        suction = float(s.data.ctrl[s._suction_act])
        zone_oh = np.array([1.0 if self._zone == z else 0.0 for z in ZONES])
        extents = s.objects[self._active].extents
        grasp = self._grasp_pt + self._obs_bias
        phase = 1.0 if self._contacted else 0.0      # прогресс задачи (монотонно)
        recovery = 1.0 if self._recovery_mode else 0.0
        return np.concatenate([
            q, qd, ee, op, oq, ee - op, ee - drop,
            [held, suction], zone_oh, extents, grasp, grasp - ee,
            [contact, phase, recovery, self._contact_conf],
        ]).astype(np.float32)

    def _reward(self) -> tuple[float, bool]:
        s = self.sim
        ee, op = s.ee_pos(), s.object_pos(self._active)
        contact = s.suction_contact(self._active)   # тактильный сигнал (касание)
        held = s.is_held(self._active)              # касание + вакуум = захват
        obj_vel = float(np.linalg.norm(s.object_vel(self._active)))
        zone = s.zone_of(op)

        # тактильная уверенность: гладкий рост при контакте, плавный спад без него
        # (без резкого сброса — это заменяет «жёсткий» pseudo-reset фазы)
        if contact:
            self._contact_conf = min(1.0, self._contact_conf + 0.34)
        else:
            self._contact_conf *= 0.8

        r = -self.time_penalty

        # --- ПЕРВЫЙ контакт: переход в фазу манипуляции (МОНОТОННО, без отката) ---
        if contact and not self._contacted:
            self._contacted = True
            r += self.first_contact_bonus            # «нашёл объект»
            self._prev_place_pot = self._place_potential()

        if not self._contacted:
            # ФАЗА ПОДВОДА: единственный сигнал — сближение присоски с точкой захвата
            pot = -np.linalg.norm(ee - self._grasp_pt)
            r += pot - self._prev_approach_pot
            self._prev_approach_pot = pot
        else:
            # ФАЗА МАНИПУЛЯЦИИ (фаза НЕ откатывается в approach — это прогресс задачи)
            # обнаружение срыва: был захват, вакуум включён, товар потерян/ползёт
            if self._was_held and not held and self._suction_cmd > 0.5 \
                    and zone != self._zone:
                r -= self.slip_penalty
                if not self._recovery_mode:          # вход в closed-loop recovery
                    self._recovery_mode = True
                    self._prev_realign_pot = -np.linalg.norm(ee - self._grasp_pt)
            # вход в recovery при «мягкой» потере контакта (уверенность просела),
            # НО только пока вакуум ещё включён — намеренный сброс (укладка) не
            # считается срывом и не запускает recovery
            if self._contacted and not held and self._suction_cmd > 0.5 \
                    and self._contact_conf < 0.3 and not self._recovery_mode:
                self._recovery_mode = True
                self._prev_realign_pot = -np.linalg.norm(ee - self._grasp_pt)

            if held:
                if self._recovery_mode:              # захват восстановлен
                    r += self.regrasp_bonus          # regrasp success reward
                    self._recovery_mode = False
                    self._prev_place_pot = self._place_potential()
                r += self.stable_contact_bonus       # стабильный захват
                place_pot = self._place_potential()  # перенос ТОВАРА (не EE) к зоне
                r += place_pot - self._prev_place_pot
                self._prev_place_pot = place_pot
            elif self._recovery_mode:
                # RECOVERY: та же фаза, closed-loop коррекция (НЕ approach с нуля).
                # слабое сведение присоски к товару + штраф за «убегающий» товар
                realign = -np.linalg.norm(ee - self._grasp_pt)
                r += self.realign_weight * (realign - self._prev_realign_pot)
                self._prev_realign_pot = realign
                r -= self.obj_vel_penalty * obj_vel
        self._was_held = held

        terminated = False
        # B — питающий конвейер (товар едет к инфиду): покой не требуется;
        # C/D — ролл-кейджи: товар должен осесть внутри
        resting = np.linalg.norm(s.object_vel(self._active)) < 0.1
        if zone == self._zone and not held and (zone == "B" or resting):
            r += 10.0
            terminated = True
        elif zone is not None and zone != self._zone and not held:
            r -= 5.0
            terminated = True
        elif op[2] < 0.15 and not held and \
                np.linalg.norm(s.object_vel(self._active)) < 0.05:
            r -= 3.0
            terminated = True
        return float(r), terminated

    def _info(self) -> dict:
        return {
            "object": self._active,
            "target_zone": self._zone,
            "current_zone": self.sim.zone_of(self.sim.object_pos(self._active)),
            "held": self.sim.is_held(self._active),
            "contacted": self._contacted,
            "recovery": self._recovery_mode,
            "scenario": self._scenario,
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import mujoco as mj
        if self._renderer is None:
            self._renderer = mj.Renderer(self.sim.model, 720, 1280)
        self._renderer.update_scene(self.sim.data, camera="cam_global")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(**kwargs):
    """Фабрика для SB3/RLlib."""
    return SortingPickEnv(**kwargs)
