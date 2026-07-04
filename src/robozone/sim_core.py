"""Ядро симуляции ПАК: обвязка MuJoCo-модели sim/scene.xml.

Отвечает за:
- «бесконечные» конвейерные ленты: тело ленты на слайд-шарнире движется
  velocity-актуатором, а его координата периодически сматывается (wrap) —
  поверхность однородна, поэтому контакты этого не замечают, а товары
  переносятся честным трением;
- спавн/парковку тестовых объектов (телепорт свободных тел);
- вакуумный захват (adhesion-актуатор) и проверку прилипания;
- детекцию попадания товара в зоны B/C/D;
- домашнюю позу манипулятора.

Этот модуль не зависит ни от gymnasium, ни от ROS2 — его используют
и RL-среда (rl_env.py), и ROS2-узел (ros2_ws/.../sim_node.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import SIM_DIR, MESHES_DIR

SCENE_XML = SIM_DIR / "scene.xml"

ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
ARM_ACTUATORS = (
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3",
)
HOME_QPOS = np.array([-1.5708, -1.8, 1.6, -1.3708, -1.5708, 0.0])

# Скорость подающего конвейера по постановке задачи
BELT_A_SPEED = 1.0   # м/с
BELT_B_SPEED = 0.7   # м/с (лента основного сортировщика)
BELT_WRAP = 0.02     # м, шаг смотки слайда ленты

# Зоны успеха (AABB, метры): товар считается доставленным,
# когда его центр внутри объёма. Для B — попадание на питающую ленту
# сортировщика (лента далее переносит товар к инфиду).
ZONE_AABB = {
    "B": ((0.45, 1.00, 0.66), (1.05, 3.40, 1.15)),
    "C": ((-0.18, -1.70, 0.05), (0.64, -0.42, 0.80)),
    "D": ((0.66, -1.70, 0.05), (1.48, -0.42, 0.80)),
}

# Стратегия и точки укладки товара по зонам.
# B — «place»: товар опускается нижней гранью на верхний край гравитационного
#     лотка (DROP_POINT["B"] — целевая позиция НИЖНЕЙ грани) и соскальзывает
#     на ленту основного сортировщика. Тип-таргет = точка + захватное смещение.
# C/D — «drop»: открытые ролл-кейджи, товар сбрасывается сверху; DROP_POINT —
#     абсолютная цель кончика присоски (высоко над кейджем), товар падает внутрь.
DROP_POINT = {
    "B": np.array([0.75, 1.15, 0.73]),
    "C": np.array([0.30, -0.70, 1.15]),
    "D": np.array([0.95, -0.70, 1.15]),
}
DROP_STRATEGY = {"B": "place", "C": "drop", "D": "drop"}

# Ориентировочная точка покоя товара в зоне (для reward-shaping и визуализации)
ZONE_TARGET = {
    "B": np.array([0.75, 1.90, 0.80]),
    "C": np.array([0.30, -1.15, 0.30]),
    "D": np.array([0.95, -1.15, 0.30]),
}


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Произведение кватернионов (w,x,y,z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def placement_tip_target(zone: str, hold_dz: float) -> np.ndarray:
    """Целевая позиция кончика присоски для укладки товара в зону."""
    if DROP_STRATEGY[zone] == "place":
        return DROP_POINT[zone] + np.array([0.0, 0.0, hold_dz])
    return DROP_POINT[zone].copy()

# Зона захвата на накопителе
PICK_CENTER = np.array([0.15, 0.0, 0.70])
ACC_TOP_Z = 0.68


@dataclass
class ObjectInfo:
    name: str
    body_id: int
    qpos_adr: int
    geom_ids: list
    zone: str            # ground-truth зона из categories.json
    category: str
    extents: np.ndarray
    park_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))


class RobozoneSim:
    """Обёртка над MuJoCo-моделью сортировочной ячейки."""

    def __init__(self, scene_xml: str | Path = SCENE_XML, seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        m = self.model
        self._arm_qadr = np.array([
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS
        ])
        self._arm_dofadr = np.array([
            m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS
        ])
        self._arm_act = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            for a in ARM_ACTUATORS
        ])
        self._suction_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "suction")
        self._belt_a_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "belt_a_drive")
        self._belt_b_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "belt_b_drive")
        self._belt_a_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "belt_a_joint")]
        self._belt_b_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "belt_b_joint")]
        self._tip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "suction_tip")
        self._pad_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "suction_pad")
        self._base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")

        self.objects: dict[str, ObjectInfo] = {}
        cats = json.loads((MESHES_DIR / "categories.json").read_text())
        siminfo = json.loads((MESHES_DIR / "sim_objects.json").read_text())
        for name, c in cats.items():
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{name}")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{name}_joint")
            geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid]
            park = np.array([*siminfo[name]["park"], 1, 0, 0, 0], dtype=float)
            self.objects[name] = ObjectInfo(
                name=name, body_id=bid, qpos_adr=m.jnt_qposadr[jid],
                geom_ids=geoms, zone=c["zone"], category=c["category"],
                extents=np.array(siminfo[name]["extents_m"]), park_pose=park,
            )

        self.active_object: str | None = None
        self.reset()

    # ------------------------------------------------------------------ core
    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    def reset(self, belt_speed: float = BELT_A_SPEED) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for o in self.objects.values():
            self._set_free_qpos(o, o.park_pose)
        self.data.qpos[self._arm_qadr] = HOME_QPOS
        self.data.ctrl[self._arm_act] = HOME_QPOS
        self.set_belt_speed(belt_speed, BELT_B_SPEED)
        self.set_suction(0.0)
        self.active_object = None
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        """n шагов физики с обслуживанием смотки лент."""
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._wrap_belt(self._belt_a_qadr)
            self._wrap_belt(self._belt_b_qadr)

    def _wrap_belt(self, qadr: int) -> None:
        q = self.data.qpos[qadr]
        if q > BELT_WRAP:
            self.data.qpos[qadr] = q - BELT_WRAP
        elif q < -BELT_WRAP:
            self.data.qpos[qadr] = q + BELT_WRAP

    # ------------------------------------------------------------- actuation
    def set_belt_speed(self, v_a: float, v_b: float | None = None) -> None:
        self.data.ctrl[self._belt_a_act] = v_a
        if v_b is not None:
            self.data.ctrl[self._belt_b_act] = v_b

    def set_arm_ctrl(self, q_target: np.ndarray) -> None:
        """Целевые позиции суставов (позиционные сервоприводы)."""
        self.data.ctrl[self._arm_act] = q_target

    def set_suction(self, on: float) -> None:
        self.data.ctrl[self._suction_act] = float(np.clip(on, 0.0, 1.0))

    # ------------------------------------------------------------------ state
    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self._arm_qadr].copy()

    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self._arm_dofadr].copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_site].copy()

    def ee_mat(self) -> np.ndarray:
        return self.data.site_xmat[self._tip_site].reshape(3, 3).copy()

    def object_pos(self, name: str | None = None) -> np.ndarray:
        o = self.objects[name or self.active_object]
        return self.data.xpos[o.body_id].copy()

    def object_qpos(self, name: str | None = None) -> np.ndarray:
        o = self.objects[name or self.active_object]
        return self.data.qpos[o.qpos_adr:o.qpos_adr + 7].copy()

    def object_vel(self, name: str | None = None) -> np.ndarray:
        o = self.objects[name or self.active_object]
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{o.name}_joint")
        dadr = self.model.jnt_dofadr[jid]
        return self.data.qvel[dadr:dadr + 3].copy()

    def object_top(self, name: str | None = None) -> np.ndarray:
        """Точка на верхней грани товара в текущей позе (лучом сверху вниз).

        Надёжнее, чем оценка по габаритам меша: для товара, лежащего в
        произвольной ориентации (напр. ручка плашмя), возвращает реальную
        высоту верхней поверхности в точке над его центром масс.
        """
        o = self.objects[name or self.active_object]
        c = self.data.xpos[o.body_id]
        # луч видит только группу 1 (товары) -> не мешают ни рука, ни стол
        geomgroup = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        pnt = np.array([c[0], c[1], c[2] + 1.0])
        vec = np.array([0.0, 0.0, -1.0])
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.model, self.data, pnt, vec,
                             geomgroup, 1, -1, gid)
        if gid[0] in o.geom_ids and dist >= 0:
            return pnt + vec * dist
        # запасной вариант — по радиусу ограничивающей сферы геома
        rb = self.model.geom_rbound[o.geom_ids[0]]
        return np.array([c[0], c[1], c[2] + rb])

    def suction_contact(self, name: str | None = None) -> bool:
        """Есть ли контакт присоски с активным объектом (включая margin)."""
        o = self.objects[name or self.active_object]
        gset = set(o.geom_ids)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == self._pad_geom and c.geom2 in gset) or \
               (c.geom2 == self._pad_geom and c.geom1 in gset):
                return True
        return False

    def is_held(self, name: str | None = None) -> bool:
        """Объект считается схваченным: вакуум включён и есть контакт."""
        return self.data.ctrl[self._suction_act] > 0.5 and self.suction_contact(name)

    def zone_of(self, pos: np.ndarray) -> str | None:
        for z, (lo, hi) in ZONE_AABB.items():
            if all(lo[k] <= pos[k] <= hi[k] for k in range(3)):
                return z
        return None

    # ------------------------------------------------------------------ spawn
    def _set_free_qpos(self, o: ObjectInfo, qpos7: np.ndarray) -> None:
        self.data.qpos[o.qpos_adr:o.qpos_adr + 7] = qpos7
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{o.name}_joint")
        dadr = self.model.jnt_dofadr[jid]
        self.data.qvel[dadr:dadr + 6] = 0.0

    def park(self, name: str) -> None:
        o = self.objects[name]
        self._set_free_qpos(o, o.park_pose)
        if self.active_object == name:
            self.active_object = None

    def _random_quat(self, pose_mode: str) -> np.ndarray:
        """Кватернион (w,x,y,z) для заданного режима случайной ориентации.

        "upright" — только рыскание (товар стоит как положили);
        "tumble"  — полностью случайная 3D-ориентация: товар после осадки
                    ложится на произвольную грань (разные позы для обучения);
        "tilt"    — рыскание + небольшой наклон (умеренная рандомизация).
        """
        if pose_mode == "tumble":
            u1, u2, u3 = self.rng.uniform(0, 1, 3)
            return np.array([
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
            ])[[3, 0, 1, 2]]  # -> (w,x,y,z)
        yaw = self.rng.uniform(-np.pi, np.pi)
        if pose_mode == "tilt":
            axis = self.rng.normal(size=3); axis[2] = 0
            axis = axis / (np.linalg.norm(axis) + 1e-9)
            tilt = self.rng.uniform(0, 0.5)   # до ~29°
            qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            qt = np.array([np.cos(tilt / 2), *(np.sin(tilt / 2) * axis)])
            return _quat_mul(qz, qt)
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    def _accumulator_xy(self, rand_level: int) -> np.ndarray:
        """Позиция товара на накопителе через (расстояние, азимут) от базы руки.

        Так политика учится захватывать при разных вылетах и углах манипулятора.
        rand_level повышает разброс дистанции и угла.
        """
        base = self.data.xpos[self._base_body_id][:2]
        if rand_level <= 0:
            r_lo, r_hi, ang = 0.62, 0.72, np.deg2rad(8)
        elif rand_level == 1:
            r_lo, r_hi, ang = 0.55, 0.82, np.deg2rad(20)
        else:
            r_lo, r_hi, ang = 0.50, 0.90, np.deg2rad(32)
        # опорное направление от базы к центру зоны захвата
        d0 = PICK_CENTER[:2] - base
        bearing0 = np.arctan2(d0[1], d0[0])
        r = self.rng.uniform(r_lo, r_hi)
        bearing = bearing0 + self.rng.uniform(-ang, ang)
        xy = base + r * np.array([np.cos(bearing), np.sin(bearing)])
        # держим в пределах стола накопителя (|x|,|y| <= 0.55)
        xy[0] = np.clip(xy[0], -0.45, 0.48)
        xy[1] = np.clip(xy[1], -0.45, 0.45)
        return xy

    def spawn(self, name: str | None = None, mode: str = "accumulator",
              settle_steps: int = 200, pose_mode: str = "upright",
              rand_level: int = 1) -> str:
        """Телепорт объекта в сцену с доменной рандомизацией.

        mode="belt":        на начало подающего конвейера, поедет со скоростью ленты;
        mode="accumulator": сразу в зону захвата (короткие RL-эпизоды).
        pose_mode: "upright" | "tilt" | "tumble" — разброс ориентации (см. _random_quat).
        rand_level (0..2): разброс позиции взятия по дистанции и углу от руки.
        """
        if self.active_object:
            self.park(self.active_object)
        if name is None:
            name = self.rng.choice(sorted(self.objects))
        o = self.objects[name]

        quat = self._random_quat("upright" if mode == "belt" else pose_mode)
        if mode == "belt":
            pos = np.array([-7.3, self.rng.uniform(-0.1, 0.1), 0.70 + 0.005])
        else:
            xy = self._accumulator_xy(rand_level)
            if pose_mode == "tumble":
                # приподнимаем, чтобы товар свободно упал и лёг на случайную грань
                z = ACC_TOP_Z + o.extents.max() / 2 + 0.12
            elif pose_mode == "tilt":
                z = ACC_TOP_Z + o.extents.min() / 2 + 0.03
            else:  # upright — кладём нижней гранью на стол (origin меша = низ)
                z = ACC_TOP_Z + 0.01
            pos = np.array([xy[0], xy[1], z])
        self._set_free_qpos(o, np.concatenate([pos, quat]))
        if mode == "belt":
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{name}_joint")
            self.data.qvel[self.model.jnt_dofadr[jid]] = BELT_A_SPEED
        mujoco.mj_forward(self.model, self.data)
        if settle_steps and mode == "accumulator":
            self.step(settle_steps)
        self.active_object = name
        return name
