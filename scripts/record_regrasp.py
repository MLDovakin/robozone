#!/usr/bin/env python3
"""Запись видео сценария regrasp от лица манипулятора (камера следует за захватом).

Показывает closed-loop recovery: подвод → захват → подъём → ВНЕЗАПНЫЙ срыв →
восстановление (recovery) → повторный захват → укладка в зону. Камера
закреплена «на запястье» (следует за кончиком присоски, POV манипулятора).
Внизу кадра — статус: фаза, held/contact, recovery, contact_conf, reward.

Запуск:  PYTHONPATH=src python scripts/record_regrasp.py --out demo_pick.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

# шрифт с поддержкой кириллицы (иначе PIL рисует «квадраты»)
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from robozone.rl_env import SortingPickEnv  # noqa: E402
from robozone.sim_core import DROP_POINT, placement_tip_target, HOME_QPOS  # noqa: E402
from robozone.ik import servo_to, ik_step  # noqa: E402

W, H, FPS = 1080, 720, 30


class Recorder:
    def __init__(self, env, out, azimuth=120.0, elevation=-32.0, dist=0.85):
        self.env = env
        self.sim = env.sim
        self.r = mujoco.Renderer(self.sim.model, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, dist
        self.writer = imageio.get_writer(out, fps=FPS, codec="libx264",
                                         quality=8, macro_block_size=None)
        self.f_title = _font(24)
        self.f_label = _font(26)
        self.f_flag = _font(21)
        self.f_event = _font(24)
        self.label = ""
        self.event = ""
        self.event_ttl = 0

    def set_stage(self, label, event=None):
        self.label = label
        if event:
            self.event, self.event_ttl = event, FPS  # держим событие ~1 c

    def _status(self):
        s, e = self.sim, self.env
        held = s.is_held(e._active)
        contact = s.suction_contact(e._active)
        return (f"фаза: {self.label}", [
            ("контакт", contact), ("захват", held),
            ("recovery", e._recovery_mode)],
            e._contact_conf)

    def frame(self, n=1, suction=None):
        for _ in range(n):
            # камера следит за кончиком присоски (POV манипулятора); смещение к
            # товару ограничено, чтобы захват оставался в кадре, даже когда товар
            # уехал по ленте B
            ee = self.sim.ee_pos()
            look = ee + np.clip(self.sim.object_pos(self.env._active) - ee,
                                -0.3, 0.3)
            self.cam.lookat[:] = look
            self.r.update_scene(self.sim.data, self.cam)
            img = Image.fromarray(self.r.render())
            self._overlay(img, suction)
            self.writer.append_data(np.asarray(img))
            if self.event_ttl > 0:
                self.event_ttl -= 1

    def _overlay(self, img, suction):
        d = ImageDraw.Draw(img, "RGBA")
        title, flags, conf = self._status()
        # нижняя плашка
        d.rectangle([0, H - 104, W, H], fill=(15, 20, 35, 210))
        d.text((24, H - 94), title, fill=(255, 255, 255), font=self.f_label)
        x = 24
        for name, on in flags:
            col = (90, 220, 120) if on else (120, 130, 150)
            d.text((x, H - 52), f"● {name}", fill=col, font=self.f_flag)
            x += 170
        d.text((x, H - 52), f"contact_conf={conf:0.2f}",
               fill=(200, 210, 230), font=self.f_flag)
        if suction is not None:
            d.text((W - 230, H - 94),
                   "вакуум: ВКЛ" if suction else "вакуум: выкл",
                   fill=(255, 180, 60) if suction else (150, 160, 180),
                   font=self.f_flag)
        # верхний заголовок
        d.rectangle([0, 0, W, 44], fill=(15, 20, 35, 190))
        d.text((24, 10), "RoboZone · POV манипулятора · сценарий REGRASP",
               fill=(220, 230, 245), font=self.f_title)
        # всплывающее событие (штраф/бонус)
        if self.event_ttl > 0 and self.event:
            w = int(self.f_event.getbbox(self.event)[2]) + 40
            d.rectangle([W // 2 - w // 2, 54, W // 2 + w // 2, 96],
                        fill=(20, 25, 45, 225))
            d.text((W // 2 - w // 2 + 20, 62), self.event,
                   fill=(255, 210, 90), font=self.f_event)

    def close(self):
        self.writer.close()
        self.r.close()


def sync(env):
    """Синхронизировать состояние среды с sim после захвата на уровне sim."""
    s, n = env.sim, env._active
    env._target_q = s.arm_qpos().copy()
    env._was_held = s.is_held(n)
    env._grasp_pt = s.object_top(n)
    env._contacted = env._contacted or s.suction_contact(n)
    env._prev_place_pot = env._place_potential()
    env._prev_approach_pot = -np.linalg.norm(s.ee_pos() - env._grasp_pt)
    env._prev_realign_pot = env._prev_approach_pot


def tick(env, suction):
    """Один шаг учёта награды/состояния (без команды рукой) для подписей."""
    s, n = env.sim, env._active
    env._suction_cmd = 1.0 if suction else 0.0
    if not s.is_held(n):
        env._grasp_pt = s.object_top(n)
    return env._reward()


def move(rec, target, suction, label, chunks=8, event=None, track=True):
    """Плавное перемещение кончика к target с записью кадров.

    track=True — обновляет награду/состояние (фазы захвата/recovery);
    track=False — состояние заморожено (для укладки, когда recovery уже завершён).
    """
    env, s = rec.env, rec.env.sim
    rec.set_stage(label, event)
    for _ in range(chunks):
        s.set_suction(1.0 if suction else 0.0)
        servo_to(s, target, tol=0.01, max_time=0.18, suction=1.0 if suction else 0.0,
                 settle=True)
        if track:
            r, _ = tick(env, suction)
            _flash_reward(rec, r)
        rec.frame(2, suction=suction)


def _flash_reward(rec, r):
    if r >= 0.4:
        rec.set_stage(rec.label, f"+bonus  reward={r:+.2f}")
    elif r <= -1.0:
        rec.set_stage(rec.label, f"ШТРАФ  reward={r:+.2f}")


def descend_grasp(rec, label):
    env, s = rec.env, rec.env.sim
    n = env._active
    rec.set_stage(label)
    s.set_suction(1.0)
    t0 = s.data.time
    while s.data.time - t0 < 2.5 and not s.is_held(n):
        tgt = s.object_top(n) + [0, 0, -0.006]
        if s.suction_contact(n):
            tgt = s.object_top(n)
        servo_to(s, tgt, tol=0.004, max_time=0.16, suction=1.0, settle=True)
        r, _ = tick(env, True)
        if r >= 0.4:
            rec.set_stage(label, f"+первый контакт  reward={r:+.2f}")
        rec.frame(2, suction=True)
    sync(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_pick.mp4")
    ap.add_argument("--object", default="lunchbox")
    args = ap.parse_args()

    env = SortingPickEnv(seed=3, inject_scenarios=False,
                         pose_mode="upright", rand_level=0)
    env.reset(options={"object": args.object})
    s, name = env.sim, env._active
    # подсветить товар (оранжевый) для наглядности на видео
    for gid in s.objects[name].geom_ids:
        s.model.geom_rgba[gid] = [1.0, 0.45, 0.05, 1.0]
        s.model.geom_matid[gid] = -1

    rec = Recorder(env, args.out)
    top = s.object_top(name)

    # 1) подвод
    move(rec, top + [0, 0, 0.16], False, "подвод к товару")
    # 2) захват (первый контакт -> бонус)
    descend_grasp(rec, "захват (вакуум ВКЛ)")
    # 3) подъём
    move(rec, s.ee_pos() + [0, 0, 0.22], True, "подъём", chunks=6)

    # 4) ВНЕЗАПНЫЙ срыв (вакуум остаётся ВКЛ) -> телепорт товара на стол
    rec.set_stage("ВНЕЗАПНЫЙ срыв детали")
    xy = s.object_pos(name)[:2]
    s._set_free_qpos(s.objects[name], np.array([xy[0], xy[1], 0.72, 1, 0, 0, 0]))
    mujoco.mj_forward(s.model, s.data)
    r, _ = tick(env, True)               # штраф за срыв + вход в recovery
    rec.set_stage("recovery: восстановление захвата", f"ШТРАФ срыв  reward={r:+.2f}")
    rec.frame(int(0.8 * FPS), suction=True)

    # 5) recovery: повторный подвод и захват (та же фаза, без reset)
    move(rec, s.object_top(name) + [0, 0, 0.15], True, "recovery: повторный подвод",
         chunks=6)
    descend_grasp(rec, "recovery: повторный захват")
    r, _ = tick(env, True)               # regrasp success bonus
    rec.set_stage("захват восстановлен", f"+regrasp  reward={r:+.2f}")
    rec.frame(int(0.7 * FPS), suction=True)

    # 6) укладка в целевую зону (recovery завершён -> состояние заморожено)
    env._recovery_mode = False
    hold_dz = float(s.ee_pos()[2] - s.object_pos(name)[2])
    move(rec, s.ee_pos() + [0, 0, 0.35], True, "подъём", chunks=6, track=False)
    zone = env._zone
    above = np.array([DROP_POINT[zone][0], DROP_POINT[zone][1], 1.38])
    move(rec, above, True, f"перенос в зону {zone}", chunks=10, track=False)
    move(rec, placement_tip_target(zone, hold_dz), True, f"укладка в зону {zone}",
         chunks=6, track=False)
    rec.set_stage(f"готово: товар в зоне {zone}")
    s.set_suction(0.0)
    rec.frame(int(1.2 * FPS), suction=False)

    rec.close()
    print(f"видео сохранено: {args.out}")


if __name__ == "__main__":
    main()
