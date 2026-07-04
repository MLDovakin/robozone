#!/usr/bin/env python3
"""Сквозная демонстрация контура ПАК со скриптовой политикой.

Цикл по объектам: спавн на ленту -> транспортировка в накопитель ->
классификация (ground truth из categories.json = выход CV-части) ->
IK-захват присоской -> перенос в зону B/C/D -> сброс. Пишет видео.

Запуск:
    python scripts/demo_pick.py [--objects bottle box_300x200x200 ...]
                                [--mode belt|accumulator] [--video out.mp4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mujoco  # noqa: E402
from robozone.sim_core import (  # noqa: E402
    RobozoneSim, DROP_POINT, HOME_QPOS, placement_tip_target,
)
from robozone.ik import servo_to  # noqa: E402


class VideoDump:
    def __init__(self, sim, path, cam="cam_global", fps=30, size=(720, 1280)):
        import imageio
        self.sim, self.cam, self.fps = sim, cam, fps
        self.renderer = mujoco.Renderer(sim.model, *size)
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                         quality=7) if path else None
        self.next_t = 0.0

    def tick(self):
        if self.writer and self.sim.data.time >= self.next_t:
            self.renderer.update_scene(self.sim.data, camera=self.cam)
            self.writer.append_data(self.renderer.render())
            self.next_t += 1.0 / self.fps

    def close(self):
        if self.writer:
            self.writer.close()


def run_steps(sim, video, seconds):
    n = int(seconds / sim.dt)
    chunk = max(1, int(1 / 30 / sim.dt))
    for _ in range(0, n, chunk):
        sim.step(chunk)
        video.tick()


def pick_and_route(sim: RobozoneSim, video: VideoDump, name: str, mode: str) -> bool:
    obj = sim.objects[name]
    print(f"\n== {name}: категория «{obj.category}» -> зона {obj.zone}")

    # скриптовый базовый прогон: плотный спавн (rand_level=0) для наглядности;
    # RL-среда обучается на широкой рандомизации (см. rl_env.py)
    sim.spawn(name, mode=mode, settle_steps=0, pose_mode="upright", rand_level=0)
    if mode == "belt":
        # ждём, пока товар доедет и остановится в накопителе
        t0 = sim.data.time
        while sim.data.time - t0 < 12.0:
            run_steps(sim, video, 0.1)
            pos, vel = sim.object_pos(name), sim.object_vel(name)
            if pos[0] > -0.6 and np.linalg.norm(vel) < 0.05:
                break
        else:
            print("   !! товар не доехал до накопителя")
            return False
    else:
        run_steps(sim, video, 0.6)

    # точка присасывания — реальная верхняя грань товара в текущей позе
    top = sim.object_top(name)
    hover = top + [0, 0, 0.20]
    ok = servo_to(sim, hover, tol=0.02, max_time=4.0)
    video.tick(); run_steps(sim, video, 0.2)
    # спуск с включённым вакуумом: опускаемся, пока присоска не окажется в
    # зоне захвата (proximity-контакт), затем держим позицию — вакуум сам
    # подтягивает товар. settle=True гарантирует продвижение времени.
    sim.set_suction(1.0)
    target = top.copy()
    t0 = sim.data.time
    while sim.data.time - t0 < 3.0 and not sim.is_held(name):
        if not sim.suction_contact(name):
            target[2] -= 0.006          # опускаемся до зоны захвата
        servo_to(sim, target, tol=0.004, max_time=0.25, suction=1.0, settle=True)
        video.tick()
    if not sim.is_held(name):
        print("   !! не удалось присосать объект")
        sim.set_suction(0.0)
        return False
    # смещение захвата: насколько кончик присоски выше нижней грани товара
    hold_dz = float(sim.ee_pos()[2] - sim.object_pos(name)[2])
    print(f"   захват есть (смещение {hold_dz:.3f} м), поднимаю")

    # Путь укладки: подъём -> перенос высоко над зоной -> вертикальный спуск.
    # Такая траектория избегает застреваний жадной IK при качании крупного
    # товара низко над препятствиями (накопитель, кейджи).
    ee = sim.ee_pos()
    servo_to(sim, np.array([ee[0], ee[1], 1.38]), tol=0.03, max_time=3.0, suction=1.0)
    run_steps(sim, video, 0.2)

    tip_target = placement_tip_target(obj.zone, hold_dz)
    above = np.array([tip_target[0], tip_target[1], max(1.38, tip_target[2] + 0.15)])
    servo_to(sim, above, tol=0.05, max_time=4.0, suction=1.0)
    video.tick()
    # крупные/тяжёлые товары медленнее — даём больше времени на укладку
    place_time = 7.0 if obj.extents.max() > 0.35 else 4.5
    ok = servo_to(sim, tip_target, tol=0.03, max_time=place_time, suction=1.0)
    run_steps(sim, video, 0.2)

    sim.set_suction(0.0)
    run_steps(sim, video, 6.0 if obj.zone == "B" else 2.5)

    final = sim.object_pos(name)
    got = sim.zone_of(final)
    print(f"   финальная позиция {np.round(final, 2)} -> зона {got} "
          f"({'OK' if got == obj.zone else 'MISS'})")

    # домой
    sim.set_arm_ctrl(HOME_QPOS)
    run_steps(sim, video, 1.0)
    sim.park(name)
    return got == obj.zone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--mode", default="belt", choices=["belt", "accumulator"])
    ap.add_argument("--video", default="demo_pick.mp4")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    sim = RobozoneSim(seed=args.seed)
    names = args.objects or sorted(sim.objects)
    video = VideoDump(sim, args.video)

    results = {}
    for n in names:
        results[n] = pick_and_route(sim, video, n, args.mode)
    video.close()

    print("\n===== итог =====")
    for n, ok in results.items():
        print(f"  {n:22s} {'OK' if ok else 'FAIL'}")
    print(f"успешно: {sum(results.values())}/{len(results)}")
    if args.video:
        print(f"видео: {args.video}")


if __name__ == "__main__":
    main()
