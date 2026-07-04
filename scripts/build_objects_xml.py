#!/usr/bin/env python3
"""Генерация sim/objects.xml из сконвертированных мешей.

Для каждого тестового объекта создаётся свободное тело (freejoint) с мешевым
геомом. Инерция задаётся явно (аппроксимация параллелепипедом по AABB) —
это устойчиво к незамкнутым shell-мешам, на которых автоинерция MuJoCo
падает или даёт мусор.

Между эпизодами объекты «паркуются» на стеллаже за пределами рабочей зоны
(x = -11 м); спавн выполняется телепортом qpos из кода симуляции.

Запуск: python scripts/build_objects_xml.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
MESHES = ROOT / "assets" / "meshes" / "objects"
OUT = ROOT / "sim" / "objects.xml"

# Правдоподобные массы товаров, кг
MASS = {
    "bag": 1.2, "bottle": 0.8, "box_300x200x200": 2.0, "box_400x400x300": 4.0,
    "cylinder": 1.5, "detergent": 1.0, "helmet": 1.1, "lunchbox": 0.5,
    "pen": 0.02, "plate": 0.4, "pouf": 3.0,
}

RGBA = {
    "bag": "0.55 0.45 0.35 1", "bottle": "0.3 0.6 0.35 1",
    "box_300x200x200": "0.72 0.55 0.35 1", "box_400x400x300": "0.75 0.6 0.4 1",
    "cylinder": "0.6 0.6 0.65 1", "detergent": "0.85 0.4 0.15 1",
    "helmet": "0.85 0.15 0.15 1", "lunchbox": "0.3 0.5 0.8 1",
    "pen": "0.2 0.2 0.7 1", "plate": "0.9 0.9 0.9 1", "pouf": "0.5 0.3 0.55 1",
}

PARK_X = -11.0
PARK_Z = 0.36
PARK_Y0, PARK_DY = -2.5, 0.5


def main() -> int:
    names = sorted(MASS)
    lines = ['<mujoco model="robozone_objects">', "  <asset>"]
    for n in names:
        lines.append(f'    <mesh name="{n}" file="meshes/objects/{n}.stl"/>')
    lines += ["  </asset>", "", "  <worldbody>"]

    info = {}
    for i, n in enumerate(names):
        mesh = trimesh.load(str(MESHES / f"{n}.stl"), force="mesh")
        lo, hi = mesh.bounds
        ext = hi - lo
        m = MASS[n]
        com = (lo + hi) / 2.0
        ixx = m / 12.0 * (ext[1] ** 2 + ext[2] ** 2)
        iyy = m / 12.0 * (ext[0] ** 2 + ext[2] ** 2)
        izz = m / 12.0 * (ext[0] ** 2 + ext[1] ** 2)
        py = PARK_Y0 + i * PARK_DY
        lines += [
            f'    <body name="obj_{n}" pos="{PARK_X} {py:.2f} {PARK_Z}">',
            f'      <freejoint name="obj_{n}_joint"/>',
            f'      <inertial pos="{com[0]:.4f} {com[1]:.4f} {com[2]:.4f}" '
            f'mass="{m}" diaginertia="{ixx:.6e} {iyy:.6e} {izz:.6e}"/>',
            f'      <geom class="object" mesh="{n}" rgba="{RGBA[n]}"/>',
            "    </body>",
        ]
        info[n] = {"extents_m": [round(float(v), 4) for v in ext], "mass_kg": m,
                   "park": [PARK_X, round(py, 2), PARK_Z]}

    lines += ["  </worldbody>", "</mujoco>", ""]
    OUT.write_text("\n".join(lines))
    (MESHES / "sim_objects.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2))
    print(f"-> {OUT}  ({len(names)} объектов)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
