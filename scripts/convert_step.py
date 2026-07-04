#!/usr/bin/env python3
"""STEP -> STL конвертация тестовых объектов для MuJoCo.

Пайплайн: cascadio (OpenCASCADE) STEP -> GLB, trimesh GLB -> STL.
- перевод единиц мм -> м
- центрирование по XY, дно на z=0 (удобно спавнить на ленту)
- децимация тяжёлых мешей до лимита граней (визуальный и коллизионный меш един,
  MuJoCo сам строит выпуклую оболочку для коллизий)

Запуск:
    python scripts/convert_step.py [--faces 8000]
"""
import argparse
import json
import sys
import tempfile
import unicodedata
from pathlib import Path

import cascadio
import trimesh

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "models_files"
DST_DIR = ROOT / "assets" / "meshes" / "objects"

# Транслитерация имён файлов: MuJoCo/ROS дружелюбнее к ascii-именам.
NAME_MAP = {
    "Бутылка": "bottle",
    "Короб 300х200х200": "box_300x200x200",
    "Короб 400х400х300": "box_400x400x300",
    "ЛанчБокс": "lunchbox",
    "Мешок": "bag",
    "Моющее средство": "detergent",
    "Пуфик": "pouf",
    "Ручка": "pen",
    "Тарелка": "plate",
    "Цилиндр": "cylinder",
    "Шлем": "helmet",
}


def convert_one(stp_path: Path, out_stl: Path, max_faces: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        glb = Path(td) / "tmp.glb"
        cascadio.step_to_glb(str(stp_path), str(glb), tol_linear=0.5, tol_angular=0.4)
        mesh = trimesh.load(str(glb), force="mesh")

    # cascadio уже переводит мм STEP -> метры в GLB, доп. масштаб не нужен

    # Центр по XY в ноль, дно на z=0
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])

    if len(mesh.faces) > max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)

    mesh.export(str(out_stl))
    ext = (mesh.bounds[1] - mesh.bounds[0])
    return {
        "faces": int(len(mesh.faces)),
        "extents_m": [round(float(v), 4) for v in ext],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", type=int, default=8000, help="лимит граней после децимации")
    args = ap.parse_args()

    DST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for stp in sorted(SRC_DIR.glob("*.stp")):
        stem = unicodedata.normalize("NFC", stp.stem)
        name = NAME_MAP.get(stem)
        if name is None:
            print(f"!! нет транслитерации для '{stem}', пропуск", file=sys.stderr)
            continue
        out = DST_DIR / f"{name}.stl"
        info = convert_one(stp, out, args.faces)
        manifest[name] = {"source": stp.name, **info}
        print(f"{stp.name:35s} -> {out.name:25s} faces={info['faces']:6d} "
              f"extents={info['extents_m']}")

    (DST_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nманифест: {DST_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
