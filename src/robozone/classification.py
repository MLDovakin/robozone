"""Классификация товара по правилам Трека 3.

Правила (см. документацию, раздел «Правила классификации»):
1. Габариты: товар «проходит», если его габариты больше 10x10x10 мм
   и меньше 450x320x320 мм. Не проходит -> категория C
   («Не подходит для сортировки по габаритам»). Проверка габаритов имеет
   приоритет над проверкой формы.
2. Форма: если в каком-либо сечении объекта отношение радиуса вписанной
   окружности к радиусу описанной r/R > K = 0.8 -> категория D
   («Не подходит для сортировки без доупаковки»).
3. Иначе -> категория B («Подходит для сортировки»).

Габариты берутся по минимальному ориентированному параллелепипеду (OBB) —
это соответствует «просвету» сортировщика независимо от позы товара на ленте.
Сечения проверяются вдоль трёх главных осей OBB на нескольких станциях.

CLI: python -m robozone.classification  — печатает таблицу по всем тестовым
объектам и пишет assets/meshes/objects/categories.json (используется сценой
и RL-средой как ground truth для маршрутизации).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon
from shapely.ops import polylabel

from . import MESHES_DIR

# Ограничения основного сортировщика, метры (отсортированы по убыванию)
MAX_DIMS_SORTED = np.array([0.450, 0.320, 0.320])
MIN_DIM = 0.010
K_ROUND = 0.8

# Станции сечений вдоль каждой главной оси (доли длины).
# Критерий — «круг в ЛЮБОМ сечении», поэтому сетка плотная и захватывает
# приторцевые зоны (round-фичи вроде штока пневмоцилиндра или горловины
# флакона находятся у краёв объекта).
SECTION_STATIONS = tuple(round(0.05 + 0.075 * i, 3) for i in range(13))

CATEGORY_ZONE = {"sortable": "B", "oversize": "C", "round": "D"}
CATEGORY_LABEL_RU = {
    "sortable": "Подходит для сортировки",
    "oversize": "Не подходит для сортировки по габаритам",
    "round": "Не подходит для сортировки без доупаковки",
}


@dataclass
class ClassResult:
    name: str
    category: str          # sortable | oversize | round
    zone: str              # B | C | D
    obb_extents_m: list    # габариты OBB по убыванию
    k_round: float         # максимальное r/R по всем сечениям
    label_ru: str


def _section_roundness(mesh: trimesh.Trimesh) -> float:
    """Максимум r_впис/R_опис по сечениям вдоль трёх главных осей OBB.

    Контур сечения формализуется как выпуклая оболочка точек сечения:
    поведение товара в сортировщике (способность катиться) определяется
    внешней огибающей, а метод устойчив к незамкнутым shell-мешам.
    """
    # Переводим меш в систему координат OBB, чтобы оси сечений совпали
    # с главными осями объекта.
    to_obb, _ = trimesh.bounds.oriented_bounds(mesh)
    m = mesh.copy()
    m.apply_transform(to_obb)
    lo, hi = m.bounds

    best = 0.0
    for axis in range(3):
        normal = np.zeros(3)
        normal[axis] = 1.0
        span = hi[axis] - lo[axis]
        if span < 1e-6:
            continue
        in_plane = [i for i in range(3) if i != axis]
        for frac in SECTION_STATIONS:
            origin = np.zeros(3)
            origin[axis] = lo[axis] + frac * span
            segments = trimesh.intersections.mesh_plane(
                m, plane_normal=normal, plane_origin=origin
            )
            if len(segments) < 3:
                continue
            pts2d = np.asarray(segments).reshape(-1, 3)[:, in_plane]
            hull = shapely.MultiPoint(pts2d).convex_hull
            if hull.geom_type != "Polygon" or hull.area < 1e-8:
                continue
            poly = Polygon(hull.exterior)
            r_out = shapely.minimum_bounding_radius(poly)
            if r_out < 1e-6:
                continue
            center = polylabel(poly, tolerance=r_out * 0.01)
            r_in = poly.exterior.distance(center)
            best = max(best, r_in / r_out)
    return best


def classify_mesh(mesh: trimesh.Trimesh, name: str = "") -> ClassResult:
    """Отнесение товара к категории по мешу (метры)."""
    _, obb_extents = trimesh.bounds.oriented_bounds(mesh)
    dims = np.sort(np.asarray(obb_extents))[::-1]

    fits_max = bool(np.all(dims < MAX_DIMS_SORTED))
    fits_min = bool(np.all(dims > MIN_DIM))

    k = _section_roundness(mesh)

    # Порядок принятия решения из документации: сначала габариты, потом форма
    if not (fits_max and fits_min):
        cat = "oversize"
    elif k > K_ROUND:
        cat = "round"
    else:
        cat = "sortable"

    return ClassResult(
        name=name,
        category=cat,
        zone=CATEGORY_ZONE[cat],
        obb_extents_m=[round(float(v), 4) for v in dims],
        k_round=round(float(k), 3),
        label_ru=CATEGORY_LABEL_RU[cat],
    )


def classify_stl(path: Path) -> ClassResult:
    mesh = trimesh.load(str(path), force="mesh")
    return classify_mesh(mesh, name=path.stem)


def main() -> None:
    results = {}
    for stl in sorted(MESHES_DIR.glob("*.stl")):
        r = classify_stl(stl)
        results[r.name] = asdict(r)
        print(f"{r.name:22s} zone={r.zone}  K={r.k_round:5.3f}  "
              f"obb={r.obb_extents_m}  {r.label_ru}")
    out = MESHES_DIR / "categories.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
