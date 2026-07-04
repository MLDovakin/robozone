"""Локатор корня проекта и подключение пакета robozone (src/).

Пакет robozone (симядро, классификатор, RL-среда) живёт в src/ репозитория
и не устанавливается через colcon. Узлы ROS2 находят его так:
1) переменная окружения ROBOZONE_ROOT (задаётся launch-файлом), либо
2) поиск вверх от текущего файла до каталога с sim/scene.xml.
"""
import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    env = os.environ.get("ROBOZONE_ROOT")
    if env and (Path(env) / "sim" / "scene.xml").exists():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "sim" / "scene.xml").exists():
            return parent
    raise RuntimeError(
        "Не найден корень проекта robozone. Задайте ROBOZONE_ROOT "
        "на каталог репозитория (где лежит sim/scene.xml).")


def ensure_robozone_on_path() -> Path:
    root = find_project_root()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root
