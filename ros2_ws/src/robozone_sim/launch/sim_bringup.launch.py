"""Запуск полного контура ПАК в ROS2: симуляция + классификатор + оркестратор.

Пример:
  ros2 launch robozone_sim sim_bringup.launch.py spawn_mode:=accumulator
  ros2 launch robozone_sim sim_bringup.launch.py run_orchestrator:=false  # ручное управление / RL

ROBOZONE_ROOT должен указывать на корень репозитория (где sim/scene.xml).
Если не задан, узлы попытаются найти корень автоматически.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    spawn_mode = LaunchConfiguration("spawn_mode")
    belt_speed = LaunchConfiguration("belt_speed")
    run_orchestrator = LaunchConfiguration("run_orchestrator")

    env = {"ROBOZONE_ROOT": os.environ.get("ROBOZONE_ROOT", "")}

    return LaunchDescription([
        DeclareLaunchArgument("spawn_mode", default_value="accumulator",
                              description="accumulator|belt"),
        DeclareLaunchArgument("belt_speed", default_value="1.0",
                              description="скорость подающей ленты, м/с"),
        DeclareLaunchArgument("run_orchestrator", default_value="true",
                              description="запускать скриптовую политику"),

        Node(package="robozone_sim", executable="sim_node", name="robozone_sim",
             output="screen", additional_env=env,
             parameters=[{"spawn_mode": spawn_mode, "belt_speed": belt_speed}]),

        Node(package="robozone_sim", executable="classifier_node",
             name="robozone_classifier", output="screen", additional_env=env),

        Node(package="robozone_sim", executable="orchestrator_node",
             name="robozone_orchestrator", output="screen", additional_env=env,
             condition=IfCondition(run_orchestrator)),
    ])
