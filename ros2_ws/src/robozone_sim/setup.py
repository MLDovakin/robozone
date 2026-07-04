from setuptools import find_packages, setup

package_name = "robozone_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/sim_bringup.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RoboZone Team",
    maintainer_email="team@robozone.local",
    description="MuJoCo-симуляция ПАК сортировки с ROS2-мостом (Трек 3).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Основной узел симуляции + мост ROS2
            "sim_node = robozone_sim.sim_node:main",
            # Классификатор (публикует категорию товара в накопителе)
            "classifier_node = robozone_sim.classifier_node:main",
            # Пример клиента-оркестратора (скриптовая политика через ROS2)
            "orchestrator_node = robozone_sim.orchestrator_node:main",
        ],
    },
)
