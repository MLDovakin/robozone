"""Узел классификации: определяет категорию товара в накопителе.

Подписывается на /robozone/active_object (имя товара) и публикует его
категорию/целевую зону. В симуляции категория берётся из ground-truth
(assets/meshes/objects/categories.json) — это соответствует «выходу CV-части»
и позволяет отлаживать исполнительный контур независимо от зрения.

При переносе на реальную систему этот узел заменяется на инференс CV-модели
по кадру камеры (тот же выходной контракт: строка зоны B/C/D), а остальной
контур — исполнительная часть и мост — не меняется.

Публикует:
  /robozone/category   std_msgs/String  — категория (sortable|oversize|round)
  /robozone/target_zone std_msgs/String — зона B|C|D
"""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ._paths import find_project_root


class ClassifierNode(Node):
    def __init__(self):
        super().__init__("robozone_classifier")
        root = find_project_root()
        cats = json.loads(
            (root / "assets" / "meshes" / "objects" / "categories.json").read_text())
        self.table = {n: (c["category"], c["zone"]) for n, c in cats.items()}

        self.pub_cat = self.create_publisher(String, "robozone/category", 10)
        self.pub_zone = self.create_publisher(String, "robozone/target_zone", 10)
        self.create_subscription(String, "robozone/active_object",
                                 self._on_object, 10)
        self._last = None
        self.get_logger().info(
            f"классификатор готов, объектов в таблице: {len(self.table)}")

    def _on_object(self, msg: String):
        name = msg.data
        if name == self._last:
            return
        self._last = name
        if name not in self.table:
            self.get_logger().warn(f"нет категории для '{name}'")
            return
        cat, zone = self.table[name]
        self.pub_cat.publish(String(data=cat))
        self.pub_zone.publish(String(data=zone))
        self.get_logger().info(f"товар '{name}': категория={cat} -> зона {zone}")


def main(args=None):
    rclpy.init(args=args)
    node = ClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
