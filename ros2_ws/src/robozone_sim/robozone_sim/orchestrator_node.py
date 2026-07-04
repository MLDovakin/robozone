"""Оркестратор исполнительного контура через ROS2 (скриптовая политика).

Замыкает контур ПАК поверх топиков sim_node и classifier_node:
  вход:  /robozone/joint_states, /robozone/object_pose, /robozone/suction_state,
         /robozone/target_zone
  выход: /robozone/position_command (6 целевых позиций суставов),
         /robozone/suction_command

Планирование положения кончика присоски -> суставы решается дифференциальной
IK. Кинематика считается на отдельной read-only копии MuJoCo-модели, которая
синхронизируется по /joint_states (никакой физики — только FK/якобиан).
На реальном роботе этот блок заменяется URDF+KDL/біблиотекой IK с тем же
контрактом топиков; sim_node меняется на ur_robot_driver.

Конечный автомат: WAIT -> APPROACH -> DESCEND -> LIFT -> TRANSFER -> PLACE ->
RELEASE -> HOME -> (следующий товар).
"""
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String, Float64MultiArray

from ._paths import ensure_robozone_on_path

ensure_robozone_on_path()
import mujoco  # noqa: E402
from robozone.sim_core import (  # noqa: E402
    RobozoneSim, DROP_POINT, HOME_QPOS, placement_tip_target,
)
from robozone.ik import ik_step  # noqa: E402


class Orchestrator(Node):
    def __init__(self):
        super().__init__("robozone_orchestrator")
        # read-only кинематическая копия
        self.kin = RobozoneSim()
        self.q = HOME_QPOS.copy()
        self.q_cmd = HOME_QPOS.copy()
        self.obj_pos = None
        self.obj_bottom_dz = 0.0     # смещение низа товара относительно центра поза
        self.held = False
        self.zone = None
        self.state = "WAIT"
        self.hold_dz = 0.12          # смещение захвата (тип выше низа товара)
        self.t_state = self.get_clock().now()

        self.pub_cmd = self.create_publisher(
            Float64MultiArray, "robozone/position_command", 10)
        self.pub_suc = self.create_publisher(Bool, "robozone/suction_command", 10)
        self.create_subscription(JointState, "robozone/joint_states",
                                 self._on_js, 10)
        self.create_subscription(PoseStamped, "robozone/object_pose",
                                 self._on_obj, 10)
        self.create_subscription(Bool, "robozone/suction_state",
                                 self._on_suc, 10)
        self.create_subscription(String, "robozone/target_zone",
                                 self._on_zone, 10)
        self.timer = self.create_timer(0.05, self._tick)  # 20 Гц планирование
        self.get_logger().info("оркестратор готов")

    # ------------------------------------------------------------- callbacks
    def _on_js(self, msg: JointState):
        if len(msg.position) >= 6:
            self.q = np.array(msg.position[:6])

    def _on_obj(self, msg: PoseStamped):
        p = msg.pose.position
        self.obj_pos = np.array([p.x, p.y, p.z])

    def _on_suc(self, msg: Bool):
        self.held = msg.data

    def _on_zone(self, msg: String):
        if msg.data in DROP_POINT and self.zone != msg.data:
            self.zone = msg.data
            if self.state == "WAIT":
                self._goto("APPROACH")

    # --------------------------------------------------------------- helpers
    def _goto(self, s):
        self.state = s
        self.t_state = self.get_clock().now()
        self.get_logger().info(f"-> {s}")

    def _dt(self) -> float:
        return (self.get_clock().now() - self.t_state).nanoseconds * 1e-9

    def _ee_pos(self) -> np.ndarray:
        self.kin.data.qpos[self.kin._arm_qadr] = self.q
        mujoco.mj_forward(self.kin.model, self.kin.data)
        return self.kin.ee_pos()

    def _servo(self, target: np.ndarray, down=True, max_dq=0.06):
        """Один шаг IK от текущей команды к цели, публикация position_command."""
        self.kin.data.qpos[self.kin._arm_qadr] = self.q
        mujoco.mj_forward(self.kin.model, self.kin.data)
        dq = ik_step(self.kin, target, down=down, max_dq=max_dq)
        self.q_cmd = np.clip(self.q_cmd + dq, -6.28, 6.28)
        self.pub_cmd.publish(Float64MultiArray(data=self.q_cmd.tolist()))

    def _reached(self, target, tol=0.03) -> bool:
        return np.linalg.norm(target - self._ee_pos()) < tol

    def _suction(self, on: bool):
        self.pub_suc.publish(Bool(data=on))

    # ------------------------------------------------------------------ loop
    def _tick(self):
        if self.obj_pos is None or self.zone is None:
            return
        ee = self._ee_pos()

        if self.state == "WAIT":
            return

        if self.state == "APPROACH":
            self.hover = self.obj_pos + [0, 0, self.hold_dz + 0.18]
            self._servo(self.hover)
            if self._reached(self.hover, tol=0.04) or self._dt() > 5:
                self._suction(True)
                self._goto("DESCEND")

        elif self.state == "DESCEND":
            grasp = self.obj_pos + [0, 0, self.hold_dz]
            self._servo(grasp, max_dq=0.03)
            if self.held:
                self.hold_dz = float(ee[2] - self.obj_pos[2])
                self._goto("LIFT")
            elif self._dt() > 5:
                self._goto("APPROACH")  # повтор

        elif self.state == "LIFT":
            self._suction(True)
            up = np.array([ee[0], ee[1], 1.25])
            self._servo(up)
            if ee[2] > 1.2 or self._dt() > 3:
                self.drop = DROP_POINT[self.zone]
                self._goto("TRANSFER")

        elif self.state == "TRANSFER":
            self._suction(True)
            via = np.array([0.55, self.drop[1] * 0.4, 1.25])
            self._servo(via)
            if self._reached(via, tol=0.08) or self._dt() > 3:
                self._goto("PLACE")

        elif self.state == "PLACE":
            self._suction(True)
            tip_target = placement_tip_target(self.zone, self.hold_dz)
            self._servo(tip_target, max_dq=0.04)
            if self._reached(tip_target, tol=0.04) or self._dt() > 4:
                self._goto("RELEASE")

        elif self.state == "RELEASE":
            self._suction(False)
            if self._dt() > 1.5:
                self._goto("HOME")

        elif self.state == "HOME":
            self.q_cmd = HOME_QPOS.copy()
            self.pub_cmd.publish(Float64MultiArray(data=self.q_cmd.tolist()))
            if self._dt() > 1.5:
                self.zone = None
                self._goto("WAIT")
                self.get_logger().info("цикл завершён, ожидаю следующий товар")


def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
