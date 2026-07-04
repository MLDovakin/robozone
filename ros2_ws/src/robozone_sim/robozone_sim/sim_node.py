"""Узел симуляции ПАК сортировки: MuJoCo-ячейка UR10e + мост ROS2.

Публикует:
  /robozone/joint_states     sensor_msgs/JointState  — 6 суставов UR10e (+ вакуум)
  /robozone/ee_pose          geometry_msgs/PoseStamped — поза присоски
  /robozone/suction_state    std_msgs/Bool           — вакуум вкл/выкл (реально держит)
  /robozone/active_object    std_msgs/String         — имя товара в накопителе
  /robozone/object_pose      geometry_msgs/PoseStamped — поза активного товара
  tf: world -> ee, world -> object

Подписывается:
  /robozone/position_command std_msgs/Float64MultiArray — целевые позиции 6 суставов
        (совместимо по смыслу с forward_position_controller / ur_robot_driver)
  /robozone/suction_command  std_msgs/Bool             — включить/выключить вакуум
  /robozone/spawn            std_msgs/String           — спавн товара по имени / "random"

Сервисы:
  /robozone/reset            std_srvs/Trigger          — сброс сцены
  /robozone/set_suction      std_srvs/SetBool          — вакуум (сервисная форма)

Параметры:
  scene: имя xml (по умолчанию scene.xml)
  step_rate_hz: частота публикации/шага (100)
  substeps: шагов физики за тик (5 -> 0.01 c при timestep 0.002)
  belt_speed: скорость подающей ленты (1.0 м/с из постановки)
  spawn_mode: accumulator|belt
  auto_spawn: спавнить случайный товар при старте
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String, Float64MultiArray

# Часть сборок ROS2 не содержит std_srvs / tf2_ros. Тогда узел работает по
# топикам (спавн/вакуум/сброс дублируются топиком /robozone/command),
# а TF и сервисы просто не поднимаются.
try:
    from std_srvs.srv import Trigger, SetBool
    _HAVE_SRV = True
except ImportError:
    _HAVE_SRV = False
try:
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster
    _HAVE_TF = True
except ImportError:
    _HAVE_TF = False

from ._paths import ensure_robozone_on_path

ensure_robozone_on_path()
import numpy as np  # noqa: E402
from robozone.sim_core import RobozoneSim, ARM_JOINTS, BELT_A_SPEED  # noqa: E402


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """3x3 -> кватернион (x, y, z, w)."""
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, m.flatten())
    return np.array([q[1], q[2], q[3], q[0]])  # mujoco: w,x,y,z -> ros: x,y,z,w


class SimNode(Node):
    def __init__(self):
        super().__init__("robozone_sim")
        self.declare_parameter("scene", "scene.xml")
        self.declare_parameter("step_rate_hz", 100.0)
        self.declare_parameter("substeps", 5)
        self.declare_parameter("belt_speed", BELT_A_SPEED)
        self.declare_parameter("spawn_mode", "accumulator")
        self.declare_parameter("auto_spawn", True)

        root = ensure_robozone_on_path()
        scene = self.get_parameter("scene").value
        self.substeps = int(self.get_parameter("substeps").value)
        self.belt_speed = float(self.get_parameter("belt_speed").value)
        self.spawn_mode = self.get_parameter("spawn_mode").value

        self.sim = RobozoneSim(scene_xml=root / "sim" / scene)
        self.sim.set_belt_speed(self.belt_speed)
        if self.get_parameter("auto_spawn").value:
            self.sim.spawn(mode=self.spawn_mode)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_js = self.create_publisher(JointState, "robozone/joint_states", qos)
        self.pub_ee = self.create_publisher(PoseStamped, "robozone/ee_pose", qos)
        self.pub_suc = self.create_publisher(Bool, "robozone/suction_state", qos)
        self.pub_obj = self.create_publisher(String, "robozone/active_object", qos)
        self.pub_objpose = self.create_publisher(PoseStamped, "robozone/object_pose", qos)
        self.tf = TransformBroadcaster(self) if _HAVE_TF else None

        self.create_subscription(Float64MultiArray, "robozone/position_command",
                                 self._on_position_cmd, qos)
        self.create_subscription(Bool, "robozone/suction_command",
                                 self._on_suction_cmd, qos)
        self.create_subscription(String, "robozone/spawn", self._on_spawn, qos)
        # текстовые команды (работают и без std_srvs): "reset" | "suction_on/off"
        self.create_subscription(String, "robozone/command", self._on_command, qos)

        if _HAVE_SRV:
            self.create_service(Trigger, "robozone/reset", self._srv_reset)
            self.create_service(SetBool, "robozone/set_suction", self._srv_suction)
        else:
            self.get_logger().warn(
                "std_srvs недоступен: используйте топик /robozone/command "
                "(reset|suction_on|suction_off)")

        rate = float(self.get_parameter("step_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self._on_tick)
        self.get_logger().info(
            f"robozone_sim запущен: сцена={scene}, лента={self.belt_speed} м/с, "
            f"товар={self.sim.active_object}")

    # ------------------------------------------------------------- callbacks
    def _on_position_cmd(self, msg: Float64MultiArray):
        if len(msg.data) >= 6:
            self.sim.set_arm_ctrl(np.array(msg.data[:6]))

    def _on_suction_cmd(self, msg: Bool):
        self.sim.set_suction(1.0 if msg.data else 0.0)

    def _on_spawn(self, msg: String):
        name = None if msg.data in ("", "random") else msg.data
        try:
            got = self.sim.spawn(name, mode=self.spawn_mode)
            self.get_logger().info(f"спавн товара: {got}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"спавн не удался: {e}")

    def _on_command(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == "reset":
            self.sim.reset(self.belt_speed)
            self.sim.spawn(mode=self.spawn_mode)
            self.get_logger().info(f"сброс, товар={self.sim.active_object}")
        elif cmd == "suction_on":
            self.sim.set_suction(1.0)
        elif cmd == "suction_off":
            self.sim.set_suction(0.0)
        else:
            self.get_logger().warn(f"неизвестная команда: {cmd}")

    def _srv_reset(self, request, response):
        self.sim.reset(self.belt_speed)
        self.sim.spawn(mode=self.spawn_mode)
        response.success = True
        response.message = f"сброшено, товар={self.sim.active_object}"
        return response

    def _srv_suction(self, request, response):
        self.sim.set_suction(1.0 if request.data else 0.0)
        response.success = True
        response.message = "вакуум " + ("вкл" if request.data else "выкл")
        return response

    # ------------------------------------------------------------------ loop
    def _on_tick(self):
        self.sim.step(self.substeps)
        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = list(ARM_JOINTS) + ["suction"]
        q = self.sim.arm_qpos()
        qd = self.sim.arm_qvel()
        suc = float(self.sim.data.ctrl[self.sim._suction_act])
        js.position = [*q.tolist(), suc]
        js.velocity = [*qd.tolist(), 0.0]
        self.pub_js.publish(js)

        ee_p = self.sim.ee_pos()
        ee_q = _mat_to_quat(self.sim.ee_mat())
        self.pub_ee.publish(self._pose_msg(now, ee_p, ee_q))
        self._send_tf(now, "world", "suction_tip", ee_p, ee_q)

        held = self.sim.is_held()
        self.pub_suc.publish(Bool(data=bool(held)))

        if self.sim.active_object:
            self.pub_obj.publish(String(data=self.sim.active_object))
            op = self.sim.object_qpos()
            opos, oquat = op[:3], np.array([op[4], op[5], op[6], op[3]])
            self.pub_objpose.publish(self._pose_msg(now, opos, oquat))
            self._send_tf(now, "world", "active_object", opos, oquat)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _pose_msg(stamp, pos, quat) -> PoseStamped:
        m = PoseStamped()
        m.header.stamp = stamp
        m.header.frame_id = "world"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
        m.pose.orientation.x = float(quat[0])
        m.pose.orientation.y = float(quat[1])
        m.pose.orientation.z = float(quat[2])
        m.pose.orientation.w = float(quat[3])
        return m

    def _send_tf(self, stamp, parent, child, pos, quat):
        if self.tf is None:
            return
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.tf.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
