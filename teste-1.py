#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from aruco_msgs.msg import MarkerArray
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

class MarkerInCamera(Node):
    def __init__(self):
        super().__init__('marker_in_camera')
        # 1) Buffer e listener de TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 2) Subscriber dos MarkerInfo (poses no frame robot_base)
        self.sub = self.create_subscription(
            MarkerArray,
            '/marker_info',
            self.cb_markers,
            10)

    def cb_markers(self, msg: MarkerArray):
        # Tenta buscar a transformação camera_imu_link ← robot_base
        try:
            t_cam2robot = self.tf_buffer.lookup_transform(
                'camera_imu_link',  # target frame
                'robot_base',       # source frame
                rclpy.time.Time(),  # agora
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF não disponível: {e}")
            return

        for m in msg.markers:
            # 3a) Monte um PointStamped no frame robot_base
            p_robot = PointStamped()
            p_robot.header.frame_id = 'robot_base'
            p_robot.header.stamp = m.header.stamp
            p_robot.point.x = m.x
            p_robot.point.y = m.y
            p_robot.point.z = 0.0   # planar

            # 3b) Transforme para camera_imu_link
            p_cam: PointStamped = do_transform_point(p_robot, t_cam2robot)

            # 4) Ajuste a orientação (psi) se quiser:
            #    supondo que m.psi seja graus no frame robot_base,
            #    e que o frame camera → robot tenha rotação yaw_offset:
            # yaw_offset = tf2_transform_to_yaw(t_cam2robot.transform.rotation)
            # psi_cam = m.psi + degrees(yaw_offset)

            self.get_logger().info(
                f"Marker {m.id} na câmera: x={p_cam.point.x:.3f}, "
                f"y={p_cam.point.y:.3f}, z={p_cam.point.z:.3f}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = MarkerInCamera()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
