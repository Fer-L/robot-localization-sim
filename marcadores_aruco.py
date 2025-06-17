#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import TransformStamped
from aruco_msgs.msg import MarkerInfo, MarkerArray
import cv2
from cv_bridge import CvBridge
import tf2_ros
import numpy as np
import math
from tf_transformations import quaternion_from_matrix

class ArucoCamera(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        self.get_logger().info("Iniciando ArUco detector...")
        
        # 1) Rotação
        self.R_c_r = np.array([
          [ 0,  0,  1],
          [-1,  0,  0],
          [ 0, -1,  0],
        ])

        # 2) Translação (medidas)
        self.t_c_r = np.array([0.25, 0.0, 0.1952])

        # 3) Homogênea 4×4
        self.T_c_r = np.eye(4)
        self.T_c_r[:3, :3] = self.R_c_r
        self.T_c_r[:3, 3] = self.t_c_r

        self.T_r_c = np.linalg.inv(self.T_c_r)
                # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/robot1/D455_1/color/image_raw',
            self.image_callback,
            10)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/robot1/D455_1/color/camera_info',
            self.camera_info_callback,
            10)

        self.imu_sub = self.create_subscription(
            Imu,
            '/robot1/D455_1/imu/data',
            self.imu_callback,
            50)
        self.latest_imu = None

        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, 'marker_info', 10)
        self.tf_broadcaster  = tf2_ros.TransformBroadcaster(self)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.camera_frame_id = 'camera_imu_link'
        self.world_frame_id  = 'world'
        self.robot_frame_id  = 'robot_base'

        # 3a) Publica estática: camera_imu_link → robot_base (offset físico)
        quat = quaternion_from_matrix(self.T_c_r)
        
        static_tf = TransformStamped()
        static_tf.header.stamp    = self.get_clock().now().to_msg()
        static_tf.header.frame_id = 'camera_imu_link'
        static_tf.child_frame_id  = 'robot_base'
        static_tf.transform.translation.x = float(self.t_c_r[0])
        static_tf.transform.translation.y = float(self.t_c_r[1])
        static_tf.transform.translation.z = float(self.t_c_r[2])
        static_tf.transform.rotation.x = quat[0]
        static_tf.transform.rotation.y = quat[1]
        static_tf.transform.rotation.z = quat[2]
        static_tf.transform.rotation.w = quat[3]
        self.static_broadcaster.sendTransform(static_tf)

        self.bridge = CvBridge()
        self.cameraMatrix = None
        self.distCoeffs = None
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.size_marker = 0.1
        self.tfBuffer = tf2_ros.Buffer()
        self.tfBroadcaster = tf2_ros.TransformBroadcaster(self)
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.marker_global = {
            # [x, y, z, qx, qy, qz, qw]
            0: [1.96, 0.0, -50.0, 0, 0, 0, 1],
            1: [3.96, 0.0, -50.0, 0, 0, 0, 1],
            2: [5.56, 0.0, -50.0, 0, 0, 0, 1],
            3: [7.56, 0.0, -50.0, 0, 0, 0, 1],
            4: [9.56, 0.0, -50.0, 0, 0, 0, 1]
        }
        #self.init_camera_frame()

    def imu_callback(self, msg: Imu):
        # guarda última orientação da IMU
        self.latest_imu = msg.orientation
        # publica T^world_camera usando orientação da IMU
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame_id
        t.child_frame_id  = self.camera_frame_id
        # suponha câmera montada com origem em world (0,0,0)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.89
        t.transform.translation.z = 0.0
        # usa orientação da IMU diretamente
        t.transform.rotation = msg.orientation
        self.tf_broadcaster.sendTransform(t)

    def camera_info_callback(self, msg):
        self.cameraMatrix = np.array(msg.k).reshape((3, 3))
        self.distCoeffs = np.array(msg.d)

    '''
    def init_camera_frame(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame_id
        t.child_frame_id = self.camera_frame_id
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tfBroadcaster.sendTransform(t)
        '''
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erro na conversão da imagem: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # Retorna T^{C}_{A} (Aruco no sistema da câmera)
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.size_marker, self.cameraMatrix, self.distCoeffs)

            # Cria a mensagem MarkerArray
            marker_array_msg = MarkerArray()

            for i, marker_id in enumerate(ids.flatten()):
                now = self.get_clock().now().to_msg()

                # 1) T^C_A: matriz homogênea do marcador na câmera
                T_c_a = np.eye(4)
                T_c_a[:3, :3] = cv2.Rodrigues(rvec[i])[0]
                T_c_a[:3, 3] = tvec[i][0]

                # 2) T^A_C = inv(T^C_A) -> câmera em relação ao marcador
                #T_a_c = np.linalg.inv(T_c_a)

                # 3) T^G_A: pose ABSOLUTA do marcador no mundo
                gx, gy, gz, qx, qy, qz, qw = self.marker_global[int(marker_id)]
                T_g_a = np.eye(4)
                T_g_a[:3, 3] = [gx, gy, gz]
                # se o marcador tiver rotação global != identidade, preencher T_g_a[:3,:3]

                # 4) T^C_R: offset câmera→robô (já enviado como static_tf)
                # como usamos static broadcaster, não precisamos montar de novo aqui

                # 5) CÁLCULO final: T^G_R = T^G_A · T^A_C · T^C_R
                #    montamos T^C_R manualmente igual ao static_tf:

                

                # (Aruco no sistema do robô)
                T_r_a = self.T_r_c.dot(T_c_a)

                #T_g_r = T_g_a.dot(T_a_c).dot(T_c_r)
                '''
                # 6) broadcast da pose global do robô
                t_robot = TransformStamped()
                t_robot.header.stamp = now
                t_robot.header.frame_id  = self.world_frame_id
                t_robot.child_frame_id = 'robot_base'
                t_robot.transform.translation.x = float(T_g_r[0,3])
                t_robot.transform.translation.y = float(T_g_r[1,3])
                t_robot.transform.translation.z = float(T_g_r[2,3])
                q = quaternion_from_matrix(T_g_r)
                t_robot.transform.rotation.x = q[0]
                t_robot.transform.rotation.y = q[1]
                t_robot.transform.rotation.z = q[2]
                t_robot.transform.rotation.w = q[3]
                self.tfBroadcaster.sendTransform(t_robot)
                '''
                marker_info = MarkerInfo()
                marker_info.id = int(marker_id)
                marker_info.x = float(T_r_a[0,3])
                marker_info.y = float(T_r_a[1,3])
                marker_info.range = float(math.hypot(marker_info.x, marker_info.y)) # distância planar robô→marcador
                
                # ângulo horizontal 
                marker_info.psi = float(math.atan2(marker_info.y, marker_info.x) * (180 / math.pi))
                # ângulo vertical
                #marker_info.alpha = float(math.atan2(marker_info.y, marker_info.range) * (180 / math.pi))

                z_cam = T_c_a[2,3]
                y_cam = T_c_a[1,3]
                marker_info.alpha = math.degrees(math.atan2(y_cam, z_cam))



                marker_info.timestamp = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9                
                '''
                marker_info = MarkerInfo()
                marker_info.id = int(marker_id)
                marker_info.range = float(tvec[i][0][2]) # Distância no eixo Z
                marker_info.psi = float(math.atan2(tvec[i][0][0], marker_info.range) * (180 / math.pi))
                marker_info.x = float(tvec[i][0][0])
                marker_info.y = float(tvec[i][0][1])
                marker_info.timestamp = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
                marker_info.alpha = float(math.atan2(tvec[i][0][1], marker_info.range) * (180 / math.pi))
                '''

                # Adiciona o MarkerInfo ao MarkerArray
                marker_array_msg.markers.append(marker_info)

                # Publica a transformação do marcador (vai estar no sistema do robô)
                t_marker = TransformStamped()
                t_marker.header.stamp = self.get_clock().now().to_msg()
                t_marker.header.frame_id = self.robot_frame_id
                t_marker.child_frame_id = f"marker_{marker_id}"
                t_marker.transform.translation.x = marker_info.x
                t_marker.transform.translation.y = marker_info.y
                t_marker.transform.translation.z = marker_info.range

                #rotation_matrix = np.eye(4)
                #rotation_matrix[0:3, 0:3] = cv2.Rodrigues(rvec[i])[0]
                #r = quaternion_from_matrix(rotation_matrix)
                #t_marker.transform.rotation.x = r[0]
                #t_marker.transform.rotation.y = r[1]
                #t_marker.transform.rotation.z = r[2]
                #t_marker.transform.rotation.w = r[3]

                self.tfBroadcaster.sendTransform(t_marker)

                # Desenhar os eixos de coordenadas do marcador na imagem
                cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, rvec[i], tvec[i], self.size_marker / 2)

            # Publica o array de marcadores
            self.get_logger().info("Publicando array de marcadores")
            self.marker_pub.publish(marker_array_msg)

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Tecla Esc para sair
            self.get_logger().info("Execução encerrada.")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoCamera()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()