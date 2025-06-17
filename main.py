#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from aruco_msgs.msg import MarkerArray, MarkerInfo
from tf_transformations import euler_from_quaternion
import subprocess
import time

import prediction as pr
import update as up
import plot

class BagListener(Node):
    def __init__(self):
        super().__init__('bag_listener')
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)
        self.mk_sub   = self.create_subscription(
            MarkerArray, '/marker_info', self.markers_cb, 10)


class EkfSlamNode(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')
        #self.xhat = np.array([[0.0], 
        #                [0.89],
        #                [0.0]]) 

        self.xhat = np.array([[0.0], 
                        [0.0],
                        [0.0]])
        
        # self.xhat_pred = np.array([[0.0], 
                        # [0.0],
                        # [0.0]]) 
        
        self.mx0 = 0.0
        self.my0 = 0.89
        self.theta0 = 0.0

        self.M = 0

        # landmark
        self.fixed_landmarks = np.array([
        [1.96, 0],
        [3.96, 0],
        [5.56, 0],
        [7.56, 0],
        [9.56, 0]
        ])
        
        # self.P_pred = np.eye(3)*1e-2
        self.P = np.eye(3)*1e-2
        self.landmark_map = []
        self.Qr = np.diag([0.05, 0.05])
        self.Rr = np.diag([10,  10])
        #self.Rr = np.diag([0.1, 0.01])
        self.prev_odom = None
        self.x_hist = [self.xhat[0,0]]  # histórico de posições x
        self.y_hist = [self.xhat[1,0]]  # histórico de posições y      
        self.landmark_xy = {}  # dicionário {id: (x,y)} dos landmarks estimados

        # Inicializações da simulação
        self.fig, self.ax, self.trajectory_line, self.robot_dot, self.landmarks_scatter = plot.init_plot()
            
        # subscribers
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(
            MarkerArray, '/marker_info', self.marker_callback, 10)


    def odom_callback(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # quaternion → (roll,pitch,yaw)
        _, _, yaw = euler_from_quaternion([qx,qy,qz,qw])

        if self.prev_odom is None:
            self.prev_odom = (px, py, yaw)

            # 1) Alinha o estado estimado ao ground-truth inicial
            self.xhat[0,0] = px + self.mx0
            self.xhat[1,0] = py + self.my0
            self.xhat[2,0] = yaw + self.theta0

            # 2) Reseta os históricos para começar do mesmo ponto
            self.x_hist = [px + self.mx0]
            self.y_hist = [py + self.my0]
            return

        x0,y0,th0 = self.prev_odom
        delta_r = math.hypot(px - x0, py - y0)
        #delta_theta = np.arctan2(np.sin(yaw),np.cos(th0))
        delta_theta = (yaw - th0 + np.pi) % (2*np.pi) - np.pi
        u_k = np.array([[delta_r],[delta_theta]])
        print('predicao')

        self.xhat, self.P = pr.prediction_step(u_k, self.xhat, self.P, self.Qr)
        self.x_hist.append(self.xhat[0,0])
        self.y_hist.append(self.xhat[1,0])

        
        #plot.plot_state(self.ax, self.fig, self.xhat, self.x_hist, self.y_hist, self.landmark_xy, self.trajectory_line, self.robot_dot, self.landmarks_scatter, self.gt_line)
        plot.plot_state(
            self.ax, self.fig,
            self.xhat,
            self.x_hist, self.y_hist,
            self.landmark_xy,
            self.trajectory_line,
            self.robot_dot,
            self.landmarks_scatter
        )

        self.prev_odom = (px, py, yaw)


    def marker_callback(self, msg: MarkerArray):              
        for m in msg.markers:
            # 1) posição RELATIVA do marcador no frame do robô (dados do tópico)
            # x_rel = m.x
            # y_rel = m.y
            # 2) pose atual do robô no mundo (estimada pelo EKF)
            # x_r   = float(self.xhat[0])
            # y_r   = float(self.xhat[1])
            # th_r  = float(self.xhat[2])
            # 3) transforma para WORLD
            #x_glb, y_glb = self.transform_to_world(x_r, y_r, th_r, x_rel, y_rel)
            # 4) armazena para visualização
            #self.landmark_xy[m.id] = (x_glb, y_glb)

            sens = np.array([[m.id], [m.range], [math.radians(m.psi)]])
            print('correção')
            self.xhat, self.P, self.M, self.landmark_map = up.update_step(sens, self.xhat, self.P, self.M, self.Rr, self.landmark_map, self.landmark_xy, self.fixed_landmarks)
            
            idx    = self.landmark_map.index(m.id)
            offset = 3 + 2 * idx
            xm     = float(self.xhat[offset,   0])
            ym     = float(self.xhat[offset+1, 0])
            self.landmark_xy[m.id] = (xm, ym)

        #plot.plot_state(self.ax, self.fig, self.xhat, self.x_hist, self.y_hist, self.landmark_xy, self.trajectory_line, self.robot_dot, self.landmarks_scatter, self.gt_line)
        plot.plot_state(
        self.ax, self.fig,
        self.xhat,
        self.x_hist, self.y_hist,
        self.landmark_xy,
        self.trajectory_line,
        self.robot_dot,
        self.landmarks_scatter,
    )

        
        #plot.plot_state(self.ax, self.fig, self.xhat, self.x_hist, self.y_hist,
        #self.landmark_xy, self.trajectory_line, self.robot_dot, self.landmarks_scatter)

def main(args=None):
    rclpy.init(args=args)
    node = EkfSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # main()

    rclpy.init()

    # 1) Dispara o rosbag em background
    #    ajuste o caminho e opções como você precisa (por ex. --loop, --rate, etc).
    bag_proc = subprocess.Popen([
        'ros2', 'bag', 'play',
        '/home/robotzu/mobile_robot_localization/robot_localization/marcadores_23mai1_0.db3', 
        '--clock'
    ])

    node = EkfSlamNode()
    time.sleep(0.5)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        bag_proc.terminate()
        bag_proc.wait()
    