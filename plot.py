#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def init_plot(xlim=(-1.0, 11.0), ylim=(-1.0, 6.0)):
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fixed_landmarks = np.array([
    [1.96, 0],
    [3.96, 0],
    [5.56, 0],
    [7.56, 0],
    [9.56, 0]
    ])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('EKF-SLAM')
    trajectory_line, = ax.plot([], [], 'b-')
    robot_dot, = ax.plot([], [], 'bo', label='robot')
    landmarks_scatter = ax.scatter([], [], c='r', marker='s', label='landmarks')
    ax.scatter(fixed_landmarks[:, 0], fixed_landmarks[:, 1],
    c='k', marker='x', label='landmark ground-truth')

    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', 'box') 
    ax.set_autoscale_on(False)
    return fig, ax, trajectory_line, robot_dot, landmarks_scatter

def plot_state(ax, fig, xhat, x_hist, y_hist, landmark_xy, trajectory_line, robot_dot, landmarks_scatter):
    x_r = float(xhat[0,0])
    y_r = float(xhat[1,0])
    x_hist.append(x_r)
    y_hist.append(y_r)

    trajectory_line.set_data(x_hist, y_hist)
    robot_dot.set_data([x_r], [y_r])

    if landmark_xy:
        xs = [p[0] for p in landmark_xy.values()]
        ys = [p[1] for p in landmark_xy.values()]
        landmarks_scatter.set_offsets(np.c_[xs, ys])
    else:
        landmarks_scatter.set_offsets(np.empty((0,2)))

    fig.canvas.draw()
    fig.canvas.flush_events()