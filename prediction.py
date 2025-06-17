#!/usr/bin/env python3
import numpy as np

def prediction_step(u_k, xhat, P, Qr):
    xr, yr, thr = xhat[0], xhat[1], xhat[2]
    delta_rd, delta_rtheta = u_k[0], u_k[1]

    xr_pred  = xr + delta_rd * np.cos(thr)
    yr_pred  = yr + delta_rd * np.sin(thr)
    thr_pred = ((thr + delta_rtheta) + np.pi) % (2*np.pi) - np.pi

    xhat_pred = np.vstack(([xr_pred, yr_pred, thr_pred], xhat[3:]))

    #xhat_pred = np.hstack([[xr_pred, yr_pred, thr_pred], xhat[3:] ])
    #xhat_pred = np.hstack(( [xr_pred, yr_pred, thr_pred],  ))

    M = (P.shape[0] - 3) // 2  # número de landmarks ativos no estado

    Fr = np.array([
    [1, 0, (-delta_rd * np.sin(thr))[0]],
    [0, 1, (delta_rd * np.cos(thr))[0]],
    [0, 0, 1]])

    F = np.block([
    [Fr, np.zeros((3, 2*M))],
    [np.zeros((2*M, 3)), np.eye(2*M)]
    ])

    Gr = np.array([
    [(np.cos(thr))[0], 0],
    [(np.sin(thr))[0], 0],
    [0, 1]
    ])

    G = np.vstack((Gr, np.zeros((2*M, 2))))

    P_pred = F @ P @ F.T + G @ Qr @ G.T
    # print(f'P_pred = {P_pred}')
    # print(f'xhat_pred = {xhat_pred}')
    return xhat_pred, P_pred
