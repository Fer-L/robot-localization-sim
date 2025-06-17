#!/usr/bin/env python3
import numpy as np
import math
from scipy.linalg import block_diag

def update_step(sens, xhat, P, M, Rr, landmark_map, landmark_xy, fixed_landmarks):

    lm_id = int(sens[0])
    rr_meas = float(sens[1])
    phi_meas = float(sens[2])

    xr = float(xhat[0][0])
    yr = float(xhat[1][0])
    thr = float(xhat[2][0])


    if lm_id not in landmark_map:
        M += 1
        landmark_map.append(lm_id)

        # Calcula posição global do landmark:
        # [xm; ym] = [xr; yr] + rot(th
        # r + phi) * [r_r; 0]

        xm = xr + rr_meas * np.cos(thr + phi_meas)
        ym = yr + rr_meas * np.sin(thr + phi_meas)


        #landmark_xy[lm_id] = (xm, ym)  # Para plotar depois
        pmi = np.array([[xm],
                    [ym]])
        xhat = np.vstack((xhat, pmi))
        
        Gmxk = np.array([
            [1, 0, -rr_meas * np.sin(thr + phi_meas)],
            [0, 1, rr_meas * np.cos(thr + phi_meas)]
        ])
        Gmzk = np.array([
            [np.cos(thr + phi_meas), -rr_meas * np.sin(thr + phi_meas)],
            [np.sin(thr + phi_meas), rr_meas * np.cos(thr + phi_meas)]
        ])

        n = P.shape[0]
        # topo: [I_n , zeros(n,2)]
        top = np.hstack([np.eye(n), np.zeros((n, 2))]) # parei o debug aqui

        zeros_block = np.zeros((2, n-3))
        # bottom = np.hstack([np.hstack([Gmxk, zeros_block]), Gmzk])
        bottom = np.hstack([Gmxk, zeros_block, Gmzk])
        Ym = np.vstack([top, bottom])

        # Expande P_pred: P_aug = Ym @ block_diag(P_pred, Rr) @ Ym.T
        P = Ym @ block_diag(P, Rr) @ Ym.T

    # --- passo de correção EKF para este landmark ---
    # Atualiza xhat e P locais
    N = xhat.size

    # Encontra índice do landmark no vetor (0-based em Python)
    idx = landmark_map.index(lm_id)  # idx varia de 0 a M-1

    offset = 3 + 2 * idx
    xm = fixed_landmarks[idx][0] 
    ym = fixed_landmarks[idx][1]

    # Cálculo da predição de medição #####
    dx = xm - xr
    dy = ym - yr

    r_r_pred = math.hypot(dx, dy)
    phi_pred = ((np.arctan2(dy, dx) - thr) + np.pi) % (2*np.pi) - np.pi

    print(f'r_pred = {r_r_pred}')
    print(f'phi_pred = {phi_pred}')

    # Jacobiana H_r (2×3) em relação à pose do robô
    H_r = np.array([
        [-dx / r_r_pred, -dy / r_r_pred, 0],
        [ dy / (r_r_pred**2), -dx / (r_r_pred**2), -1]
    ])

    # Jacobiana H_m (2×2) em relação ao landmark
    H_m = np.array([
        [ dx / r_r_pred, dy / r_r_pred ],
        [-dy / (r_r_pred**2), dx / (r_r_pred**2) ]
    ])
    
    # Monta H completo (2×N)
    H = np.zeros((2, N))
    H[:, 0:3] = H_r
    H[:, offset:offset+2] = H_m

    # Inovação
    z    = np.array([rr_meas, phi_meas])
    zhat = np.array([r_r_pred, phi_pred])
    v    = z - zhat
    v[1] = (v[1] + np.pi) % (2*np.pi) - np.pi

    # print(f'z = {z}')
    # print(f'z_hat = {zhat}')

    # Ganho de Kalman
    S = H @ P @ H.T + Rr
    K = P @ H.T @ np.linalg.inv(S)

    # Atualiza estado e covariância
    diff = (K @ v.reshape(-1, 1)) 

    xhat = xhat + diff
    xhat[2,0] = (xhat[2,0] + math.pi) % (2*math.pi) - math.pi

    P = (np.eye(N) - K @ H) @ P

    # xhat = np.add(xhat, diff)
    # print(landmark_map)

    return xhat, P, M, landmark_map