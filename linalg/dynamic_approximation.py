import numpy as np


def cayley_map_simple(B: np.ndarray):
    half_B = B * 0.5
    m = B.shape[0]
    I_m = np.eye(m)
    return np.linalg.inv(I_m - half_B) @ (I_m + half_B)


def cayley_map_efficient(C: np.ndarray, D: np.ndarray):
    m, p = C.shape
    I_m, I_p = np.eye(m), np.eye(p)
    cay = I_m + C @ np.linalg.inv(I_p - 0.5 * D.T @ C) @ D.T
    return cay


def cayley_map_plus(F, U):
    """B=[F, -U]@[U, F].T, U@U.T =I, F.T@U=0"""
    C = np.concatenate((F, -U), axis=1)
    D = np.concatenate((U, F), axis=1)
    m, k = F.shape
    I_m, I_k = np.eye(m), np.eye(k)
    Q = F.T @ F
    A = np.linalg.inv(I_k + 0.25 * Q)
    B = -1 / 2 * A
    inner_inv = np.block([[A, B], [0.5 * Q @ A, -2 * B]])
    cay = I_m + C @ inner_inv @ D.T
    return cay
