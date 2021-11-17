import numpy as np


def get_FU(m, k):
    H = np.random.rand(m, k)
    G = np.random.rand(m, k)
    U, _ = np.linalg.qr(H)
    F = (np.eye(m) - U @ U.T) @ G
    return F, U


def get_CD(F, U):
    C = np.concatenate((F, -U), axis=1)
    D = np.concatenate((U, F), axis=1)
    return C, D


def get_B(C, D):
    return C @ D.T


def get_FUCDB(m, k):
    F, U = get_FU(m, k)
    C, D = get_CD(F, U)
    B = get_B(C, D)
    return F, U, C, D, B
