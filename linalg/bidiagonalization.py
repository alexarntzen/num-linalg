import numpy as np


def lanczos_bidiag(A: np.ndarray, k: int, b: np.ndarray):
    # b is an arbitrary vector in R^m
    # from algoritm:
    # P[:, i] = u_{i+1}
    # Q[:, i] = v_{i+1}
    alpha = np.zeros(k)
    beta = np.zeros(k)
    P = np.zeros((k, k))
    Q = np.zeros((k, k))

    beta[0] = np.linalg.norm(b, ord=2)
    P[:, 0] = b / beta
    v = A.T @ P[:, 0]
    alpha[0] = np.linalg.norm(v, ord=2)
    Q[:, 0] = v / alpha

    for i in range(k - 1):
        # find next u and beta
        u = A @ Q[:, i] - alpha[i] * P[:, i]
        beta[i + 1] = np.linalg.norm(u, ord=2)
        P[:, i + 1] = u / beta[i + 1]

        # find next v and alpha
        v = A.T @ P[:, i + 1] - beta[i + 1] * Q[:, i]
        alpha[i + 1] = np.linalg.norm(v, ord=2)
        P[:, i + 1] = v / alpha[i + 1]

    return P, Q, alpha, beta


def lanczos_bidiag_reorth(A: np.ndarray, k: int, b: np.ndarray):
    # b is an arbitrary vector in R^m
    # from algoritm:
    # P[:, i] = u_{i+1}
    # Q[:, i] = v_{i+1}
    alpha = np.zeros(k)
    beta = np.zeros(k)
    P = np.zeros((k, k))
    Q = np.zeros((k, k))

    beta[0] = np.linalg.norm(b, ord=2)
    P[:, 0] = b / beta
    v = A.T @ P[:, 0]
    alpha[0] = np.linalg.norm(v, ord=2)
    Q[:, 0] = v / alpha

    for i in range(k - 1):
        # find next u and beta
        u = A @ Q[:, i] - alpha[i] * P[:, i]
        beta[i + 1] = np.linalg.norm(u, ord=2)
        P[:, i + 1] = u / beta[i + 1]

        # find next v and alpha
        v = A.T @ P[:, i + 1] - beta[i + 1] * Q[:, i]
        alpha[i + 1] = np.linalg.norm(v, ord=2)
        P[:, i + 1] = v / alpha[i + 1]

    return P, Q, alpha, beta
