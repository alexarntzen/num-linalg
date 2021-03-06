import numpy as np
from scipy.sparse import diags
from linalg.helpers import multiply_factorized


def lanczos_bidiag(A: np.ndarray, k: int, b: np.ndarray):
    # b is an arbitrary vector in R^m
    # from algorithm:
    # P[:, i] = u_{i+1}
    # Q[:, i] = v_{i+1}
    alpha = np.zeros(k)
    beta = np.zeros(k)
    m, n = A.shape
    P = np.zeros((m, k))
    Q = np.zeros((n, k))

    beta[0] = np.linalg.norm(b, ord=2)
    P[:, 0] = b / beta[0]
    v = A.T @ P[:, 0]
    alpha[0] = np.linalg.norm(v, ord=2)
    Q[:, 0] = v / alpha[0]

    for i in range(k - 1):
        # find next u and beta
        u = A @ Q[:, i] - alpha[i] * P[:, i]
        beta[i + 1] = np.linalg.norm(u, ord=2)
        P[:, i + 1] = u / beta[i + 1]

        # find next v and alpha
        v = A.T @ P[:, i + 1] - beta[i + 1] * Q[:, i]
        alpha[i + 1] = np.linalg.norm(v, ord=2)
        Q[:, i + 1] = v / alpha[i + 1]

    return P, Q, alpha, beta


def lanczos_bidiag_reorth(A: np.ndarray, k: int, b: np.ndarray, full_matrices=False):
    # b is an arbitrary vector in R^m
    # from algorithm:
    # P[:, i] = u_{i+1}
    # Q[:, i] = v_{i+1}
    alpha = np.zeros(k)
    beta = np.zeros(k)
    m, n = A.shape
    P = np.zeros((m, k))
    Q = np.zeros((n, k))

    beta[0] = np.linalg.norm(b, ord=2)
    P[:, 0] = b / beta[0]
    v = A.T @ P[:, 0]
    alpha[0] = np.linalg.norm(v, ord=2)
    Q[:, 0] = v / alpha[0]

    for i in range(k - 1):
        # find next u and beta
        u = A @ Q[:, i] - alpha[i] * P[:, i]
        beta[i + 1] = np.linalg.norm(u, ord=2)
        P[:, i + 1] = u / beta[i + 1]

        w = A.T @ P[:, i + 1] - beta[i + 1] * Q[:, i]
        # this is weird because of stupid 0-indexing
        for j in range(i + 1):
            w = w - np.inner(Q[:, j], w) * Q[:, j]

        # find next v and alpha
        alpha[i + 1] = np.linalg.norm(w, ord=2)
        Q[:, i + 1] = w / alpha[i + 1]
    if full_matrices:
        return P, make_bidiagonal(alpha, beta), Q
    else:
        return P, Q, alpha, beta


def make_bidiagonal(alpha: np.ndarray, beta: np.ndarray):
    """Transform two vectors to a bidiagonal matrix"""
    bidiagonals = [alpha, beta[1:]]
    return diags(bidiagonals, [0, -1]).toarray()


def get_bidiagonal_approx(A: np.ndarray, k: int, b: np.ndarray):
    P, Q, alpha, beta = lanczos_bidiag_reorth(A, k, b)
    S = make_bidiagonal(alpha, beta)
    return multiply_factorized(P, S, Q)
