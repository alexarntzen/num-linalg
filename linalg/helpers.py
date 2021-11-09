import numpy as np
from scipy.sparse import diags


def make_bidiagonal(alpha: np.ndarray, beta: np.ndarray):
    bidiagonals = [alpha, beta[1:]]
    return diags(bidiagonals, [0, -1])


def get_best_approx(A, k):
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    s_k = s
    s_k[k:] = 0  # this will overwrite s as well
    A_k = (u * s_k) @ vh
    return A_k


def multiply_factorized(U, S, V):
    return np.linalg.multi_dot([U, S, V.T])


def get_singular_values_list(Y_list):
    pass


def get_singular_values(Y):
    pass
