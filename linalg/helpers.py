import numpy as np
from scipy.sparse import diags

def operator_to_matrix(operator:callable):
    A = np.zero(2)

def make_bidiagonal(alpha: np.ndarray, beta: np.ndarray):
    bidiagonals = [alpha, beta[1:]]
    return diags(bidiagonals, [0, -1])


def get_best_approx(A, k):
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    s_k = s
    s_k[k:] = 0  # this will overwrite s as well
    A_k = (u * s_k) @ vh
    return A_k
