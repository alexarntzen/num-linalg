import numpy as np
from scipy.sparse import diags
import timeit
from functools import partial


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
    # check if of type U, S, V
    if isinstance(Y_list[0], tuple):
        mat_array = np.array(Y_list, dtype=object)[:, 1]
    elif len(Y_list[0].shape) != 3:
        raise ValueError("Y_list format not supported")
    else:
        mat_array = np.array(Y_list)
    return np.linalg.eig(mat_array)


def get_data_of_N(method: callable, case, N_list, tol=1e-12, **method_kwargs):
    """Get convergence data"""
    iters = np.zeros(len(N_list))
    final_res = np.zeros(len(N_list))
    for i, N in enumerate(N_list):
        rhs = case.get_rhs(N)
        x_0 = case.get_u_0(N)
        _, conv_hist = method(
            x_0=x_0, rhs=rhs, N=N, tol=tol, conv_hist=True, **method_kwargs
        )
        N_list[i] = N
        iters[i] = len(conv_hist)
        final_res[i] = conv_hist[-1]
    return N_list, iters, final_res


def get_function_timings(func: callable, inputs, number=1000):
    times = np.zeros(len(inputs))
    for i, input in enumerate(inputs):
        # return in ms
        times[i] = timeit.timeit(partial(func, *input), number=number) * 1000 / number
    return times
