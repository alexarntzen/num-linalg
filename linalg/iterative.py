import numpy as np


def cg(A: callable, x_0, rhs, N: int, tol=1e-12, maxiter=None, conv_hist=False):
    """Assuming all numbers are real"""

    r = rhs - A(x_0, N)
    p = r
    x = x_0
    r_dot_r_0 = np.tensordot(r, r, 2)  # two norm **2
    r_dot_r = r_dot_r_0
    i = 0
    hist = list()
    while r_dot_r / r_dot_r_0 >= tol ** 2:
        # if i % 100 == 0:
        #     print(i, ((r_dot_r / r_dot_r_0) ** 0.5))

        Ap = A(p, N)
        alpha = r_dot_r / np.tensordot(Ap, p, 2)

        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_r_new = np.tensordot(r, r, 2)
        beta = r_dot_r_new / r_dot_r
        p = r + beta * p

        # update for iteration
        r_dot_r = r_dot_r_new
        i += 1
        if conv_hist:
            hist.append(np.sqrt(r_dot_r))
        if maxiter is not None and i > maxiter:
            break

    if conv_hist:
        return x, hist
    else:
        return x


def weighted_jacobi(x_0, rhs, N, w, nu, J_w: callable, D_inv):
    """Perform weighted jacobi iteration"""
    f_w = w * D_inv(rhs, N=N)
    x = x_0  # so we do not change x0
    for _ in range(nu):
        x = J_w(x, N, w) + f_w

    return x
