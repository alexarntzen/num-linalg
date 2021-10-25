import numpy as np


def cg(A: callable, x_0, rhs, N: int, tol=1e-12, maxiter=None, conv_hist=False):
    """Implementation of conjugate gradient

    Assuming all numbers are real"""
    r = rhs - A(x_0, N)
    p = r
    x = x_0
    r_dot_r_0 = np.sum(r * r)  # two norm **2
    r_dot_r = r_dot_r_0
    i = 0
    hist = list()
    while r_dot_r / r_dot_r_0 >= tol ** 2:
        Ap = A(p, N)
        alpha = r_dot_r / np.sum(Ap * p)
        x = x + alpha * p
        r = r - alpha * Ap
        # print(type(x[0,0]),type(r[0,0]),type(p[0,0]))
        r_dot_r_new = np.sum(r * r)
        beta = r_dot_r_new / r_dot_r
        p = r + beta * p
        # update for iteration
        r_dot_r = r_dot_r_new
        i += 1

        # restart the problem after N iterations and convergence not improving
        if i % N == 0 and beta > 0.98:
            r = p = rhs - A(x, N)
        # p[:,[0,N]] = 0
        if conv_hist:
            hist.append(np.sqrt(r_dot_r / r_dot_r_0))
        if maxiter is not None and i >= maxiter:
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
