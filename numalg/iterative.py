import numpy as np


def cg(A, x_0, rhs, N, tol=1e-5, maxiter=None):
    """Solving -Lu*h**2 = f*h**2*1(interior) + g*boundary()

    Assuming all numbers are real
    """

    r = rhs - A(x_0, N)
    p = r
    u = x_0
    r_dot_r_0 = np.tensordot(r, r, 2)  # two norm **2
    r_dot_r = r_dot_r_0
    i = 0
    while r_dot_r / r_dot_r_0 >= tol ** 2:
        # if i % 100 == 0:
        #     print(i, ((r_dot_r / r_dot_r_0) ** 0.5))

        Ap = A(p, N)
        alpha = r_dot_r / np.tensordot(Ap, p, 2)

        u = u + alpha * p
        r = r - alpha * Ap
        r_dot_r_new = np.tensordot(r, r, 2)
        beta = r_dot_r_new / r_dot_r
        p = r + beta * p

        # update for iteration
        r_dot_r = r_dot_r_new

        i += 1
        if maxiter is not None and i > maxiter:
            break
    # print("res: ",  r_dot_r / r_dot_r_0)
    return u


def weighted_jacobi(x_0, rhs, N, w, nu, J_w: callable, D_inv):
    """Perform weighted jacobi iteration"""
    f_w = w * D_inv(rhs, N=N)
    x = x_0  # so we do not change x0
    for _ in range(nu):
        x = J_w(x, N, w) + f_w

    return x
