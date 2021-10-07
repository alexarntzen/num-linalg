import numpy as np


def L(u, index):
    """Compute discrete laplacian times h**2"""
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index - 1, index)
    ixp_y = np.ix_(index + 1, index)
    ix_ym = np.ix_(index, index - 1)
    ix_yp = np.ix_(index, index + 1)

    Lu = -4 * u[ixy] + u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp]
    return Lu


def cg(u_0, rhs, N, tol):
    """Solving Lu*h**2 = f*h**2

    rhs must f*h**2"""
    index = np.arange(1, N)
    u = u_0
    r = rhs - L(u, index)
    p = r
    r_dot_r = np.tensordot(r, r, 2)
    while True:
        Lp = L(p, index)
        alpha = r_dot_r / np.tensordot(Lp, p)

        u = u - alpha * p
        r = r - alpha * Lp
        r_dot_r_new = np.tensordot(r, r, 2)
        beta = r_dot_r / r_dot_r_new
        p = r + beta * p

        r_dot_r = r_dot_r_new

        break
    return u_0
