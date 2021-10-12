import numpy as np


def A(u, N):
    """Compute negative discrete laplacian times h**2 + id on the boundaryt """
    index = np.arange(1, N)
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index - 1, index)
    ixp_y = np.ix_(index + 1, index)
    ix_ym = np.ix_(index, index - 1)
    ix_yp = np.ix_(index, index + 1)

    # id for boundary points
    Lu = u.copy()

    # - \delta
    Lu[ixy] = -(u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp]) + 4 * u[ixy]
    return Lu


def cg(A, u_0, rhs, N, tol):
    """Solving -Lu*h**2 = f*h**2*1(interior) + g*boundary()

    Assuming all numbers are real
    """

    r = rhs - A(u_0, N)
    p = r
    u = u_0
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
    return u
