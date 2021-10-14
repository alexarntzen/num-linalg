import numpy as np
from numalg.iterative import cg, weighted_jacobi
from numalg.laplacian import D_inv, J_w


def mgv_poisson(u_0, rhs, N, nu1, nu2, level, max_level: int, A: callable):
    """
    the fucntion performs the function mgv(u0,rhs,N,nu1,nu2,level,max_level) performs
    one multigrid V-cycle on the 2D Poisson problem on the unit
    square [0,1]x[0,1] with initial guess u0 and righthand side rhs.

    input:
         rhs       - righthand side
         N         - u0 is a (N+1)x(N+1) matrix
         nu1       - number of presmoothings
         nu2       - number of postsmoothings
         level     - current level
         max_level - total number of levels"""
    assert (
        N % 2 ** max_level == 0
    ), "Number of grid nodes along each axis is not divisable by 2**max_level"
    if level == max_level:
        u = cg(u_0, rhs, N, 1e-13, 500)
    else:
        u = weighted_jacobi(u_0, rhs, w=2 / 3, N=N, nu=nu1, J_w=J_w, D_inv=D_inv)
        rf = A(u) - rhs  # compute residual
        rc = restriction(rf, N)  # restriction
        ec = mgv_poisson(
            np.zeros((N // 2 + 1, N // 2 + 1)),
            rc,
            N // 2,
            nu1,
            nu2,
            level + 1,
            max_level,
        )
        ef = interpolation(ec, int(N / 2))
        u = u + ef
        u = weighted_jacobi(u_0, rhs, w=2 / 3, N=N, nu=nu2, J_w=J_w, D_inv=D_inv)
    return u


def restriction(v_h, N):

    index_h = np.arange(2, N, 2)
    index_2h = np.arange(1, N // 2)

    index_h_full = np.arange(0, N + 1, 2)
    ixy_2h = np.ix_(index_2h, index_2h)

    # take care of boundary values
    v_2h = v_h[np.ix_(index_h_full, index_h_full)]
    v_2h[ixy_2h] = 0  # interior points

    stencil = [0.25, 0.5, 0.25]
    rel_index = [-1, 0, 1]
    for i, Ii in zip(rel_index, stencil):
        for j, Ij in zip(rel_index, stencil):
            ixy_h = np.ix_(index_h + i, index_h + j)
            v_2h[ixy_2h] += Ij * Ii * v_h[ixy_h]

    return v_2h


def interpolation(v_2h, N):

    # include one extra row to accomodate the stencil
    v_h_extra = np.zeros((2 * N + 3, 2 * N + 3))
    index_extra = np.arange(1, 2 * N + 2)
    index_extra_even = np.arange(1, 2 * N + 3, 2)
    ixy_extra = np.ix_(index_extra, index_extra)

    stencil = [0.5, 1, 0.5]
    rel_index = [-1, 0, 1]
    for i, Ii in zip(rel_index, stencil):
        for j, Ij in zip(rel_index, stencil):
            # distribute all interior points according to stencil
            ixy_h_s = np.ix_(index_extra_even + i, index_extra_even + j)
            v_h_extra[ixy_h_s] += Ij * Ii * v_2h

    return v_h_extra[ixy_extra]