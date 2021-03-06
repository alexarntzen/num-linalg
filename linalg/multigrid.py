import numpy as np
from linalg.iterative import cg, weighted_jacobi
from linalg.laplacian import neg_discrete_laplacian, D_inv, J_w


def multigrid_minus_poisson(
    x_0,
    rhs,
    N,
    nu1,
    nu2,
    level=0,
    max_level: int = 1,
    tol=1e-13,
    maxiter=None,
    conv_hist=False,
):
    """Multigrid with multiple v-cycles

    performs multigrid V-cycles until convergence
    """

    x = x_0
    res_0 = np.linalg.norm(rhs - neg_discrete_laplacian(x, N), ord="fro")
    res = res_0
    i = 0
    hist = list()
    while res / res_0 >= tol:
        x = mgv_minus_poisson(x, rhs, N, nu1, nu2, level, max_level)
        i += 1
        if maxiter is not None and i >= maxiter:
            break
        res = np.linalg.norm(rhs - neg_discrete_laplacian(x, N), ord="fro")
        if conv_hist:
            hist.append(res / res_0)
    if conv_hist:
        return x, hist
    else:
        return x


def mgv_minus_poisson(x_0, rhs, N, nu1, nu2, level=0, max_level=1):
    """
    the function mgv(u0,rhs,N,nu1,nu2,level,max_level) performs
    one multigrid V-cycle on the 2D Poisson problem on the unit
    square [0,1]x[0,1] with initial guess u0 and righthand side rhs.

    input:
         rhs       - righthand side
         N         - u0 is a (N+1)x(N+1) matrix
         nu1       - number of presmoothings
         nu2       - number of postsmoothings
         level     - current level
         max_level - total number of levels"""
    x = x_0
    assert (
        N % 2 ** (max_level - level) == 0
    ), "Number of grid nodes along each axis is not divisible by 2**max_level"
    # check if reach bottom of V-cycle
    if level == max_level:
        x = cg(A=neg_discrete_laplacian, x_0=x, rhs=rhs, N=N, tol=1e-12, maxiter=500)
    else:
        x = weighted_jacobi(x_0=x, rhs=rhs, w=2 / 3, N=N, nu=nu1, J_w=J_w, D_inv=D_inv)
        r_h = rhs - neg_discrete_laplacian(x, N)  # compute residual
        r_2h = restriction(r_h, N)
        e_2h = mgv_minus_poisson(
            np.zeros((N // 2 + 1, N // 2 + 1)),
            r_2h,
            N // 2,
            nu1,
            nu2,
            level + 1,
            max_level,
        )  # recursively restart at lover level
        e_h = interpolation(e_2h, int(N / 2))
        x = x + e_h
        x = weighted_jacobi(x_0=x, rhs=rhs, w=2 / 3, N=N, nu=nu2, J_w=J_w, D_inv=D_inv)
    return x


def restriction(v_h, N):
    """Restriction operator for multigrid
    Even indexed boundary values are preserved.
    Interior points are restricted according to the full weighted stencil"""
    assert N % 2 == 0
    "N must be even"
    index_h = np.arange(2, N, 2)
    index_2h = np.arange(1, N // 2)

    index_h_full = np.arange(0, N + 1, 2)
    ixy_2h = np.ix_(index_2h, index_2h)

    # take care of boundary values
    v_2h = v_h[np.ix_(index_h_full, index_h_full)]

    # interior points
    v_2h[ixy_2h] = 0

    # use stencil on interior points tensorised to 2D
    stencil = [0.25, 0.5, 0.25]
    rel_index = [-1, 0, 1]
    for i, Ii in zip(rel_index, stencil):
        for j, Ij in zip(rel_index, stencil):
            ixy_h = np.ix_(index_h + i, index_h + j)
            v_2h[ixy_2h] += Ij * Ii * v_h[ixy_h]

    return v_2h


def interpolation(v_2h, N):
    """Interpolation operator for multigrid
    This function uses extra nodes outside the boundary to accommodate the stencil
    """

    # define v_h with extra nodes
    v_h_extra = np.zeros((2 * N + 3, 2 * N + 3))

    # find indexes for different parts of v_h_extra
    index_extra = np.arange(1, 2 * N + 2)
    index_extra_even = np.arange(1, 2 * N + 3, 2)
    ixy_extra = np.ix_(index_extra, index_extra)

    # use stencil in reverse tensorised to 2D
    stencil = [0.5, 1, 0.5]
    rel_index = [-1, 0, 1]
    for i, Ii in zip(rel_index, stencil):
        for j, Ij in zip(rel_index, stencil):
            # distribute all interior points according to stencil
            # this will exclude
            ixy_h_s = np.ix_(index_extra_even + i, index_extra_even + j)
            v_h_extra[ixy_h_s] += Ij * Ii * v_2h

    # return v_h, exclude the extra rows
    return v_h_extra[ixy_extra]
