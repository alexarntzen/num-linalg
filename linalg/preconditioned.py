import numpy as np
from linalg.laplacian import neg_discrete_laplacian
from linalg.multigrid import mgv_minus_poisson


def mgv_conditioned_cg_minus_poisson(
    x_0,
    rhs,
    N,
    nu1,
    nu2,
    tol=1e-5,
    maxiter=None,
    conv_hist=False,
):

    cond_kwargs = dict(
        x_0=np.zeros((N + 1, N + 1)),
        N=N,
        nu1=nu1,
        nu2=nu2,
        level=1,
        max_level=1,
    )
    return preconditioned_cg(
        A=neg_discrete_laplacian,
        x_0=x_0,
        rhs=rhs,
        N=N,
        M_inv=mgv_minus_poisson,
        tol=tol,
        maxiter=maxiter,
        conv_hist=conv_hist,
        cond_kwargs=cond_kwargs,
    )


def preconditioned_cg(
    A: callable,
    x_0,
    rhs,
    N: int,
    M_inv: callable,
    tol=1e-12,
    maxiter=None,
    conv_hist=False,
    cond_kwargs=None,
):
    """Solving -Lu*h**2 = f*h**2*1(interior) + g*boundary()

    Assuming all numbers are real
    """
    if cond_kwargs is None:
        cond_kwargs = dict()
    r = rhs - A(x_0, N)
    z = M_inv(rhs=r, **cond_kwargs)
    p = z
    x = x_0

    hist = list()
    r_dot_r_0 = np.tensordot(r, r, 2)  # two norm **2
    r_dot_r = r_dot_r_0
    r_dot_z = np.tensordot(r, z, 2)  # two norm **2
    i = 0
    while r_dot_r / r_dot_r_0 >= tol ** 2:
        Ap = A(p, N)
        alpha = r_dot_z / np.tensordot(Ap, p, 2)

        x = x + alpha * p
        r = r - alpha * Ap
        z = M_inv(rhs=r, **cond_kwargs)
        r_dot_r_new = np.tensordot(r, r, 2)
        r_dot_z_new = np.tensordot(r, z, 2)
        beta = r_dot_z_new / r_dot_z
        p = z + beta * p

        # update for iteration
        r_dot_r = r_dot_r_new
        r_dot_z = r_dot_z_new

        i += 1
        if conv_hist:
            hist.append(np.sqrt(r_dot_r / r_dot_r_0))
        if maxiter is not None and i > maxiter:
            break
        # print("res: ",  r_dot_r / r_dot_r_0)
    if conv_hist:
        return x, hist
    else:
        return x
