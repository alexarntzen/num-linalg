import numpy as np
from numalg.laplacian import minus_laplace
from numalg.multigrid import mgv_minus_poisson


def mgv_conditioned_cg_minus_possion(
    x_0, rhs, N, nu1, nu2, max_level, tol=1e-5, maxiter=None
):
    cond_kwargs = dict(
        u_0=np.zeros((N + 1, N + 1)),
        N=N,
        nu1=nu1,
        nu2=nu2,
        level=1,
        max_level=max_level,
    )
    return preconditioned_cg(
        A=minus_laplace,
        x_0=x_0,
        rhs=rhs,
        N=N,
        M_inv=mgv_minus_poisson,
        cond_kwargs=cond_kwargs,
        tol=tol,
        maxiter=maxiter,
    )


def preconditioned_cg(
    A: callable, x_0, rhs, N: int, M_inv: callable, cond_kwargs, tol=1e-5, maxiter=None
):
    """Solving -Lu*h**2 = f*h**2*1(interior) + g*boundary()

    Assuming all numbers are real
    """

    r = rhs - A(x_0, N)
    z = M_inv(rhs=r, **cond_kwargs)
    p = z
    u = x_0

    r_dot_r_0 = np.tensordot(r, r, 2)  # two norm **2
    r_dot_r = r_dot_r_0
    r_dot_z = np.tensordot(r, z, 2)  # two norm **2
    i = 0
    while r_dot_r / r_dot_r_0 >= tol ** 2:
        # if i % 100 == 0:
        #     print(i, ((r_dot_r / r_dot_r_0) ** 0.5))

        Ap = A(p, N)
        alpha = r_dot_z / np.tensordot(Ap, p, 2)

        u = u + alpha * p
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
        if maxiter is not None and i > maxiter:
            break
        # print("res: ",  r_dot_r / r_dot_r_0)
    return u
