import numpy as np

from linalg.cayley_map import cayley_map_plus
from linalg.helpers import multiply_factorized

h_default = 0.001
TOL_default = 0.001


def matrix_ode_simple(
    t_0, t_f, Y_0: tuple, X: callable, h_0=h_default, TOL=TOL_default, verbose=True
):
    """X is the vector field, Y is a tuple:= (U, S, V)"""
    # initialize variables
    h = h_0
    h_old = 0
    j = 0
    count = 0
    t = t_0
    T = [t_0]
    Y = [Y_0]
    while t <= t_f:
        Y_new, Y_new_est = rk_2_step(Y[j], X, h=h, t=t)
        Y_new_matrix = multiply_factorized(*Y_new)
        Y_new_est_matrix = multiply_factorized(*Y_new_est)

        sigma = np.linalg.norm(Y_new_matrix - Y_new_est_matrix, ord="fro")
        t_new, h_new = step_control(sigma=sigma, TOL=TOL, t=t, h=h)
        if verbose:
            print(t_new)
        if t_new < t and count <= 3:
            count += 1
            t = t_new
            h = h_new
            continue  # try to calculate again
        else:
            Y.append(Y_new)
            T.append(t_new)
            # take step
            j = j + 1
            t = t_new
            h_old, h = h, h_new
            count = 0

    # go one step back and calculate to the end
    t = t - h_old
    h = t_f - t

    Y[j], _ = rk_2_step(Y[j - 1], X, h=h, t=t)
    T[j] = t + h
    # Return a list of approximations and their times
    return Y, T


def step_control(sigma, TOL, t, h):
    if sigma > TOL:
        # take a step back and try again with half step
        t_new = t - h
        h_new = 0.5 * h
    else:
        if sigma > 0.5 * TOL:
            # reduce step size
            R = (TOL / sigma) ** (1 / 3)
            if R > 0.5 or R < 0.5:
                R = 0.7
        else:
            if sigma > 1 / 16 * TOL:
                # same step size
                R = 1
            else:
                # double step size
                R = 2
        t_new = t + h
        h_new = h * R
    return t_new, h_new


def rk_2_step(Y: tuple, X: callable, t, h, step_control=True):
    """Y = (S, U ,V)"""

    Y_05, F_0 = rk_1_step(Y, X, h=0.5 * h, t=t)
    Y_1_est = caylay_lie_step(Y, F=F_0, h=h)
    Y_1, F_05 = rk_1_step(Y_05, X, h=0.5 * h, t=(t + 0.5 * h))
    if step_control:
        return Y_1, Y_1_est
    else:
        return Y_1


def rk_1_step(Y: tuple, X: callable, t, h) -> tuple:
    """Y = (S, U ,V)"""
    F = X_proj(Y, X=X, t=t)
    Y_new = caylay_lie_step(Y=Y, F=F, h=h)
    return Y_new, F


def X_proj(Y, X: callable, t):
    """g can have many representations, feks be {F=F, U=U)"""
    # split the tuple
    U, S, V = Y

    # calculate variables that are used multiple times
    m, k = U.shape
    I_m = np.eye(m)
    A_dot = X(t)
    S_inv = np.linalg.inv(S)

    # calculate projected vector field
    F_U = (I_m - U @ U.T) @ A_dot @ V @ S_inv
    F_S = U.T @ A_dot @ V
    F_V = (I_m - V @ V.T) @ A_dot.T @ U @ S_inv.T

    return F_U, F_S, F_V


def caylay_lie_step(Y: tuple, F: tuple, h: float):
    U, S, V = Y
    F_U, F_S, F_V = F
    U_new = cayley_map_plus(F=(h * F_U), U=U) @ U
    S_new = S + h * F_S
    V_new = cayley_map_plus(F=(h * F_V), U=V) @ V
    return U_new, S_new, V_new
