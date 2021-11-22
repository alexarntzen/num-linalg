import numpy as np
from linalg.cayley_map import cayley_map_plus
from linalg.helpers import multiply_factorized

h_default = 1e-6
TOL_default = 1e-5


def matrix_ode_simple(
    t_0, t_f, Y_0: tuple, X: callable, h_0=h_default, TOL=TOL_default, verbose=False
):
    """X is the vector field, Y is a tuple:= (U, S, V)"""

    # check that initial  factorization is valid
    U_0, _, V_0 = Y_0
    m, k = U_0.shape
    assert (
        np.linalg.norm(U_0.T @ U_0 - np.eye(k), ord="fro") < TOL
    ), "factorization not valid"
    assert (
        np.linalg.norm(V_0.T @ V_0 - np.eye(k), ord="fro") < TOL
    ), "factorization not valid"

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
            if j % 100 == 0:
                print(f"step t={t}")
        # orthogonality check could also be here
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

    # calculate current vector field
    F_0 = X_proj(X, Y=Y, t=t)

    # take half step with F_0
    Y_05 = caylay_lie_step(Y=Y, F=F_0, h=0.5 * h)

    # calculate vector field at Y_05
    F_05 = X_proj(X=X, Y=Y_05, t=(t + 0.5 * h))

    # estimate next Y_i with RK1
    Y_1_est = caylay_lie_step(Y, F=F_0, h=h)

    # Take step at Y_0 with "frozen" vector field at Y_05
    Y_1 = caylay_lie_step(Y=Y, Y_frozen=Y_05, F=F_05, h=h)

    if step_control:
        return Y_1, Y_1_est
    else:
        return Y_1


def rk_1_step(Y: tuple, X: callable, t, h) -> tuple:
    """Y = (S, U ,V)"""
    F = X_proj(X=X, Y=Y, t=t)
    Y_new = caylay_lie_step(Y=Y, F=F, h=h)
    return Y_new, F


def X_proj(X: callable, Y, t):
    """g can have many representations, feks be {F=F, U=U)"""
    # split the tuple
    U, S, V = Y

    # calculate variables that are used multiple times
    m, _ = U.shape
    n, _ = V.shape
    I_m = np.eye(m)
    I_n = np.eye(n)
    A_dot = X(t)
    S_inv = np.linalg.inv(S)

    # calculate projected vector field
    F_U = (I_m - U @ U.T) @ A_dot @ V @ S_inv
    F_S = U.T @ A_dot @ V
    F_V = (I_n - V @ V.T) @ A_dot.T @ U @ S_inv.T

    return F_U, F_S, F_V


def caylay_lie_step(Y: tuple, F: tuple, h: float, Y_frozen=None):
    U, S, V = Y
    U_f, S_f, V_f = Y_frozen if Y_frozen is not None else Y
    F_U, F_S, F_V = F
    U_new = cayley_map_plus(F=(h * F_U), U=U_f) @ U
    S_new = S + h * F_S
    V_new = cayley_map_plus(F=(h * F_V), U=V_f) @ V
    return U_new, S_new, V_new


def get_y_dot(A_dot: callable, Y: np.ndarray, t):
    U_dot, S_dot, V_dot = X_proj(X=A_dot, Y=Y, t=t)
    U, S, V = Y
    Y_dot = U_dot @ S @ V.T + U @ S_dot @ V.T + U @ S @ V_dot.T
    return Y_dot
