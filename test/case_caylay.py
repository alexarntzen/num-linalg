import numpy as np

from test.case_matrix_ode import generate_heat_equation
from linalg.integrate import X_proj
from linalg.helpers import truncated_svd


def get_FU(m, k):
    A_0, A, A_dot = generate_heat_equation(n=m, m=m, k=k)
    t = 0
    Y = truncated_svd(A_0, k)
    U = Y[0]
    F_U, F_S, F_V = X_proj(X=A_dot, Y=Y, t=t)

    return F_U, U


def get_CD(F, U):
    C = np.concatenate((F, -U), axis=1)
    D = np.concatenate((U, F), axis=1)
    return C, D


def get_B(C, D):
    return C @ D.T


def get_FUCDB(m, k):
    F, U = get_FU(m, k)
    C, D = get_CD(F, U)
    B = get_B(C, D)
    return F, U, C, D, B
