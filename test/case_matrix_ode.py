"""Implements the test cases from the report"""
import scipy.linalg as sl
import numpy as np


def generate_heat_equation(n, m, k, boundary="periodic"):
    dxx, dyy = get_laplacian(m, boundary=boundary), get_laplacian(n, boundary=boundary)
    D = kron_sum(dxx, dyy)
    B = D
    C_0 = np.random.rand(m, k)
    D_0 = np.random.rand(m, k)
    A_0 = C_0 @ D_0.T

    def A(t) -> np.ndarray:
        """since exp(N (+) M) =  exp(N) (x) exp(M)"""
        expXt = sl.expm(dxx * t)
        expYt = sl.expm(dyy * t)
        expBt = np.kron(expXt, expYt)
        return raveldot(expBt, A_0)

    def A_dot(t) -> np.ndarray:
        A_dot = raveldot(B, A(t))
        return A_dot

    return A_0, A, A_dot


def generate_first_example(eps=1e-3):
    n = 100
    T_1 = get_skew_symmetric(n, [1])
    T_2 = get_skew_symmetric(n, [0.5, 1])
    A_1 = get_A(eps=eps)
    A_2 = get_A(eps=eps)

    A_0 = A_1 + A_2

    def Q_1(t):
        return sl.expm(T_1 * t)

    def Q_2(t):
        return sl.expm(T_2 * t)

    def A(t):
        return Q_1(t) @ (A_1 + np.exp(t) * A_2) @ Q_2(t).T

    def A_dot(t):
        Q_1_t = Q_1(t)
        Q_2_t = Q_2(t)
        A_t = A(t)

        d_one = T_1 @ A_t
        d_two = A_t @ T_2.T
        d_tree = Q_1_t @ (np.exp(t) * A_2) @ Q_2_t.T

        return d_one + d_two + d_tree

    return A_0, A, A_dot


def generate_second_example(eps=1e-1):
    n = 100
    T_1 = get_skew_symmetric(n, [1])
    T_2 = get_skew_symmetric(n, [0.5, 1])
    A_1 = get_A(eps=eps)
    A_2 = get_A(eps=eps)

    A_0 = A_1 + A_2

    def Q_1(t):
        return sl.expm(T_1 * t)

    def Q_2(t):
        return sl.expm(T_2 * t)

    def A(t):
        return Q_1(t) @ (A_1 + np.cos(t) * A_2) @ Q_2(t).T

    def A_dot(t):
        Q_1_t = Q_1(t)
        Q_2_t = Q_2(t)
        A_t = A(t)

        d_one = T_1 @ A_t
        d_two = A_t @ T_2.T
        d_tree = Q_1_t @ (-np.sin(t) * A_2) @ Q_2_t.T

        return d_one + d_two + d_tree

    return A_0, A, A_dot


def raveldot(A: np.ndarray, x: np.ndarray):
    m, n = x.shape
    Ax_vec = A @ x.ravel()
    Ax = Ax_vec.reshape(m, n)
    return Ax


def get_laplacian(n, boundary="periodic"):
    """Assumes homogeneous Dirichlet boundary conditions on outside nodes
    on unit square"""
    assert n > 2, f"cannot make laplacian with {n} few nodes"
    # zero is homogeneous dirichlet
    assert boundary in ("zero", "periodic")
    # dxx = sl.convolution_matrix(laplace_kernel_1d * m ** 2, m, mode="same")
    # dyy = sl.convolution_matrix(laplace_kernel_1d * n ** 2, n, mode="same")

    laplace_kernel_1d = np.array([1, -2, 1])
    dxx = sl.convolution_matrix(laplace_kernel_1d * n ** 2, n, mode="full")

    if boundary == "periodic":
        # the boundary points are assumed to be the ones on the other side
        dxx[1] += dxx[-1]
        dxx[-2] += dxx[0]
    # exclude boundary points
    dxx = dxx[1:-1]
    return dxx


def kron_sum(A, B):
    m = A.shape[0]
    n = B.shape[0]
    left = np.kron(A, np.eye(n))
    right = np.kron(np.eye(m), B)
    return left + right


def get_skew_symmetric(n, coeffs):
    T = np.zeros((n, n))
    for k, coeff in enumerate(coeffs):
        i = k + 1
        np.fill_diagonal(T[i:, :-i], coeff)
        np.fill_diagonal(T[:-i, i:], -coeff)

    return T


def get_A(eps):
    A_ = np.eye(10)
    A_delta = np.random.rand(10, 10) * 0.5
    A = np.random.rand(100, 100) * eps
    A[:10, :10] = A_ + A_delta
    return A
