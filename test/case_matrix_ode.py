"""Implements the test cases from the report"""
import scipy.linalg as sl
import numpy as np

laplace_kernel_1d = np.array([1, -2, -1])


def raveldot(A: np.ndarray, x: np.ndarray):
    m, n = x.shape
    Ax_vec = A @ x.ravel()
    Ax = Ax_vec.reshape(m, n)
    return Ax


def get_laplacians(m, n):
    """Assumes homogeneous Dirichlet boundary conditions on outside nodes
    on unit square"""
    assert n > 2 and m > 2, "cannot make laplacian with so few nodes"
    # dxx = sl.convolution_matrix(laplace_kernel_1d * m ** 2, m, mode="same")
    # dyy = sl.convolution_matrix(laplace_kernel_1d * n ** 2, n, mode="same")
    dxx = sl.convolution_matrix(laplace_kernel_1d, m, mode="same")
    dyy = sl.convolution_matrix(laplace_kernel_1d, n, mode="same")
    return dxx, dyy


def kron_sum(A, B):
    m = A.shape[0]
    n = B.shape[0]
    left = np.kron(A, np.eye(n))
    right = np.kron(np.eye(m), B)
    return left + right


class HeatEquation:
    """d/dt A(t)= D @ A(t)"""

    @staticmethod
    def generate_case(n, m, k):
        dxx, dyy = get_laplacians(m, n)
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


class FirstExample:
    """d/dt A(t)= D @ A(t)"""

    @classmethod
    def generate_case(cls, eps):
        n = 100
        T_1 = cls.get_skew_symmetric(n, [1])
        T_2 = cls.get_skew_symmetric(n, [0.5, 1])
        A_1 = cls.get_A(eps=eps)
        A_2 = cls.get_A(eps=eps)

        A_0 = A_1 + A_2

        def Q_1(t):
            return sl.expm(T_1 * t)

        def Q_1_dot(t):
            return T_1 @ Q_1(t)

        def Q_2(t):
            return sl.expm(T_2 * t)

        def Q_2_dot(t):
            return T_2 @ Q_2(t)

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

    @staticmethod
    def get_skew_symmetric(n, coeffs):
        T = np.zeros((n, n))
        for k, coeff in enumerate(coeffs):
            i = k + 1
            np.fill_diagonal(T[i:, :-i], coeff)
            np.fill_diagonal(T[:-i, i:], -coeff)

        return T

    @staticmethod
    def get_A(eps=1e-3):
        A_ = np.eye(10)
        A_delta = np.random.rand(10, 10) * 0.5
        A = np.random.rand(100, 100) * eps
        A[:10, :10] = A_ + A_delta
        return A
