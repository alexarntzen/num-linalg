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
    dxx = sl.convolution_matrix(laplace_kernel_1d * m ** 2, m, mode="same")
    dyy = sl.convolution_matrix(laplace_kernel_1d * n ** 2, n, mode="same")
    return dxx, dyy


def kron_sum(A, B):
    m = A.shape[0]
    n = B.shape[0]
    left = np.kron(A, np.eye(n))
    right = np.kron(np.eye(m), B)
    return left + right


class HeatEquation:
    """d/dt A(t)= D @ A(t)"""

    def __init__(self, n, m, k):
        self.dxx, self.dyy = get_laplacians(m, n)
        self.D = kron_sum(self.dxx, self.dyy)
        self.B = self.D
        C_0 = np.random.rand(m, k)
        D_0 = np.random.rand(m, k)
        self.A_0 = C_0 @ D_0.T

    def A_dot(self, t):
        return raveldot(self.B, self.A(t))

    def A(self, t) -> np.ndarray:
        """since exp(N (+) M) =  exp(N) (x) exp(M)"""
        expXt = sl.expm(self.dxx * t)
        expYt = sl.expm(self.dyy * t)
        expBt = np.kron(expXt, expYt)
        return raveldot(expBt, self.A_0)


class SchoedingerEquation:
    """d/dt A(t)= i*D @ A(t)"""

    def __init__(self, n, m, k):
        self.dxx, self.dyy = get_laplacians(m, n)
        self.D = kron_sum(self.dxx, self.dyy)
        self.B = 1j * self.D
        C_0 = np.random.rand(m, k)
        D_0 = np.random.rand(m, k)
        self.A_0 = C_0 @ D_0.T

    def A_dot(self, t):
        return raveldot(self.B, self.A(t))

    def A(self, t) -> np.ndarray:
        """since exp(N (+) M) =  exp(N) (x) exp(M)"""
        expXt = sl.expm(self.dxx * t * 1j)
        expYt = sl.expm(self.dyy * t * 1j)
        expBt = np.kron(expXt, expYt)
        return raveldot(expBt, self.A_0)
