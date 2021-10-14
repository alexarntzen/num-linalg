import unittest
import numpy as np

from numalg.laplacian import L, D_inv, J_w
from numalg.iterative import cg, weighted_jacobi


class TestCase1(unittest.TestCase):
    @staticmethod
    def get_f(N):
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = 20 * np.pi ** 2 * np.sin(2 * np.pi * X) * np.sin(4 * np.pi * Y)
        return f

    @staticmethod
    def get_u(N):
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")

        U = np.sin(2 * np.pi * X) * np.sin(4 * np.pi * Y)
        return U

    def get_rhs(self, N):
        # assuming g=U on the boundary, which is true

        ix = np.arange(1, N)
        ixy = np.ix_(ix, ix)

        rhs = self.get_u(N)  # = g
        rhs[ixy] = self.get_f(N)[ixy] / N ** 2

        return rhs

    def test_cg(self):
        print("\ncg:" )
        for N in [32, 64, 128]:

            rhs = self.get_rhs(N)
            U_sol = self.get_u(N)
            U_0 = np.random.rand(N + 1, N + 1)
            U_num = cg(L, U_0, rhs=rhs, N=N, tol=1e-5, maxiter=500)
            DU = U_num - U_sol
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.1)
            print(f"\ndetla_{N}: {np.max(np.abs(DU))}")

    def test_weighted_jacobi(self):
        print("\nJacobi:" )
        for N in [32, 64, 128]:

            rhs = self.get_rhs(N)
            U_sol = self.get_u(N)
            U_0 = np.random.rand(N + 1, N + 1)
            U_num = weighted_jacobi(U_0, rhs, N, w=2/3, nu=25*N, J_w=J_w,D_inv=D_inv)
            DU = U_num - U_sol
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.5)
            print(f"\ndetla_{N}: {np.max(np.abs(DU))}")


if __name__ == "__main__":
    unittest.main()
