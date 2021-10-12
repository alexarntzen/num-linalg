import unittest
import numpy as np

from numalg.laplacian import L
from numalg.cg import cg



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
        for N in [32, 64, 128]:

            rhs = self.get_rhs(N)
            U_sol = self.get_u(N)
            U_0 = np.random.rand(N + 1, N + 1)
            U_num = cg(L, U_0, rhs=rhs, N=N, tol=2 / N ** 1.5)
            DU = U_num - U_sol
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
