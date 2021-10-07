import unittest
import numpy as np
from numalg.poisson import cg


class TestCase1(unittest.TestCase):
    @staticmethod
    def get_f(N):
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = 20 * np.pi ** 2 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        return f

    @staticmethod
    def get_u(N):
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")

        U = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        return U

    def test_cg(self):
        for N in [32, 64, 128]:
            rhs = self.get_f(N)
            U_sol = self.get_u(N)
            U_0 = np.random.rand(N + 1, N + 1)

            U_num = cg(U_0, rhs=rhs, N=N, tol=10e-12)

            np.testing.assert_allclose(U_num, U_sol)


if __name__ == "__main__":
    unittest.main()
