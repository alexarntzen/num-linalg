import unittest
import numpy as np
from linalg.cayley_map import (
    cayley_map_simple,
    cayley_map_efficient,
    cayley_map_plus,
)
from linalg.integrate import matrix_ode_simple
from linalg.bidiagonalization import lanczos_bidiag_reorth
from linalg.helpers import make_bidiagonal, multiply_factorized

from test.case_matrix_ode import HeatEquation


class TestMatrixOde(unittest.TestCase):
    def testSolutionMap(self):
        eq = HeatEquation(5, 5, k=3)
        np.testing.assert_allclose(eq.A(0), eq.A_0)

    def test_integrator(self):
        m = 5
        k = 3
        eq = HeatEquation(m, m, k=k)

        # b = np.random.rand(m)
        # U_0, V_0, alpha, beta = lanczos_bidiag_reorth(eq.A_0, k, b)
        # S_0 = make_bidiagonal(alpha, beta).toarray()

        U_0, V_0 = eq.C_0, eq.D_0
        S_0 = np.eye(k)

        Y, T = matrix_ode_simple(0, 1, Y_0=(U_0, S_0, V_0), X=eq.A_dot)
        Y_mat = multiply_factorized(*Y[-1])

        A_1 = eq.A(1)
        A_1_fro = np.linalg.norm(A_1, ord="fro")
        fro_diff = np.linalg.norm(Y_mat - A_1, ord="fro")

        # this is apparently wrong
        # something wanky is happening here its integrating in the opposite way :(
        self.assertLess(fro_diff, A_1_fro)


class TestCayley(unittest.TestCase):
    def test_caylay(self):
        for m in [10]:
            k = m // 2

            # make a test problem
            H = np.random.rand(m, k)
            G = np.random.rand(m, k)
            U, _ = np.linalg.qr(H)
            F = (np.eye(m) - U @ U.T) @ G
            np.testing.assert_allclose(np.eye(k), U.T @ U, atol=1e-14)

            # create C and D
            C = np.concatenate((F, -U), axis=1)
            D = np.concatenate((U, F), axis=1)
            np.testing.assert_allclose(C @ D.T, F @ U.T - U @ F.T, atol=1e-14)

            S = cayley_map_simple(C @ D.T)
            E = cayley_map_efficient(C, D)
            P = cayley_map_plus(F, U)

            # test that cayley maps give same answer
            np.testing.assert_allclose(S, E)
            np.testing.assert_allclose(S, P)


if __name__ == "__main__":
    unittest.main()
