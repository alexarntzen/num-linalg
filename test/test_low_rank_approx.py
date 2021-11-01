import unittest
import numpy as np
from linalg.dynamic_approximation import (
    cayley_map_simple,
    cayley_map_efficient,
    cayley_map_plus,
)

from test.case_matrix_ode import HeatEquation


class TestMatrixOde(unittest.TestCase):
    def testSolutionMap(self):
        eq = HeatEquation(5, 5, k=3)
        np.testing.assert_allclose(eq.A(0), eq.A_0)


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
