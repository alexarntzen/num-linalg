import unittest

import numpy as np

from linalg.cayley_map import cayley_map_simple, cayley_map_efficient, cayley_map_plus

from test.case_caylay import get_FUCDB


class TestCayley(unittest.TestCase):
    def test_caylay(self):
        for m in [
            64,
            128,
            256,
        ]:
            tol = dict(atol=1e-8, rtol=1e-6)
            k = m // 2
            print(f"\nTest Caylay maps are equal m={m}, k={k}:")
            # make a test problem
            F, U, C, D, B = get_FUCDB(m, k)
            np.testing.assert_allclose(np.eye(k), U.T @ U, **tol)
            np.testing.assert_allclose(C @ D.T, F @ U.T - U @ F.T, **tol)

            S = cayley_map_simple(C @ D.T)
            E = cayley_map_efficient(C, D)
            P = cayley_map_plus(F, U)

            # test that cayley maps give same answer
            np.testing.assert_allclose(S, E, **tol)
            np.testing.assert_allclose(S, P, **tol)

            # test orthogonality
            np.testing.assert_allclose(np.eye(m), S.T @ S, **tol)
            np.testing.assert_allclose(np.eye(m), E.T @ E, **tol)
            np.testing.assert_allclose(np.eye(m), P.T @ P, **tol)


if __name__ == "__main__":
    unittest.main()
