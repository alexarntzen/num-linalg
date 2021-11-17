import unittest

import numpy as np

from linalg.cayley_map import cayley_map_simple, cayley_map_efficient, cayley_map_plus

from test.case_caylay import get_FUCDB


class TestCayley(unittest.TestCase):
    def test_caylay(self):
        print("\nTest Caylay maps are equal:")
        for m in [10, 20, 30]:
            k = m // 2
            # make a test problem
            F, U, C, D, B = get_FUCDB(m, k)
            np.testing.assert_allclose(np.eye(k), U.T @ U, atol=1e-14)
            np.testing.assert_allclose(C @ D.T, F @ U.T - U @ F.T, atol=1e-14)

            S = cayley_map_simple(C @ D.T)
            E = cayley_map_efficient(C, D)
            P = cayley_map_plus(F, U)

            # test that cayley maps give same answer
            np.testing.assert_allclose(S, E)
            np.testing.assert_allclose(S, P)

            # test orthogonality
            np.testing.assert_allclose(np.eye(m), S.T @ S, atol=1e-14)
            np.testing.assert_allclose(np.eye(m), E.T @ E, atol=1e-14)
            np.testing.assert_allclose(np.eye(m), P.T @ P, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
