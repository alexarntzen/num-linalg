import unittest
import numpy as np
from linalg.multigrid import interpolation, restriction


class TestResInt(unittest.TestCase):
    def test_restriction(self):
        for N in [32, 64, 128]:
            # has correct shape
            v_h = np.random.rand(N + 1, N + 1)
            v_2h = restriction(v_h, N)
            self.assertEqual(v_2h.shape, (N // 2 + 1, N // 2 + 1))

            # test two edges
            self.assertEqual(v_2h[0, 0], v_h[0, 0])
            self.assertEqual(v_2h[N // 2, N // 2], v_h[N, N])

            # constant function
            c_h = np.ones((N + 1, N + 1))
            c_2h = restriction(c_h, N)
            self.assertEqual(np.max(np.abs(c_2h - 1)), 0)

    def test_interpolation(self):
        for N in [32, 64, 128]:
            # N is the number of nodes in the coarse space

            index_h_even = np.arange(N + 1) * 2
            ixy_h = np.ix_(index_h_even, index_h_even)

            # has correct shape
            v_2h = np.random.rand(N + 1, N + 1)
            v_h = interpolation(v_2h, N)
            self.assertEqual(v_h.shape, (2 * N + 1, 2 * N + 1))

            # test two edges
            self.assertEqual(v_2h[0, 0], v_h[0, 0])
            self.assertEqual(v_2h[N, N], v_h[2 * N, 2 * N])

            # test even indexes are the same
            np.testing.assert_array_equal(v_h[ixy_h], v_2h)

            # constant function
            c_2h = np.ones((N + 1, N + 1))
            c_h = interpolation(c_2h, N)
            self.assertEqual(np.max(np.abs(c_h - 1)), 0)


if __name__ == "__main__":
    unittest.main()
