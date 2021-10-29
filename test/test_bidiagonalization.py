import unittest

import numpy as np
from linalg.bidiagonalization import lanczos_bidiag, lanczos_bidiag_reorth
from linalg.helpers import make_bidiagonal, get_best_approx


class TestLanczos(unittest.TestCase):
    def test_bidiagonal_approx(self):
        for M in [32, 54, 128]:
            for N in [32, 64, 128]:
                k = min(M,N)
                k //=2
                print(f"\nTest appoximatons for N={N}, k={k}")
                A = np.random.rand(M, N)
                A_norm = np.linalg.norm(A, ord="fro")
                b = np.random.rand(M)

                P, Q, alpha, beta = lanczos_bidiag(A, k, b)
                B = make_bidiagonal(alpha, beta)
                dA = np.linalg.norm(A - P @ B @ Q.T, ord="fro")
                print("Lanchos delta: ", dA)
                # self.assertGreater(A_norm, dA)

                P, Q, alpha, beta = lanczos_bidiag_reorth(A, k, b)
                B = make_bidiagonal(alpha, beta)
                dA = np.linalg.norm(A - P @ B @ Q.T, ord="fro")
                print("Lanczo reorth delta: ", dA)
                self.assertGreater(A_norm, dA)

                A_k = get_best_approx(A, k)
                dA = np.linalg.norm(A - A_k, ord="fro")
                print("Best approx delta: ", dA)
                self.assertGreater(A_norm, dA)

    if __name__ == "__main__":
        unittest.main()
