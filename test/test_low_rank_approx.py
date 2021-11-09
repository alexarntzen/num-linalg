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

from test.case_matrix_ode import generate_heat_equation, generate_first_example, generate_second_example


class TestMatrixOde(unittest.TestCase):
    def testSolutionMap(self):
        print("\nTesting Heat Eqation Case:")
        A_0, A, A_dot = generate_heat_equation(5, 5, k=3)
        np.testing.assert_allclose(A(0), A_0)

    def test_integrator_heat(self):
        m = 100
        k = 100
        t_f = 0.5
        print(f"\nTesting simple integrator on HeatEqation m={m}, k={k}:")

        # generate case and start conditions
        A_0, A, A_dot = generate_heat_equation(n=m, m=m, k=k)
        b = np.random.rand(m)
        U_0, V_0, alpha, beta = lanczos_bidiag_reorth(A_0, k, b)
        S_0 = make_bidiagonal(alpha, beta).toarray()

        # integrate
        Y, T = matrix_ode_simple(0, t_f, Y_0=(U_0, S_0, V_0), X=A_dot, TOL=1e-4)
        Y_mat = multiply_factorized(*Y[-1])

        # measure difference from last matrix
        A_1 = A(0.5)
        A_1_fro = np.linalg.norm(A_1, ord="fro")
        fro_diff = np.linalg.norm(Y_mat - A_1, ord="fro")

        self.assertLess(fro_diff, A_1_fro)

        print("Total: ", A_1_fro, " Residual", fro_diff)

    def test_integrator_eks_1(self):
        m = 100
        k = 20
        t_f = 0.5
        eps = 1e-3
        print(f"\nTesting simple integrator on first example eps={eps}:")

        # generate case and start conditions
        A_0, A, A_dot =generate_first_example(eps=eps)
        b = np.random.rand(m)
        U_0, V_0, alpha, beta = lanczos_bidiag_reorth(A_0, k, b)
        S_0 = make_bidiagonal(alpha, beta).toarray()

        # integrate
        Y, T = matrix_ode_simple(
            0, t_f, Y_0=(U_0, S_0, V_0), X=A_dot, TOL=1e-4, verbose=True
        )
        Y_mat = multiply_factorized(*Y[-1])

        # measure difference from last matrix
        A_1 = A(t_f)
        A_1_fro = np.linalg.norm(A_1, ord="fro")
        fro_diff = np.linalg.norm(Y_mat - A_1, ord="fro")

        # does not work on this case yet!
        self.assertLess(fro_diff, A_1_fro)

        print("Total: ", A_1_fro, " Residual", fro_diff)

    def test_integrator_eks_2(self):
        m = 100
        k = 20
        t_f = 0.5
        eps = 1e-1
        print(f"\nTesting simple integrator on first example eps={eps}:")

        # generate case and start conditions
        A_0, A, A_dot =generate_second_example(eps=eps)
        b = np.random.rand(m)
        U_0, V_0, alpha, beta = lanczos_bidiag_reorth(A_0, k, b)
        S_0 = make_bidiagonal(alpha, beta).toarray()

        # integrate
        Y, T = matrix_ode_simple(
            0, t_f, Y_0=(U_0, S_0, V_0), X=A_dot, TOL=1e-4, verbose=True
        )
        Y_mat = multiply_factorized(*Y[-1])

        # measure difference from last matrix
        A_1 = A(t_f)
        A_1_fro = np.linalg.norm(A_1, ord="fro")
        fro_diff = np.linalg.norm(Y_mat - A_1, ord="fro")

        # does not work on this case yet!
        self.assertLess(fro_diff, A_1_fro)

        print("Total: ", A_1_fro, " Residual", fro_diff)

class TestCayley(unittest.TestCase):
    def test_caylay(self):
        print("\nTest Caylay maps are equal:")
        for m in [10, 20, 30]:
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
