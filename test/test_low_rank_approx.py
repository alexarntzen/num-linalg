import unittest
import numpy as np
from linalg.integrate import matrix_ode_simple
from linalg.helpers import multiply_factorized, truncated_svd

from test.case_matrix_ode import (
    generate_heat_equation,
    generate_first_example,
    generate_second_example,
)


class TestMatrixOde(unittest.TestCase):
    def testSolutionMap(self):
        print("\nTesting Heat Equation Case:")
        A_0, A, A_dot = generate_heat_equation(25, 25, k=10)
        np.testing.assert_allclose(A(0), A_0)

    def test_integrator_heat(self):
        m = 10
        k = 7
        t_f = 1
        print(f"\nTesting simple integrator on HeatEqation m={m}, k={k}:")

        # generate case and start conditions
        A_0, A, A_dot = generate_heat_equation(n=m, m=m, k=k)
        Y_0 = truncated_svd(A_0, k=k)

        # integrate
        Y, T = matrix_ode_simple(0, t_f, Y_0=(Y_0), X=A_dot, TOL=1e-3, verbose=True)
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
        print(f"\nTesting simple integrator on first example m={m}, k={k}:")

        # generate case and start conditions
        A_0, A, A_dot = generate_first_example(eps=eps)
        Y_0 = truncated_svd(A_0, k)

        # integrate
        Y, T = matrix_ode_simple(0, t_f, Y_0=Y_0, X=A_dot, TOL=1e-4, verbose=True)
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
        print(f"\nTesting simple integrator on second example m={m}, k={k}:")

        # generate case and start conditions
        A_0, A, A_dot = generate_second_example(eps=eps)
        Y_0 = truncated_svd(A_0, k)

        # integrate
        Y, T = matrix_ode_simple(0, t_f, Y_0=Y_0, X=A_dot, TOL=1e-4, verbose=True)
        Y_mat = multiply_factorized(*Y[-1])

        # measure difference from last matrix
        A_1 = A(t_f)
        A_1_fro = np.linalg.norm(A_1, ord="fro")
        fro_diff = np.linalg.norm(Y_mat - A_1, ord="fro")

        # does not work on this case yet!
        self.assertLess(fro_diff, A_1_fro)

        print("Total: ", A_1_fro, " Residual", fro_diff)


if __name__ == "__main__":
    unittest.main()
