import unittest
import numpy as np

import test.case_one as c1
import test.case_two as c2

from numalg.laplacian import minus_laplace, D_inv, J_w
from numalg.iterative import cg, weighted_jacobi
from numalg.multigrid import mgv_minus_poisson, multigrid_minus_poisson
from numalg.preconditioned import mgv_conditioned_cg_minus_possion


class TestIterCaseOne(unittest.TestCase):
    def test_cg(self):
        print("\ncg:")
        for N in [32, 64, 128]:

            rhs = c1.get_rhs(N)
            U_sol = c1.get_u(N)
            U_0 = np.random.rand(N + 1, N + 1)
            U_num = cg(minus_laplace, U_0, rhs=rhs, N=N, tol=1e-5, maxiter=500)
            DU = U_num - U_sol
            print(f"detla_{N}: {np.max(np.abs(DU))}")
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.1)

    def test_weighted_jacobi(self):
        print("\nJacobi:")
        for N in [32, 64, 128]:

            rhs = c1.get_rhs(N)
            U_sol = c1.get_u(N)
            U_0 = c1.get_u_0(N)
            U_num = weighted_jacobi(
                U_0, rhs, N, w=2 / 3, nu=25 * N, J_w=J_w, D_inv=D_inv
            )
            DU = U_num - U_sol
            print(f"detla_{N}: {np.max(np.abs(DU))}")
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.5)

    def test_multigrid(self):
        print("\nMultigrid:")
        for N in [32, 64]:

            rhs = c1.get_rhs(N)
            U_sol = c1.get_u(N)
            U_0 = c1.get_u_0(N)
            U_num = mgv_minus_poisson(
                U_0, rhs, N, nu1=2000, nu2=2000, level=0, max_level=2
            )
            DU = U_num - U_sol
            print(f"detla_{N}: {np.max(np.abs(DU))}")
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.1)

    def test_precnditioned_cg(self):
        print("\nMultigrid preconditioned cg:")
        for N in [32, 64]:

            rhs = c1.get_rhs(N)
            U_sol = c1.get_u(N)
            U_0 = c1.get_u_0(N)
            U_num = mgv_conditioned_cg_minus_possion(
                U_0, rhs, N, nu1=2, nu2=2, max_level=2, tol=1e-5, maxiter=500
            )
            DU = U_num - U_sol
            print(f"detla_{N}: {np.max(np.abs(DU))}")
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.1)


class TestResdiual(unittest.TestCase):
    def test_v_multigrid(self):
        print("\n1V-Multigrid:")
        for case in [c1, c2]:
            for N in [32, 64, 128]:
                rhs = case.get_rhs(N)
                U_0 = case.get_u_0(N)
                U_num = mgv_minus_poisson(
                    U_0, rhs, N, nu1=20, nu2=20, level=0, max_level=2
                )
                res = rhs - minus_laplace(U_num, N)
                print(f"Residual_{N}: {np.max(np.abs(res))}")
                self.assertAlmostEqual(np.max(np.abs(res)), 0, delta=1e-1)

    def test_multigrid(self):
        print("\nMultigrid:")
        for case in [c1, c2]:
            for N in [32, 64, 128]:
                rhs = case.get_rhs(N)
                U_0 = case.get_u_0(N)
                U_num = multigrid_minus_poisson(
                    U_0, rhs, N, nu1=2, nu2=2, level=0, max_level=3, tol=10e-5
                )
                res = rhs - minus_laplace(U_num, N)
                print(f"Residual_{N}: {np.max(np.abs(res))}")
                self.assertAlmostEqual(np.max(np.abs(res)), 0, delta=1e-3)

    def test_multigrid_conditioned_cg(self):
        print("\nMultigrid:")
        for case in [c1, c2]:
            for N in [32, 64, 128]:
                rhs = case.get_rhs(N)
                U_0 = case.get_u_0(N)
                U_num = mgv_conditioned_cg_minus_possion(
                    U_0, rhs, N, nu1=2, nu2=2, max_level=2, tol=1e-5, maxiter=500
                )
                res = rhs - minus_laplace(U_num, N)
                print(f"Residual_{N}: {np.max(np.abs(res))}")
                self.assertAlmostEqual(np.max(np.abs(res)), 0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
