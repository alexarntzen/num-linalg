import unittest
import numpy as np

from test.case_poisson import CaseOne, CaseTwo

from linalg.laplacian import neg_discrete_laplacian, D_inv, J_w
from linalg.iterative import cg, weighted_jacobi
from linalg.multigrid import mgv_minus_poisson, multigrid_minus_poisson
from linalg.preconditioned import mgv_conditioned_cg_minus_poisson


class TestIterCaseOne(unittest.TestCase):
    def test_cg(self):
        print("\ncg:")
        for N in [32, 64, 128]:
            rhs = CaseOne.get_rhs(N)
            U_sol = CaseOne.get_u(N)
            U_0 = np.random.rand(N + 1, N + 1)
            U_num, hist = cg(
                neg_discrete_laplacian,
                U_0,
                rhs=rhs,
                N=N,
                tol=1e-12,
                maxiter=10000,
                conv_hist=True,
            )
            DU = U_num - U_sol
            print(f"detla_{N}: {np.max(np.abs(DU))}")
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.1)

    def test_weighted_jacobi(self):
        print("\nJacobi:")
        for N in [32, 64, 128]:

            rhs = CaseOne.get_rhs(N)
            U_sol = CaseOne.get_u(N)
            U_0 = CaseOne.get_u_0(N)
            U_num = weighted_jacobi(
                U_0,
                rhs,
                N,
                w=2 / 3,
                nu=25 * N,
                J_w=J_w,
                D_inv=D_inv,
            )
            DU = U_num - U_sol
            print(f"detla_{N}: {np.max(np.abs(DU))}")
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.5)

    def test_multigrid(self):
        print("\nMultigrid:")
        for N in [32, 64]:

            rhs = CaseOne.get_rhs(N)
            U_sol = CaseOne.get_u(N)
            U_0 = CaseOne.get_u_0(N)
            U_num = mgv_minus_poisson(
                U_0, rhs, N, nu1=2000, nu2=2000, level=0, max_level=2
            )
            DU = U_num - U_sol
            print(f"detla_{N}: {np.max(np.abs(DU))}")
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.1)

    def test_precnditioned_cg(self):
        print("\nMultigrid preconditioned cg:")
        for N in [32, 64]:

            rhs = CaseOne.get_rhs(N)
            U_sol = CaseOne.get_u(N)
            U_0 = CaseOne.get_u_0(N)
            U_num = mgv_conditioned_cg_minus_poisson(
                U_0, rhs, N, nu1=2, nu2=2, max_level=2, tol=1e-12, maxiter=500
            )
            DU = U_num - U_sol
            print(f"detla_{N}: {np.max(np.abs(DU))}")
            self.assertAlmostEqual(np.max(np.abs(DU)), 0, delta=0.1)


class TestResdiual(unittest.TestCase):
    def test_cg(self):
        print("\nConjugate Gradient:")
        for case in [CaseOne, CaseTwo]:
            for N in [32, 64, 128]:
                rhs = case.get_rhs(N)
                U_0 = case.get_u_0(N)
                U_num, hist = cg(
                    neg_discrete_laplacian,
                    U_0,
                    rhs,
                    N,
                    tol=1e-12,
                    maxiter=10000,
                    conv_hist=True,
                )
                res = rhs - neg_discrete_laplacian(U_num, N)
                print(f"Residual_{N}: {np.max(np.abs(res))}, iters:{len(hist)}")
                self.assertAlmostEqual(np.max(np.abs(res)), 0, delta=1e-1)

    def test_v_multigrid(self):
        print("\n1V-Multigrid:")
        for case in [CaseOne, CaseTwo]:
            for N in [32, 64, 128]:
                rhs = case.get_rhs(N)
                U_0 = case.get_u_0(N)
                U_num = mgv_minus_poisson(
                    U_0, rhs, N, nu1=20, nu2=20, level=0, max_level=2
                )
                res = rhs - neg_discrete_laplacian(U_num, N)
                print(f"Residual_{N}: {np.max(np.abs(res))}")
                self.assertAlmostEqual(np.max(np.abs(res)), 0, delta=1e-1)

    def test_multigrid(self):
        print("\nMultigrid:")
        for case in [CaseOne, CaseTwo]:
            for N in [32, 64, 128]:
                rhs = case.get_rhs(N)
                U_0 = case.get_u_0(N)
                U_num = multigrid_minus_poisson(
                    U_0, rhs, N, nu1=2, nu2=2, level=0, max_level=2, tol=1e-12
                )
                res = rhs - neg_discrete_laplacian(U_num, N)
                print(f"Residual_{N}: {np.max(np.abs(res))}")
                self.assertAlmostEqual(np.max(np.abs(res)), 0, delta=1e-3)

    def test_multigrid_conditioned_cg(self):
        print("\nMultigrid conditioned cg:")
        for case in [CaseOne, CaseTwo]:
            for N in [32, 64, 128]:
                rhs = case.get_rhs(N)
                U_0 = case.get_u_0(N)
                U_num = mgv_conditioned_cg_minus_poisson(
                    U_0,
                    rhs,
                    N,
                    nu1=2,
                    nu2=1,
                    max_level=2,
                    tol=1e-12,
                    maxiter=500,
                )
                res = rhs - neg_discrete_laplacian(U_num, N)
                print(f"Residual_{N}: {np.max(np.abs(res))}")
                self.assertAlmostEqual(np.max(np.abs(res)), 0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
