import numpy as np
import matplotlib.pyplot as plt


def get_data_of_N(method: callable, case, max_log_N=8, tol=1e-10, **method_kwargs):
    N_list = np.zeros(max_log_N, dtype=np.int)
    iters = np.zeros(max_log_N - 1)
    final_res = np.zeros(max_log_N - 1)
    for i in range(max_log_N - 1):
        N = 2 ** (i + 2)
        rhs = case.get_rhs(N)
        x_0 = case.get_u_0(N)

        _, res_hist = method(x_0=x_0, rhs=rhs, N=N, tol=tol, **method_kwargs)

        N_list[i] = N
        iters[i] = len(res_hist)
        final_res[i] = res_hist[-1]
    return N_list, iters, final_res


def plot_convergence(
    method: callable, case, n_list=[32, 64, 128], tol=1e-10, **method_kwargs
):
    fig, ax = plt.subplots(1)
    ax.set_ytit
    for i, N in enumerate(n_list):
        rhs = case.get_rhs(N)
        x_0 = case.get_u_0(N)
        _, res_hist = method(x_0=x_0, rhs=rhs, N=N, tol=tol, **method_kwargs)
        iters = len(res_hist)
        ax.pot(np.arange(iters), res_hist)

    fig.show()
    return
