import numpy as np
import matplotlib.pyplot as plt


def get_data_of_N(method: callable, case, N_list, tol=1e-12, **method_kwargs):
    """Get convergence data"""
    iters = np.zeros(len(N_list))
    final_res = np.zeros(len(N_list))
    for i, N in enumerate(N_list):
        rhs = case.get_rhs(N)
        x_0 = case.get_u_0(N)
        _, conv_hist = method(
            x_0=x_0, rhs=rhs, N=N, tol=tol, conv_hist=True, **method_kwargs
        )
        N_list[i] = N
        iters[i] = len(conv_hist)
        final_res[i] = conv_hist[-1]
    return N_list, iters, final_res


def plot_first_iterations(
    iterations, method, U_0, title="Result after i iterations", **method_kwargs
):
    "Plot the first iterations of one method"
    U_num = U_0
    N = 2 ** iterations
    fig, axs = plt.subplots(ncols=iterations + 1, sharey=True, figsize=(15, 5))
    fig.suptitle(title)
    for i in range(iterations + 1):
        im = axs[i].imshow(
            U_num.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            label=f"N={N}",
            interpolation=None,
        )
        axs[0].set_ylabel("y")
        axs[i].set_xlabel("x")
        axs[i].set_title(f"Iterations: {i}")
        U_num = method(U_num, **method_kwargs)
    fig.colorbar(im, ax=axs, orientation="horizontal")
    return fig


def plot_convergence(method: callable, title="Convergence history", **method_kwargs):
    """Plot convergence plot for one method"""
    fig_conv, axs_conv = plt.subplots(1, figsize=(10, 5))
    u_num, conv_hist = method(conv_hist=True, **method_kwargs)
    axs_conv.grid(True)
    axs_conv.set_ylabel("Convergence criterion")
    axs_conv.set_xlabel("Iterations")
    fig_conv.suptitle(title)
    axs_conv.semilogy(conv_hist)
    return fig_conv


def plot_convergence_iters_to_convergence(
    method: callable,
    case,
    N_list,
    tol=1e-12,
    title="Iteratins to convergence",
    loglog=False,
    **method_kwargs,
):
    N_list, iters, final_res = get_data_of_N(
        method=method, case=case, N_list=N_list, tol=tol, **method_kwargs
    )
    fig_conv, axs_conv = plt.subplots(1, figsize=(10, 5))
    axs_conv.grid(True)
    axs_conv.set_ylabel("Iterations")
    axs_conv.set_xlabel("N")
    fig_conv.suptitle(title)
    if loglog:
        axs_conv.loglog(N_list, iters, base=2)
    else:
        axs_conv.semilogx(N_list, iters, base=2)
    return fig_conv
