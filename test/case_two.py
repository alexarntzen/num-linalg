import numpy as np


def get_f(N):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    f = X * 0 - 1  # hack to get correct shape
    return f


def get_g(N):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    g = X * 0

    ix = np.arange(1)
    iy = np.arange(0, N + 1)
    ixy = np.ix_(ix, iy)
    g[ixy] = 4 * Y[ixy] * (1 - Y[ixy])

    return g


# get_u = get_g


def get_rhs(N):
    # assuming g=U on the boundary, which is true

    ix = np.arange(1, N)
    ixy = np.ix_(ix, ix)

    rhs = get_g(N)  # = g
    rhs[ixy] = get_f(N)[ixy] / N ** 2

    return rhs


def get_u_0(N):
    return np.random.rand(N + 1, N + 1)
