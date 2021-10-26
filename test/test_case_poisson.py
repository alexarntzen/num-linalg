import numpy as np


class CaseOne:
    @staticmethod
    def get_f(N):
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = 20 * np.pi ** 2 * np.sin(2 * np.pi * X) * np.sin(4 * np.pi * Y)
        return f

    @staticmethod
    def get_u(N):
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")

        U = np.sin(2 * np.pi * X) * np.sin(4 * np.pi * Y)
        return U

    @classmethod
    def get_rhs(cls, N):
        # assuming g=U on the boundary, which is true

        ix = np.arange(1, N)
        ixy = np.ix_(ix, ix)

        rhs = cls.get_u(N)  # = g
        rhs[ixy] = cls.get_f(N)[ixy] / N ** 2

        return rhs

    @staticmethod
    def get_u_0(N):
        return np.random.rand(N + 1, N + 1)


class CaseTwo:
    @staticmethod
    def get_f(N):
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = X * 0 - 1  # hack to get correct shape
        return f

    @staticmethod
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

    @classmethod
    def get_rhs(cls, N):
        # assuming g=U on the boundary, which is true

        ix = np.arange(1, N)
        ixy = np.ix_(ix, ix)

        rhs = cls.get_g(N)  # = g
        rhs[ixy] = cls.get_f(N)[ixy] / N ** 2

        return rhs

    @staticmethod
    def get_u_0(N):
        return np.random.rand(N + 1, N + 1)
