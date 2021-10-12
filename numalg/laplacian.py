import numpy as np


def L(u, N):
    """Compute negative discrete laplacian times h**2 + id on the boundaryt"""
    index = np.arange(1, N)
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index - 1, index)
    ixp_y = np.ix_(index + 1, index)
    ix_ym = np.ix_(index, index - 1)
    ix_yp = np.ix_(index, index + 1)

    # id for boundary points
    Lu = u.copy()

    # - \delta
    Lu[ixy] = -(u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp]) + 4 * u[ixy]
    return Lu
