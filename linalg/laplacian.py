import numpy as np


def neg_discrete_laplacian(u, N):
    """Compute negative discrete laplacian times h**2, id on the boundary"""
    index = np.arange(1, N)
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index - 1, index)
    ixp_y = np.ix_(index + 1, index)
    ix_ym = np.ix_(index, index - 1)
    ix_yp = np.ix_(index, index + 1)

    # id on boundary points boundary points
    Lu = u.copy()

    # calculate -laplacian(u)
    Lu[ixy] = -(u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp]) + 4 * u[ixy]
    return Lu


def J_w(u, N, w):
    """J_w, function based on laplacian
    Used in jacobi iteration
    """
    return u - w * D_inv(neg_discrete_laplacian(u, N), N)


def D_inv(u, N):
    """D is diagonal of the negative discrete laplacian above.
    Compute D^-1(u) for a given."""
    u_new = u.copy()
    index = np.arange(1, N)
    ixy = np.ix_(index, index)
    u_new[ixy] = u[ixy] / 4
    return u_new
