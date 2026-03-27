"""Finite-difference Laplace solver with LU factorization reuse."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def build_laplacian_matrix(nx=64):
    """Build 2D Laplacian matrix for interior points of an nx x nx grid.

    5-point stencil. Returns sparse CSC matrix of shape (n^2, n^2)
    where n = nx - 2 (number of interior points per dimension).
    """
    n = nx - 2
    N = n * n

    main_diag = np.full(N, -4.0)

    # j+/-1 neighbors (offset +/-1), zeroed at row boundaries
    side_diag = np.ones(N - 1)
    side_diag[np.arange(n - 1, N - 1, n)] = 0.0

    # i+/-1 neighbors (offset +/-n)
    vert_diag = np.ones(N - n)

    L = sparse.diags(
        [vert_diag, side_diag, main_diag, side_diag, vert_diag],
        [-n, -1, 0, 1, n],
        shape=(N, N),
        format="csc",
    )
    return L


class LaplaceSolver:
    """Reusable Laplace solver with one-time LU factorization.

    Build the sparse Laplacian once, factorize once, then solve
    for any boundary conditions via back-substitution (~0.5ms per solve).
    """

    def __init__(self, nx=64):
        self.nx = nx
        self.n = nx - 2
        L = build_laplacian_matrix(nx)
        self._lu = splu(L)

    def _assemble_rhs(self, bc_top, bc_bottom, bc_left, bc_right):
        """Assemble RHS vector from boundary conditions.

        bc_top: T[0, :] (top row), bc_bottom: T[nx-1, :] (bottom row),
        bc_left: T[:, 0] (left col), bc_right: T[:, nx-1] (right col).

        Boundary neighbor contributions move to the RHS with sign flip:
        for interior point (i,j), if neighbor is boundary value b,
        then rhs[k] -= b.
        """
        n, nx = self.n, self.nx
        rhs = np.zeros((n, n))

        rhs[0, :] -= bc_top[1 : nx - 1]
        rhs[n - 1, :] -= bc_bottom[1 : nx - 1]
        rhs[:, 0] -= bc_left[1 : nx - 1]
        rhs[:, n - 1] -= bc_right[1 : nx - 1]

        return rhs.ravel()

    def solve(self, bc_top, bc_bottom, bc_left, bc_right):
        """Solve Laplace equation with given Dirichlet BCs.

        Returns (nx, nx) array with boundary values set and interior solved.
        """
        nx, n = self.nx, self.n
        rhs = self._assemble_rhs(bc_top, bc_bottom, bc_left, bc_right)
        interior = self._lu.solve(rhs)

        field = np.zeros((nx, nx))
        field[0, :] = bc_top
        field[nx - 1, :] = bc_bottom
        field[:, 0] = bc_left
        field[:, nx - 1] = bc_right
        field[1 : nx - 1, 1 : nx - 1] = interior.reshape(n, n)

        return field
