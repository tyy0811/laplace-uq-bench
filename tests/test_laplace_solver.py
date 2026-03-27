"""Tests for the Laplace finite-difference solver."""

import numpy as np
import pytest
from diffphys.pde.boundary import sample_four_edges
from diffphys.pde.laplace import build_laplacian_matrix, LaplaceSolver


class TestBuildLaplacianMatrix:
    def test_shape(self):
        L = build_laplacian_matrix(nx=64)
        n = 62
        assert L.shape == (n * n, n * n)

    def test_symmetry(self):
        L = build_laplacian_matrix(nx=16)
        diff = L - L.T
        assert diff.nnz == 0

    def test_diagonal_values(self):
        L = build_laplacian_matrix(nx=16)
        np.testing.assert_array_equal(L.diagonal(), -4.0)

    def test_row_sum_interior_point(self):
        """Rows for fully-interior points (no boundary neighbors) sum to 0."""
        L = build_laplacian_matrix(nx=16)
        n = 14
        # Point at interior position (1,1) -> linear index n + 1
        k = n + 1
        row = L.getrow(k).toarray().ravel()
        assert row.sum() == pytest.approx(0.0)


class TestInputValidation:
    @pytest.mark.parametrize("nx", [0, 1, 2, -1])
    def test_small_nx_raises(self, nx):
        with pytest.raises(ValueError, match="nx must be >= 3"):
            LaplaceSolver(nx=nx)

    @pytest.mark.parametrize("nx", [0, 1, 2, -1])
    def test_build_laplacian_small_nx_raises(self, nx):
        with pytest.raises(ValueError, match="nx must be >= 3"):
            build_laplacian_matrix(nx=nx)

    def test_inconsistent_corners_raises(self):
        solver = LaplaceSolver(nx=8)
        bc_top = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        bc_bottom = np.zeros(8)
        bc_left = np.ones(8)   # bc_left[0]=1.0 != bc_top[0]=0.0
        bc_right = np.zeros(8)
        with pytest.raises(ValueError, match="Inconsistent top-left corner"):
            solver.solve(bc_top, bc_bottom, bc_left, bc_right)


class TestLaplaceSolver:
    @pytest.fixture
    def solver(self):
        return LaplaceSolver(nx=64)

    def test_output_shape(self, solver):
        bc = np.zeros(64)
        field = solver.solve(bc, bc, bc, bc)
        assert field.shape == (64, 64)

    def test_zero_bcs_zero_interior(self, solver):
        bc = np.zeros(64)
        field = solver.solve(bc, bc, bc, bc)
        np.testing.assert_allclose(field, 0.0, atol=1e-12)

    def test_constant_bcs_constant_solution(self, solver):
        """Constant BCs -> constant interior (equals the constant)."""
        bc = np.ones(64)
        field = solver.solve(bc, bc, bc, bc)
        np.testing.assert_allclose(field, 1.0, atol=1e-12)

    def test_analytical_solution(self, solver):
        """Compare against T(x,y) = sin(pi*x)*sinh(pi*y)/sinh(pi).

        BCs: T(x,0)=0 (top row), T(x,1)=sin(pi*x) (bottom row),
             T(0,y)=0 (left), T(1,y)=0 (right).
        Grid: y = i/(nx-1), x = j/(nx-1).
        """
        nx = 64
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, nx)

        bc_top = np.zeros(nx)           # y=0
        bc_bottom = np.sin(np.pi * x)   # y=1
        bc_left = np.zeros(nx)           # x=0
        bc_right = np.zeros(nx)          # x=1, sin(pi) ~ 0

        field = solver.solve(bc_top, bc_bottom, bc_left, bc_right)

        X, Y = np.meshgrid(x, y)
        T_exact = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)

        # FD error is O(h^2) ~ (1/63)^2 ~ 2.5e-4
        np.testing.assert_allclose(field, T_exact, atol=5e-4)

    def test_maximum_principle(self, solver):
        """Interior values bounded by boundary extremes."""
        rng = np.random.default_rng(42)
        bcs = sample_four_edges(rng, nx=64)
        field = solver.solve(*bcs)
        interior = field[1:-1, 1:-1]
        all_bc = np.concatenate(bcs)
        assert interior.min() >= all_bc.min() - 1e-10
        assert interior.max() <= all_bc.max() + 1e-10

    def test_finiteness(self, solver):
        rng = np.random.default_rng(42)
        bcs = sample_four_edges(rng, nx=64)
        field = solver.solve(*bcs)
        assert np.all(np.isfinite(field))

    def test_symmetric_bcs_symmetric_solution(self, solver):
        """Left-right symmetric BCs -> left-right symmetric solution."""
        nx = 64
        x = np.linspace(0, 1, nx)
        bc_sym = np.sin(np.pi * x)  # symmetric about x=0.5
        bc_zero = np.zeros(nx)

        field = solver.solve(bc_sym, bc_sym, bc_zero, bc_zero)

        np.testing.assert_allclose(
            field[:, :nx // 2],
            np.flip(field[:, nx // 2:], axis=1),
            atol=1e-10,
        )

    def test_solver_reuse(self, solver):
        """Same solver instance, different BCs, both correct."""
        bc_zero = np.zeros(64)
        field1 = solver.solve(bc_zero, bc_zero, bc_zero, bc_zero)

        bc_ones = np.ones(64)
        field2 = solver.solve(bc_ones, bc_ones, bc_ones, bc_ones)

        np.testing.assert_allclose(field1, 0.0, atol=1e-12)
        np.testing.assert_allclose(field2, 1.0, atol=1e-12)

    def test_residual(self, solver):
        """Numerical Laplacian of solution should be near zero."""
        rng = np.random.default_rng(42)
        bcs = sample_four_edges(rng, nx=64)
        field = solver.solve(*bcs)

        lap = (
            field[:-2, 1:-1] + field[2:, 1:-1]
            + field[1:-1, :-2] + field[1:-1, 2:]
            - 4 * field[1:-1, 1:-1]
        )
        np.testing.assert_allclose(lap, 0.0, atol=1e-8)

    def test_bc_edges_match(self, solver):
        """Solution edges should exactly match input BCs (corner-consistent)."""
        rng = np.random.default_rng(42)
        c_tl, c_tr, c_bl, c_br = rng.uniform(-1, 1, 4)

        bc_top = rng.uniform(-1, 1, 64)
        bc_top[0], bc_top[-1] = c_tl, c_tr

        bc_bottom = rng.uniform(-1, 1, 64)
        bc_bottom[0], bc_bottom[-1] = c_bl, c_br

        bc_left = rng.uniform(-1, 1, 64)
        bc_left[0], bc_left[-1] = c_tl, c_bl

        bc_right = rng.uniform(-1, 1, 64)
        bc_right[0], bc_right[-1] = c_tr, c_br

        field = solver.solve(bc_top, bc_bottom, bc_left, bc_right)

        np.testing.assert_array_equal(field[0, :], bc_top)
        np.testing.assert_array_equal(field[-1, :], bc_bottom)
        np.testing.assert_array_equal(field[:, 0], bc_left)
        np.testing.assert_array_equal(field[:, -1], bc_right)
