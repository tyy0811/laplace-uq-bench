# Day 1: PDE Solver + Dataset Generation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validated Laplace solver with LU factorization reuse, corner-consistent BC sampling (5 families), and train/val/test/OOD dataset splits.

**Architecture:** Sparse 5-point FD Laplacian with scipy.sparse.linalg.splu one-time factorization, then O(n^2) back-substitution per sample (~0.5ms). Corner-consistent BC profiles via x(1-x) endpoint envelope forcing perturbations to zero at corners. Piecewise-constant family held out entirely for OOD testing.

**Tech Stack:** Python 3.9+, NumPy, SciPy (sparse), pytest

**Reference:** See `docs/plans/2026-03-27-design.md` for full project design.

**End-of-day targets:**
- LU-factorized solver validated against analytical solution
- 5 BC families with corner consistency
- 4 dataset splits: train (40K), val (5K), test_in (5K), test_ood (1K)
- 30+ tests passing

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/diffphys/__init__.py` (empty)
- Create: `src/diffphys/pde/__init__.py` (empty)
- Create: `src/diffphys/data/__init__.py` (empty)
- Create: `src/diffphys/model/__init__.py` (empty)
- Create: `src/diffphys/evaluation/__init__.py` (empty)

**Step 1: Initialize git and create directory structure**

```bash
cd /Users/zenith/Desktop/diffusion-physics
git init
mkdir -p src/diffphys/{pde,data,model,evaluation} tests configs experiments modal docs/{plans,figures}
touch src/diffphys/__init__.py src/diffphys/pde/__init__.py src/diffphys/data/__init__.py src/diffphys/model/__init__.py src/diffphys/evaluation/__init__.py
```

**Step 2: Write `.gitignore`**

Create `.gitignore`:
```
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.egg
data/
.vscode/
.idea/
.DS_Store
```

**Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "diffphys"
version = "0.1.0"
description = "Benchmarking generative surrogates for PDE solution fields"
requires-python = ">=3.9"
dependencies = [
    "numpy",
    "scipy",
    "torch",
    "matplotlib",
    "pyyaml",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 4: Install in development mode**

```bash
pip install -e ".[dev]"
```

**Step 5: Verify install**

```bash
python -c "import diffphys; print('OK')"
```

Expected: `OK`

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with src layout and dev dependencies"
```

---

### Task 2: Laplace Solver

**Files:**
- Create: `tests/test_laplace_solver.py`
- Create: `src/diffphys/pde/laplace.py`

**Context:** The solver builds a sparse 5-point FD Laplacian matrix for the interior points of a 64x64 grid (62x62 = 3844 unknowns), LU-factorizes it once via `scipy.sparse.linalg.splu`, then solves for any BCs via back-substitution. Grid convention: `T[i,j]` where i is row (y-direction), j is column (x-direction). `i=0` is top edge, `i=63` is bottom, `j=0` is left, `j=63` is right.

**Step 1: Write tests**

Create `tests/test_laplace_solver.py`:

```python
"""Tests for the Laplace finite-difference solver."""

import numpy as np
import pytest
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
        bcs = [rng.uniform(-1, 1, 64) for _ in range(4)]
        field = solver.solve(*bcs)
        interior = field[1:-1, 1:-1]
        all_bc = np.concatenate(bcs)
        assert interior.min() >= all_bc.min() - 1e-10
        assert interior.max() <= all_bc.max() + 1e-10

    def test_finiteness(self, solver):
        rng = np.random.default_rng(42)
        bcs = [rng.uniform(-1, 1, 64) for _ in range(4)]
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
        bcs = [rng.uniform(-1, 1, 64) for _ in range(4)]
        field = solver.solve(*bcs)

        lap = (
            field[:-2, 1:-1] + field[2:, 1:-1]
            + field[1:-1, :-2] + field[1:-1, 2:]
            - 4 * field[1:-1, 1:-1]
        )
        np.testing.assert_allclose(lap, 0.0, atol=1e-8)

    def test_bc_edges_match(self, solver):
        """Solution edges should exactly match input BCs."""
        rng = np.random.default_rng(42)
        bc_top = rng.uniform(-1, 1, 64)
        bc_bottom = rng.uniform(-1, 1, 64)
        bc_left = rng.uniform(-1, 1, 64)
        bc_right = rng.uniform(-1, 1, 64)

        field = solver.solve(bc_top, bc_bottom, bc_left, bc_right)

        np.testing.assert_array_equal(field[0, :], bc_top)
        np.testing.assert_array_equal(field[-1, :], bc_bottom)
        np.testing.assert_array_equal(field[:, 0], bc_left)
        np.testing.assert_array_equal(field[:, -1], bc_right)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_laplace_solver.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'diffphys.pde.laplace'`

**Step 3: Write implementation**

Create `src/diffphys/pde/laplace.py`:

```python
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
```

**Step 4: Run tests**

```bash
pytest tests/test_laplace_solver.py -v
```

Expected: `11 passed`

**Step 5: Commit**

```bash
git add src/diffphys/pde/laplace.py tests/test_laplace_solver.py
git commit -m "feat: Laplace solver with LU factorization reuse

5-point FD Laplacian on 64x64 grid, one-time splu factorization,
O(n^2) back-substitution per solve. Validated against analytical
solution, maximum principle, and residual checks. 11 tests."
```

---

### Task 3: Boundary Condition Sampling

**Files:**
- Create: `tests/test_boundary.py`
- Create: `src/diffphys/pde/boundary.py`

**Context:** 5 BC families (sinusoidal, Fourier, bump, piecewise, linear). Each edge profile is built as `baseline + perturbation * envelope` where baseline interpolates corner values linearly and `envelope = x*(1-x)*4` forces perturbation to zero at endpoints. Corner consistency is guaranteed because all 4 edges share the same 4 corner values and the envelope kills perturbations at x=0 and x=1.

**Step 1: Write tests**

Create `tests/test_boundary.py`:

```python
"""Tests for corner-consistent boundary condition sampling."""

import numpy as np
import pytest
from diffphys.pde.boundary import (
    BC_TYPES,
    sample_corners,
    sample_edge_profile,
    sample_four_edges,
    sample_perturbation,
)


class TestSampleCorners:
    def test_returns_four_values_in_range(self):
        rng = np.random.default_rng(42)
        corners = sample_corners(rng)
        assert len(corners) == 4
        for c in corners:
            assert -1.0 <= c <= 1.0

    def test_determinism(self):
        c1 = sample_corners(np.random.default_rng(42))
        c2 = sample_corners(np.random.default_rng(42))
        assert c1 == c2


class TestSamplePerturbation:
    def test_shape(self):
        p = sample_perturbation(np.random.default_rng(42), 64, "sinusoidal")
        assert p.shape == (64,)

    @pytest.mark.parametrize("bc_type", BC_TYPES)
    def test_all_types_finite(self, bc_type):
        p = sample_perturbation(np.random.default_rng(42), 64, bc_type)
        assert p.shape == (64,)
        assert np.all(np.isfinite(p))

    def test_linear_is_zero(self):
        p = sample_perturbation(np.random.default_rng(42), 64, "linear")
        np.testing.assert_array_equal(p, 0.0)

    def test_piecewise_has_step_structure(self):
        """Piecewise perturbation should have flat plateaus with sharp transitions."""
        rng = np.random.default_rng(42)
        p = sample_perturbation(rng, 64, "piecewise")
        grad = np.abs(np.diff(p))
        # Most gradient values should be small (flat regions)
        threshold = 0.1 * np.max(grad)
        flat_fraction = np.mean(grad < threshold)
        assert flat_fraction > 0.7

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown BC type"):
            sample_perturbation(np.random.default_rng(42), 64, "unknown")


class TestSampleEdgeProfile:
    def test_shape(self):
        profile = sample_edge_profile(
            np.random.default_rng(42), -0.5, 0.8, "sinusoidal", 64
        )
        assert profile.shape == (64,)

    @pytest.mark.parametrize("bc_type", BC_TYPES)
    def test_endpoints_match(self, bc_type):
        """x(1-x) envelope guarantees profile[0]=c_start, profile[-1]=c_end."""
        c_start, c_end = 0.3, -0.7
        profile = sample_edge_profile(
            np.random.default_rng(42), c_start, c_end, bc_type, 64
        )
        assert profile[0] == pytest.approx(c_start, abs=1e-12)
        assert profile[-1] == pytest.approx(c_end, abs=1e-12)

    def test_finiteness(self):
        profile = sample_edge_profile(
            np.random.default_rng(42), 0.0, 1.0, "fourier", 64
        )
        assert np.all(np.isfinite(profile))


class TestSampleFourEdges:
    def test_shapes(self):
        edges = sample_four_edges(np.random.default_rng(42), nx=64)
        assert len(edges) == 4
        for e in edges:
            assert e.shape == (64,)

    def test_corner_consistency(self):
        """All 4 corners must match between adjacent edges."""
        bc_top, bc_bottom, bc_left, bc_right = sample_four_edges(
            np.random.default_rng(42), nx=64
        )
        # top-left
        assert bc_top[0] == pytest.approx(bc_left[0], abs=1e-12)
        # top-right
        assert bc_top[-1] == pytest.approx(bc_right[0], abs=1e-12)
        # bottom-left
        assert bc_bottom[0] == pytest.approx(bc_left[-1], abs=1e-12)
        # bottom-right
        assert bc_bottom[-1] == pytest.approx(bc_right[-1], abs=1e-12)

    def test_allowed_types_filtering(self):
        """With linear-only, profile should be exactly the linear baseline."""
        edges = sample_four_edges(
            np.random.default_rng(42), allowed_types=["linear"], nx=64
        )
        bc_top = edges[0]
        x = np.linspace(0, 1, 64)
        expected = bc_top[0] + (bc_top[-1] - bc_top[0]) * x
        np.testing.assert_allclose(bc_top, expected, atol=1e-12)

    def test_determinism(self):
        e1 = sample_four_edges(np.random.default_rng(42), nx=64)
        e2 = sample_four_edges(np.random.default_rng(42), nx=64)
        for a, b in zip(e1, e2):
            np.testing.assert_array_equal(a, b)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_boundary.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'diffphys.pde.boundary'`

**Step 3: Write implementation**

Create `src/diffphys/pde/boundary.py`:

```python
"""Corner-consistent boundary condition sampling with 5 BC families.

The "piecewise constant" OOD family uses smooth step transitions
(via the x(1-x) endpoint envelope), not true discontinuities.
The OOD signal is the step-like interior structure, which is
absent from all training BC families.
"""

import numpy as np

BC_TYPES = ["sinusoidal", "fourier", "bump", "piecewise", "linear"]
IN_DIST_TYPES = ["sinusoidal", "fourier", "bump", "linear"]


def sample_corners(rng):
    """Sample 4 corner values from Uniform(-1, 1).

    Returns (c_tl, c_tr, c_bl, c_br) — top-left, top-right,
    bottom-left, bottom-right.
    """
    return tuple(rng.uniform(-1, 1, 4).tolist())


def sample_perturbation(rng, nx, bc_type):
    """Sample a raw perturbation profile for the given BC type.

    Returns (nx,) array. Applied BEFORE the x(1-x) endpoint envelope.
    """
    x = np.linspace(0, 1, nx)

    if bc_type == "sinusoidal":
        A = rng.uniform(0.5, 2.0)
        n = int(rng.choice([1, 2, 3, 4]))
        return A * np.sin(n * np.pi * x)

    elif bc_type == "fourier":
        K = 5
        result = np.zeros(nx)
        for k in range(1, K + 1):
            a_k = rng.normal(0, 1.0 / k**2)
            result += a_k * np.sin(k * np.pi * x)
        return result

    elif bc_type == "bump":
        A = rng.uniform(0.5, 3.0)
        mu = rng.uniform(0.3, 0.7)
        sigma = rng.uniform(0.05, 0.15)
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    elif bc_type == "piecewise":
        n_levels = int(rng.choice([2, 3]))
        levels = rng.uniform(-2, 2, size=n_levels)
        width = 0.05

        if n_levels == 2:
            t = rng.uniform(0.3, 0.7)
            step = 0.5 * (1 + np.tanh((x - t) / width))
            return levels[0] * (1 - step) + levels[1] * step
        else:
            t1, t2 = sorted(rng.uniform(0.2, 0.8, size=2))
            s1 = 0.5 * (1 + np.tanh((x - t1) / width))
            s2 = 0.5 * (1 + np.tanh((x - t2) / width))
            return levels[0] * (1 - s1) + levels[1] * (s1 - s2) + levels[2] * s2

    elif bc_type == "linear":
        return np.zeros(nx)

    else:
        raise ValueError(f"Unknown BC type: {bc_type}")


def sample_edge_profile(rng, c_start, c_end, bc_type=None, nx=64):
    """Sample an edge profile constrained to match endpoint corners.

    The x(1-x)*4 envelope forces perturbation to zero at x=0 and x=1,
    guaranteeing profile[0] = c_start, profile[-1] = c_end.
    """
    if bc_type is None:
        bc_type = rng.choice(BC_TYPES)

    x = np.linspace(0, 1, nx)
    baseline = c_start + (c_end - c_start) * x
    perturbation = sample_perturbation(rng, nx, bc_type)
    envelope = x * (1 - x) * 4  # zero at endpoints, max 1 at center

    return baseline + perturbation * envelope


def sample_four_edges(rng, allowed_types=None, nx=64):
    """Sample 4 corner-consistent edge profiles.

    Returns (bc_top, bc_bottom, bc_left, bc_right) where corners match:
    - bc_top[0] == bc_left[0]       (top-left)
    - bc_top[-1] == bc_right[0]     (top-right)
    - bc_bottom[0] == bc_left[-1]   (bottom-left)
    - bc_bottom[-1] == bc_right[-1] (bottom-right)
    """
    if allowed_types is None:
        allowed_types = BC_TYPES

    c_tl, c_tr, c_bl, c_br = sample_corners(rng)

    bc_top = sample_edge_profile(rng, c_tl, c_tr, rng.choice(allowed_types), nx)
    bc_bottom = sample_edge_profile(rng, c_bl, c_br, rng.choice(allowed_types), nx)
    bc_left = sample_edge_profile(rng, c_tl, c_bl, rng.choice(allowed_types), nx)
    bc_right = sample_edge_profile(rng, c_tr, c_br, rng.choice(allowed_types), nx)

    return bc_top, bc_bottom, bc_left, bc_right
```

**Step 4: Run tests**

```bash
pytest tests/test_boundary.py -v
```

Expected: `14 passed` (5 parametrized BC_TYPES tests count as 5 each for the two parametrized tests)

Note: The exact count depends on parametrization. `test_all_types_finite` runs 5 times, `test_endpoints_match` runs 5 times, plus 8 non-parametrized tests = 18 total.

**Step 5: Commit**

```bash
git add src/diffphys/pde/boundary.py tests/test_boundary.py
git commit -m "feat: corner-consistent BC sampling with 5 families

Sinusoidal, Fourier, bump, piecewise (OOD), and linear BC types.
x(1-x) envelope guarantees corner consistency. Piecewise uses smooth
tanh transitions for step-like OOD signal. 18 tests."
```

---

### Task 4: Dataset Generation

**Files:**
- Create: `tests/test_generate.py`
- Create: `src/diffphys/pde/generate.py`

**Context:** Generates train/val/test_in/test_ood splits. Train/val/test_in use in-distribution BC types (sinusoidal, Fourier, bump, linear). test_ood uses piecewise only (held out entirely from training). Each sample: sample BCs, solve with LaplaceSolver, store field + BCs as float32.

**Step 1: Write tests**

Create `tests/test_generate.py`:

```python
"""Tests for dataset generation with held-out OOD split."""

import numpy as np
import pytest
from diffphys.pde.generate import generate_dataset
from diffphys.pde.laplace import LaplaceSolver


@pytest.fixture
def solver():
    return LaplaceSolver(nx=64)


def test_shapes(solver):
    data = generate_dataset(10, seed=42, solver=solver, nx=64)
    assert data["fields"].shape == (10, 64, 64)
    assert data["bc_top"].shape == (10, 64)
    assert data["bc_bottom"].shape == (10, 64)
    assert data["bc_left"].shape == (10, 64)
    assert data["bc_right"].shape == (10, 64)


def test_dtype_is_float32(solver):
    data = generate_dataset(5, seed=42, solver=solver, nx=64)
    assert data["fields"].dtype == np.float32
    assert data["bc_top"].dtype == np.float32


def test_determinism(solver):
    d1 = generate_dataset(5, seed=42, solver=solver, nx=64)
    d2 = generate_dataset(5, seed=42, solver=solver, nx=64)
    np.testing.assert_array_equal(d1["fields"], d2["fields"])


def test_bc_consistency(solver):
    """Field edges should match stored BCs (both float32)."""
    data = generate_dataset(5, seed=42, solver=solver, nx=64)
    for i in range(5):
        np.testing.assert_allclose(
            data["fields"][i, 0, :], data["bc_top"][i], atol=1e-6
        )
        np.testing.assert_allclose(
            data["fields"][i, -1, :], data["bc_bottom"][i], atol=1e-6
        )
        np.testing.assert_allclose(
            data["fields"][i, :, 0], data["bc_left"][i], atol=1e-6
        )
        np.testing.assert_allclose(
            data["fields"][i, :, -1], data["bc_right"][i], atol=1e-6
        )


def test_ood_split(solver):
    """OOD dataset with piecewise-only BCs should produce valid fields."""
    data = generate_dataset(
        5, seed=45, solver=solver, allowed_bc_types=["piecewise"], nx=64
    )
    assert data["fields"].shape == (5, 64, 64)
    assert np.all(np.isfinite(data["fields"]))


def test_max_principle(solver):
    """All generated fields should satisfy the maximum principle."""
    data = generate_dataset(10, seed=42, solver=solver, nx=64)
    for i in range(10):
        interior = data["fields"][i, 1:-1, 1:-1]
        all_bc = np.concatenate(
            [
                data["bc_top"][i],
                data["bc_bottom"][i],
                data["bc_left"][i],
                data["bc_right"][i],
            ]
        )
        assert interior.min() >= all_bc.min() - 1e-5
        assert interior.max() <= all_bc.max() + 1e-5
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generate.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'diffphys.pde.generate'`

**Step 3: Write implementation**

Create `src/diffphys/pde/generate.py`:

```python
"""Dataset generation with held-out OOD split."""

import argparse
import os

import numpy as np

from .boundary import IN_DIST_TYPES, sample_four_edges
from .laplace import LaplaceSolver


def generate_dataset(n, seed, solver, allowed_bc_types=None, nx=64):
    """Generate n PDE solution samples with given BC types.

    Args:
        n: Number of samples to generate.
        seed: Random seed for reproducibility.
        solver: LaplaceSolver instance (reused for all samples).
        allowed_bc_types: List of BC family names, or None for all in-dist.
        nx: Grid resolution.

    Returns:
        Dict with keys: fields (n, nx, nx), bc_top/bottom/left/right (n, nx).
        All arrays are float32.
    """
    if allowed_bc_types is None:
        allowed_bc_types = IN_DIST_TYPES

    rng = np.random.default_rng(seed)

    fields = np.zeros((n, nx, nx), dtype=np.float32)
    bc_tops = np.zeros((n, nx), dtype=np.float32)
    bc_bottoms = np.zeros((n, nx), dtype=np.float32)
    bc_lefts = np.zeros((n, nx), dtype=np.float32)
    bc_rights = np.zeros((n, nx), dtype=np.float32)

    for i in range(n):
        bc_top, bc_bottom, bc_left, bc_right = sample_four_edges(
            rng, allowed_types=allowed_bc_types, nx=nx
        )
        field = solver.solve(bc_top, bc_bottom, bc_left, bc_right)

        fields[i] = field.astype(np.float32)
        bc_tops[i] = bc_top.astype(np.float32)
        bc_bottoms[i] = bc_bottom.astype(np.float32)
        bc_lefts[i] = bc_left.astype(np.float32)
        bc_rights[i] = bc_right.astype(np.float32)

    return {
        "fields": fields,
        "bc_top": bc_tops,
        "bc_bottom": bc_bottoms,
        "bc_left": bc_lefts,
        "bc_right": bc_rights,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate PDE dataset splits")
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--n-ood", type=int, default=1000)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    solver = LaplaceSolver(nx=args.nx)

    splits = [
        ("train", args.n_train, IN_DIST_TYPES, 42),
        ("val", args.n_val, IN_DIST_TYPES, 43),
        ("test_in", args.n_test, IN_DIST_TYPES, 44),
        ("test_ood", args.n_ood, ["piecewise"], 45),
    ]

    for name, n, bc_types, seed in splits:
        print(f"Generating {name} ({n} samples)...")
        data = generate_dataset(n, seed, solver, allowed_bc_types=bc_types, nx=args.nx)
        path = os.path.join(args.output, f"{name}.npz")
        np.savez_compressed(path, **data)
        print(f"  Saved to {path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
pytest tests/test_generate.py -v
```

Expected: `7 passed`

**Step 5: Commit**

```bash
git add src/diffphys/pde/generate.py tests/test_generate.py
git commit -m "feat: dataset generation with held-out piecewise OOD split

Generates train (40K), val (5K), test_in (5K), test_ood (1K) splits.
Piecewise BC family held out entirely from training data. CLI via
python -m diffphys.pde.generate. 7 tests."
```

---

### Task 5: Generate Full Dataset

**Files:**
- Create: `data/train.npz`, `data/val.npz`, `data/test_in.npz`, `data/test_ood.npz`

**Step 1: Run dataset generation**

```bash
cd /Users/zenith/Desktop/diffusion-physics
python -m diffphys.pde.generate --output data
```

Expected output (takes ~30-60s on CPU):
```
Generating train (40000 samples)...
  Saved to data/train.npz
Generating val (5000 samples)...
  Saved to data/val.npz
Generating test_in (5000 samples)...
  Saved to data/test_in.npz
Generating test_ood (1000 samples)...
  Saved to data/test_ood.npz
```

**Step 2: Verify dataset files**

```bash
python -c "
import numpy as np
for split in ['train', 'val', 'test_in', 'test_ood']:
    d = np.load(f'data/{split}.npz')
    print(f'{split}: fields={d[\"fields\"].shape}, bc_top={d[\"bc_top\"].shape}, dtype={d[\"fields\"].dtype}')
"
```

Expected:
```
train: fields=(40000, 64, 64), bc_top=(40000, 64), dtype=float32
val: fields=(5000, 64, 64), bc_top=(5000, 64), dtype=float32
test_in: fields=(5000, 64, 64), bc_top=(5000, 64), dtype=float32
test_ood: fields=(1000, 64, 64), bc_top=(1000, 64), dtype=float32
```

Note: data/ is in .gitignore — do NOT commit the dataset files.

---

### Task 6: Final Verification + Day 1 Commit

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: `36 passed` (11 solver + 18 boundary + 7 generate)

**Step 2: Verify test count meets target**

The Day 1 target is 17+ tests. We have 36, which exceeds the target.

**Step 3: Final commit if any uncommitted changes**

```bash
git status
# If clean, no action needed. If any stray files:
git add -A
git commit -m "chore: Day 1 complete — solver, BCs, dataset generation, 36 tests"
```

**Day 1 deliverables:**
- LU-factorized Laplace solver validated against analytical solution
- 5 BC families with corner consistency via x(1-x) envelope
- Dataset: train (40K), val (5K), test_in (5K), test_ood (1K piecewise-only)
- 36 tests passing
