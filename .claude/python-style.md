# DTWC++ Python Style Guide

## Python Version

- **Minimum:** Python 3.9
- **Supported:** 3.9 through 3.14

## Style

Follow PEP 8. Use `ruff` for linting and formatting.

## Naming

- Modules: `snake_case` (`_clustering.py`, `__init__.py`)
- Classes: `PascalCase` (`DTWClustering`, `Problem`)
- Functions: `snake_case` (`dtw_distance`, `fast_pam`, `check_system`)
- Constants: `UPPER_SNAKE_CASE` (`CUDA_AVAILABLE`, `OPENMP_AVAILABLE`)

## Docstrings

NumPy-style for consistency with scientific Python:

```python
def dtw_distance(x, y, band=-1, metric="l1"):
    """Compute DTW distance between two time series.

    Parameters
    ----------
    x : array-like
        First time series.
    y : array-like
        Second time series.
    band : int, optional
        Sakoe-Chiba band width. -1 for full DTW.

    Returns
    -------
    float
        The DTW distance.
    """
```

## Class Design

- `DTWClustering` inherits `sklearn.BaseEstimator + ClusterMixin` (optional sklearn dep)
- `fit(X)`, `predict(X)`, `fit_predict(X)` API
- Attributes: `labels_`, `medoid_indices_`, `inertia_`

## Bindings (nanobind)

- Release GIL for any C++ call >10ms: `nb::gil_scoped_release`
- Zero-copy numpy via `nb::ndarray` with `nb::capsule` ownership
- Accept lists and numpy arrays; convert internally with `np.asarray`

## Package Structure

```
python/
  dtwcpp/
    __init__.py       # Public API, check_system()
    _clustering.py    # DTWClustering sklearn-compatible class
    io.py             # CSV/HDF5/Parquet I/O utilities
  src/
    _dtwcpp_core.cpp  # nanobind C++ bindings
```

## Build & Publish

- **Build system:** scikit-build-core + nanobind + CMake
- **Package manager:** `uv` (not pip)
- **Wheels:** cibuildwheel for cp39-cp314 on Linux/macOS/Windows
- **Publish:** `uv publish` with OIDC trusted publishing
