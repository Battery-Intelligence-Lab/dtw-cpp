# DTWC++ C++ Style Guide

## C++ Standard

- **Minimum:** C++20
- **Compiler support:** GCC 11+, Clang 14+, MSVC 17.8+, Apple Clang 15+

## Naming

| Element | Convention | Examples |
|---------|-----------|----------|
| Classes/Structs | PascalCase | `Problem`, `Data`, `ClusteringResult` |
| Functions | camelCase (legacy) / snake_case (new) | `dtwBanded`, `fast_pam`, `fill_distance_matrix` |
| Variables | snake_case | `p_vec`, `clusters_ind`, `band` |
| Public members | snake_case (no underscore) | `band`, `name`, `data` |
| Private members | snake_case + trailing `_` | `dtw_fn_`, `data_`, `series_storage_owner_` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_BAND_LENGTH` |
| Enums | PascalCase class + PascalCase values | `Method::Kmedoids`, `Solver::HiGHS` |
| Namespaces | snake_case | `dtwc`, `dtwc::core`, `dtwc::algorithms` |

## File Organization

1. Doxygen block
2. `#pragma once`
3. Project includes
4. Standard library includes

## Performance Rules

- **No virtual dispatch in hot paths** — use CRTP or templates; virtual only at API boundary
- **No `std::min({a,b,c})`, `std::max({…})` or `std::min_element` in a hot loop — nest
  two-argument `std::min`.** The MSVC STL (MSVC, and clang on Windows) compiles the
  initialiser-list form to an out-of-line `__std_min_d` call per element; libc++ inlines
  it, which is why a Mac measures no difference. `std::min(std::min(a, b), c)` makes the
  comparisons `min_element` makes, so results are unchanged, NaN ordering included
  (`baselines/2026-09-29-windows-kernel-msvc-stl-min.md`).
- **Template judiciously** — on constraint type only (2-3 variants), NOT on metric type
- **thread_local scratch buffers** — resize, never shrink, avoid per-call allocation
- **Lock-free parallel** — structure decomposition so threads write non-overlapping regions

## Formatting

- **2 spaces** indent (no tabs)
- **Allman braces** for class/function definitions
- Pointer/reference: attached to type (`const std::vector<data_t> &x`)
- No hard line length limit

## Error Handling

- Typed errors from `dtwc/base/error.hpp` for every failure a caller can cause — `InvalidInput`, `IOError`,
  `DeviceError`, `SolverError` — as the Errors table of `docs/content/api/tier-1.md` assigns them; `std::logic_error` only for a
  programming error (an unreachable branch, a broken invariant). No bare `std::runtime_error`.
- Validate at public API boundaries
- OpenMP must be optional (`#ifdef _OPENMP` + serial fallback)
