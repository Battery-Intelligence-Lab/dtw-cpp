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
| Private members | snake_case + trailing `_` | `dtw_fn_`, `distMat` |
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
- **`std::min({a,b,c})` vs nested `std::min` — measure, don't assume.** The
  2.45×–3.14× win recorded at `65e249b` was specific to the legacy full/banded
  kernels; rolling `BM_dtwFull_L/4000` was unchanged (1.006×). The live
  recurrences use the initialiser-list form (`core/dtw_kernel.hpp:60, 77, 176`,
  `core/msm.hpp`, `core/twe.hpp`) and are **not** to be rewritten on this rule
  alone — see LESSONS.md, "Nested `std::min` helped specific legacy kernels".
- **Template judiciously** — on constraint type only (2-3 variants), NOT on metric type
- **thread_local scratch buffers** — resize, never shrink, avoid per-call allocation
- **Lock-free parallel** — structure decomposition so threads write non-overlapping regions

## Formatting

- **2 spaces** indent (no tabs)
- **Allman braces** for class/function definitions
- Pointer/reference: attached to type (`const std::vector<data_t> &x`)
- No hard line length limit

## Error Handling

- `std::runtime_error` for runtime errors
- Validate at public API boundaries
- OpenMP must be optional (`#ifdef _OPENMP` + serial fallback)
