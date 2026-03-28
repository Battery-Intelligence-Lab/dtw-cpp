# DTWC++ C++ Style Guide

This document describes the C++ coding conventions used in the DTWC++ project, derived from analyzing the existing codebase.

## C++ Standard

- **Minimum:** C++17
- **Preferred:** C++23 (use feature-test macros for C++20/23 features)
- **Compiler support:** GCC 12+, Clang 15+, MSVC 17.8+

## Naming Conventions

### Classes and Structs
- **PascalCase** (UpperCamelCase)
- Examples: `Problem`, `Data`, `Range`, `Index`, `DataLoader`

### Functions and Methods
- **snake_case** (enforced for all new code; legacy camelCase will be migrated incrementally)
- New code examples: `dtw_full`, `dtw_banded`, `fill_distance_matrix`, `read_file`
- Legacy (to be renamed): `dtwFull` -> `dtw_full`, `readFile` -> `read_file`, `distByInd` -> `dist_by_ind`

### DTW Algorithm Naming
- New code: `dtw_full`, `dtw_banded`, `dtw_full_linear` (snake_case)
- Legacy (to rename): `dtwFull` -> `dtw_full`, `dtwBanded` -> `dtw_banded`, `dtwFull_L` -> `dtw_full_linear`
- Clustering: `lloyd_kmedoids`, `fast_pam`, `clara` (snake_case, descriptive)
- Legacy rename: `cluster_by_kMedoidsPAM` -> `cluster_by_kMedoidsLloyd` (Phase 0), then -> `lloyd_kmedoids` (Phase 1)

### Variables
- **snake_case**
- Examples: `p_vec`, `p_names`, `clusters_ind`, `centroids_ind`

### Member Variables
- **Public members:** `snake_case` without trailing underscore (e.g., `band`, `name`, `data`)
- **Private members:** `snake_case` with trailing underscore (e.g., `dist_mat_`, `solver_`)
- **Legacy names (to be migrated):** `Nc`, `distMat`, `mipSolver` -- rename to `n_clusters`, `dist_mat`, `mip_solver` incrementally

### Constants
- **UPPER_SNAKE_CASE** for compile-time constants
- Examples: `DEFAULT_BAND_LENGTH`, `DEFAULT_MIP_SOLVER`

### Template Parameters
- Descriptive names with `_t` suffix or `T` prefix
- Examples: `data_t`, `Tfun`, `Tpath`

### Namespaces
- **snake_case** or short lowercase names
- Primary namespace: `dtwc`
- Nested: `dtwc::settings`, `dtwc::init`, `dtwc::scores`

### Enums
- Enum class names: **PascalCase**
- Enum values: **PascalCase**
- Examples: `Method::Kmedoids`, `Method::MIP`, `Solver::Gurobi`

## File Organization

### Header Files
1. Doxygen documentation block at top
2. `#pragma once` (not traditional include guards)
3. Project-local includes
4. Standard library includes
5. Third-party includes (e.g., Armadillo)

Example:
```cpp
/**
 * @file Problem.hpp
 * @brief Brief description
 * @date 19 Oct 2022
 * @author Author Name
 */

#pragma once

#include "Data.hpp"
#include "settings.hpp"

#include <string>
#include <vector>

#include <armadillo>
```

### Namespace Aliases
Use `namespace fs = std::filesystem;` at namespace scope.

## Performance Rules

### No Virtual Dispatch in Hot Paths
- Distance matrix access, DTW inner loops, and clustering iterations must use direct calls or templates
- Virtual dispatch costs ~3ns per call; at N=10K this adds 300ms per PAM iteration
- Use CRTP or template parameters for hot-path abstractions
- Virtual dispatch is acceptable at outer API boundaries (bindings, CLI)

### Template Judiciously
- Template on constraint type only (None, SakoeChibaBand, Itakura) -- 2-3 variants
- Do NOT template on metric type -- runtime callable dispatch overhead is 0.003% of DTW cost
- Maximum ~6 explicit instantiations (3 constraints x 2 scalar types)

### ScratchMatrix Pattern
- Use `ScratchMatrix<T>` (row-major vector + stride) instead of `arma::Mat` for scratch buffers
- Row-major layout ensures contiguous access in DTW inner loop
- Use `thread_local ScratchMatrix<T>` for per-thread scratch space
- For banded DTW, use rolling buffer of width `2*band+1` (not full matrix)

### Memory-Bound Awareness
- DTW is memory-bound (0.125 FLOP/byte). Fix memory access patterns before adding SIMD.
- Prefer rolling buffers over full matrix allocation
- Use cache-friendly access patterns (contiguous inner loop)

## Formatting

### Indentation
- **2 spaces** (no tabs)
- Configured in `.clang-format`

### Braces
- **Allman style** for classes, structs, functions
- Opening brace on new line for class/function definitions

```cpp
class Problem
{
public:
  void doSomething();
};

void Problem::doSomething()
{
  if (condition) {
    // short blocks may use attached braces
  }
}
```

### Line Length
- No hard limit enforced (ColumnLimit: 0 in clang-format)
- Prefer readable line breaks for long parameter lists

### Pointer/Reference Alignment
- Attached to type (right alignment): `const std::vector<data_t> &x`

### Template Formatting
- Space after `template` keyword: `template <typename T>`
- No spaces inside angle brackets: `std::vector<int>`

## Documentation

### Doxygen Style
```cpp
/**
 * @brief Short description.
 * @details Longer explanation if needed.
 * @tparam data_t Description of template parameter.
 * @param name Description of parameter.
 * @return Description of return value.
 */
```

### Member Variable Comments
- Use `//!<` or `/*!< */` for inline documentation
```cpp
int Nc{ 1 }; /*!< Number of clusters. */
bool is_distMat_filled{ false }; //!< Whether distance matrix is computed.
```

### Inline Comments
- Use `//` for single-line comments
- Place on same line for short explanations
```cpp
if (&x == &y) return 0; // Same data, distance is 0
```

## Modern C++ Features

### Initialization
- Prefer brace initialization for members:
```cpp
int Nc{ 1 };
bool flag{ false };
```

### Type Inference
- Use `auto` for complex return types and iterators
- Be explicit when type clarity matters

### Move Semantics
- Use `std::move` for efficiency in constructors and functions
```cpp
p_vec = std::move(p_vec_new);
p_vec.push_back(std::move(p));
```

### Structured Bindings (C++17)
```cpp
auto &[short_vec, long_vec] = getVectors();
```

### constexpr
- Use for compile-time constants:
```cpp
constexpr int DEFAULT_BAND_LENGTH = -1;
constexpr data_t maxValue = std::numeric_limits<data_t>::max();
```

### Type Aliases
- Prefer `using` over `typedef`:
```cpp
using distMat_t = arma::Mat<double>;
using path_t = std::decay_t<decltype(settings::resultsPath)>;
```

## Memory Management

### No Raw new/delete
- Use standard containers (`std::vector`, `std::string`)
- Use smart pointers when dynamic allocation is needed

### RAII
- All resources managed through constructors/destructors
- File handles, locks, etc. cleaned up automatically

### Thread-Local Storage
- Use `thread_local` for per-thread scratch buffers:
```cpp
thread_local arma::Mat<data_t> C;
thread_local static std::vector<data_t> buffer(10000);
```

## Error Handling

### Exceptions
- Use `std::runtime_error` for runtime errors
- Provide descriptive error messages
```cpp
throw std::runtime_error("Error opening file: " + path.string());
```

### Validation
- Validate inputs at public API boundaries
- Use assertions for internal invariants in debug builds

## Performance Considerations

### Avoid Allocations in Hot Loops
- Reuse buffers via thread-local storage
- Reserve capacity for vectors when size is known

### Use References for Large Objects
```cpp
void process(const std::vector<data_t> &data);
```

### Prefer operator[] Over .at() in Hot Paths
- `.at()` has bounds checking overhead
- Use `operator[]` when indices are guaranteed valid

## Parallelization

### OpenMP
- Must be optional (guarded by `#ifdef _OPENMP`)
- Provide serial fallback for all parallel code

## Formatting Tool

Run clang-format before committing:
```bash
clang-format -i path/to/file.cpp
```

Or check without modifying:
```bash
clang-format --dry-run --Werror path/to/file.cpp
```
