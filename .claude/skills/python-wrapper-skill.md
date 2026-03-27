---
description: "Analyze C++ headers and generate pybind11 Python bindings with Pythonic sugar layer. Maintains OOP consistency (CasADi-style) so switching between C++ and Python feels seamless."
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Task
---

# Python Wrapper Generator

Generate complete pybind11 Python bindings from C++ header files, following CasADi's cross-language design philosophy: **the same class names, same method signatures, same feel** — Python code should read naturally while mapping 1:1 to the C++ API.

## Requirements

- **pybind11** >= 2.10 (for `pybind11/stl/filesystem.h`)
- **C++17** or newer
- **Python** >= 3.8

## Input

`$ARGUMENTS` = path(s) to C++ header file(s) to wrap. If empty, scan the project's `include/` and source directories for public headers.

## Step 0: Check for Existing Bindings

Before generating anything, search the project for existing binding files:
```
Glob pattern: **/*pybind*  **/*py_main*  **/bindings/python/**  **/python/**/*.cpp
```

If existing pybind11 bindings are found:
- Read them carefully to understand what is already bound
- **Extend** the existing file rather than creating a new one
- Note any commented-out code (may indicate known issues with certain bindings)
- Preserve the existing module name and structure

## Step 1: Analyze C++ Headers

Read each specified header and extract a structured inventory:

### For each class/struct:
- Class name, namespace, base class (if any)
- **Constructors**: parameter types, defaults
- **Public methods**: name, return type, parameter types, const-ness, overloads
- **Public fields**: name, type, default value
- **Builder pattern**: detect methods returning `ClassName&` (fluent setters) — these become Python properties
- **Operator overloads**: `operator==`, `operator<<`, etc.

### For each enum class:
- Name, namespace, enumerators with values

### For each free function:
- Name, namespace, template parameters, regular parameters, return type
- If template: note which types to instantiate (default: `double`)

### For type aliases:
- Name, underlying type

Record the full inventory before generating any code.

### Tricky C++ patterns to watch for

These require special handling in pybind11:

1. **`auto` return types** — Cannot take address of `auto`-returning methods. Use lambda wrappers:
   ```cpp
   .def("method", [](const Class& self, size_t i) { return self.method(i); })
   ```

2. **Rvalue-reference constructors** (`Class(T&& arg)`) — pybind11 cannot bind rvalue refs directly. Create a lambda that copies:
   ```cpp
   .def(py::init([](std::vector<double> v1, std::vector<std::string> v2) {
       return Class(std::move(v1), std::move(v2));
   }), py::arg("data"), py::arg("names"))
   ```

3. **Mutable lvalue reference parameters** (`void foo(std::vector<int>& v)`) — pybind11's STL conversion creates a temporary, so mutations are lost. Either:
   - Accept by value and return modified copy, or
   - Use `py::array_t<int>` with buffer protocol for true reference semantics

4. **Type aliases** (e.g., `using data_t = double`) — Check the project's type alias and instantiate templates for the actual underlying type, not the alias name.

5. **`thread_local` scratch buffers** — Common in DTW implementations. These work correctly under pybind11 for single-threaded Python calls, but be aware the buffer persists across calls. Document this in comments.

6. **`std::function` fields** — When a class has a `std::function` public field, bind it as a read-write property using lambdas. The `pybind11/functional.h` header enables automatic conversion of Python callables.

7. **Overloaded getter/setter (builder pattern)** — When a class has `int foo()` (getter) and `Class& foo(int)` (setter) with the same name, disambiguate with `static_cast` and expose as a property.

## Step 2: Generate pybind11 Binding Code

Create a `.cpp` file (e.g., `py_bindings.cpp`) with the pybind11 module definition.

### File structure:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>  // for std::function
#include <pybind11/stl/filesystem.h>  // for fs::path <-> str/Path

// Project headers
#include "project_header.hpp"

namespace py = pybind11;

PYBIND11_MODULE(module_name, m) {
    m.doc() = "Module docstring";
    // ... bindings ...
}
```

### CRITICAL: GIL release for expensive computations

Any C++ method that may run for more than a few milliseconds **must** release the Python GIL so other Python threads are not blocked. Use `py::call_guard<py::gil_scoped_release>()`:

```cpp
// GOOD — releases GIL during expensive C++ computation
.def("fillDistanceMatrix", &Problem::fillDistanceMatrix,
     py::call_guard<py::gil_scoped_release>(),
     "Fill the full distance matrix (releases GIL)")

.def("cluster", &Problem::cluster,
     py::call_guard<py::gil_scoped_release>(),
     "Run clustering (releases GIL)")
```

**Rules for GIL release:**
- Release for: distance matrix computation, clustering, any O(n²) or worse methods
- Do NOT release for: simple getters/setters, size(), name queries
- Do NOT release for methods that call back into Python (e.g., if `init_fun` is a Python callable) — this will deadlock. For such methods, acquire the GIL inside the callback:
```cpp
// For methods that may invoke Python callbacks during execution:
.def("cluster", [](Problem& self) {
    // If init_fun might be a Python callable, we need careful GIL handling
    py::gil_scoped_release release;
    // BUT: if self.init_fun calls Python, it must re-acquire GIL internally
    // pybind11's std::function wrapper handles this automatically
    self.cluster();
})
```

**Note on `std::function` callbacks and GIL**: When pybind11 wraps a Python callable into a `std::function`, the wrapper **automatically acquires the GIL** before calling the Python function. So it is safe to release the GIL on the outer method even if it eventually calls back into Python through a `std::function` field.

### CRITICAL: Lifetime management for reference-returning methods

Methods that return references to internal data (e.g., `const std::vector<double>& p_vec(size_t i)`) can create **dangling references** if the parent object is garbage-collected while Python still holds the returned value.

**Use `py::return_value_policy::reference_internal`** — this tells pybind11 to keep the parent alive as long as the returned reference exists:

```cpp
// DANGEROUS — dangling reference if Problem is GC'd
.def("p_vec", [](const Problem& self, size_t i) { return self.p_vec(i); })

// SAFE — keeps Problem alive while the returned list exists
.def("p_vec", [](const Problem& self, size_t i) { return self.p_vec(i); },
     py::return_value_policy::reference_internal,
     py::arg("index"))

// SAFEST — return a copy (slight overhead, but no lifetime issues)
.def("p_vec", [](const Problem& self, size_t i) -> std::vector<double> {
    return self.p_vec(i);  // returns by value = copy
}, py::arg("index"))
```

**Decision rule**: Use `reference_internal` for frequently-accessed large data. Use copy for small data or when the reference pattern is complex. When in doubt, **copy is always safe**.

### Type conversion rules (apply consistently):

| C++ Type | pybind11 Handling |
|----------|------------------|
| `std::vector<double>` | Auto-converted via `pybind11/stl.h`; also accept `numpy.ndarray` |
| `std::vector<std::vector<double>>` | List of lists; provide numpy helper if needed |
| `std::vector<int>` | Auto-converted via `pybind11/stl.h` |
| `std::vector<std::string>` | Auto-converted via `pybind11/stl.h` |
| `std::string` / `std::string_view` | Python `str` (auto) |
| `std::filesystem::path` | Python `str` or `pathlib.Path` (via `pybind11/stl/filesystem.h`) |
| `std::function<R(Args...)>` | Python callable (via `pybind11/functional.h`) |
| `arma::Mat<double>` | NumPy 2D array — use buffer protocol or copy helper (see below) |
| `arma::Col<double>` / `arma::Row<double>` | NumPy 1D array |
| `enum class E` | `py::enum_<E>` |
| Builder method `T& foo(int)` | Expose as regular method; also create Python property (see sugar layer) |

### Armadillo ↔ NumPy conversion helper

**Important**: Armadillo uses **column-major** storage, NumPy defaults to **row-major** (C-contiguous). Choose one of these approaches:

**Option A: Zero-copy with Fortran-order NumPy array** (preferred for read-only)
```cpp
#include <armadillo>

// Armadillo → NumPy (zero-copy, returns column-major / Fortran-order array)
py::array_t<double> arma_to_numpy(const arma::Mat<double>& mat) {
    return py::array_t<double>(
        {(ssize_t)mat.n_rows, (ssize_t)mat.n_cols},             // shape
        {(ssize_t)sizeof(double), (ssize_t)(mat.n_rows * sizeof(double))}, // strides (col-major)
        mat.memptr(),   // data pointer
        py::none()      // base object (no ownership)
    );
}
```

**Option B: Copy with row-major output** (safer, compatible with most NumPy ops)
```cpp
py::array_t<double> arma_to_numpy_copy(const arma::Mat<double>& mat) {
    auto result = py::array_t<double>({(ssize_t)mat.n_rows, (ssize_t)mat.n_cols});
    auto buf = result.mutable_unchecked<2>();
    for (ssize_t i = 0; i < (ssize_t)mat.n_rows; ++i)
        for (ssize_t j = 0; j < (ssize_t)mat.n_cols; ++j)
            buf(i, j) = mat(i, j);
    return result;
}

arma::Mat<double> numpy_to_arma(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.unchecked<2>();
    arma::Mat<double> mat(buf.shape(0), buf.shape(1));
    for (ssize_t i = 0; i < buf.shape(0); ++i)
        for (ssize_t j = 0; j < buf.shape(1); ++j)
            mat(i, j) = buf(i, j);
    return mat;
}
```

### Binding each class:
```cpp
py::class_<Namespace::ClassName>(m, "ClassName")
    // Constructors
    .def(py::init<>())
    .def(py::init<ParamType1, ParamType2>(), py::arg("param1"), py::arg("param2"))

    // Methods — always use py::arg() with descriptive names
    .def("method_name", &ClassName::method_name, py::arg("x"), py::arg("y"),
         "Docstring from C++ Doxygen comment")

    // Overloaded methods — use static_cast
    .def("overloaded", static_cast<ReturnType(ClassName::*)(ParamType) const>(
         &ClassName::overloaded), py::arg("param"))

    // Public fields — use def_readwrite for mutable, def_readonly for const
    .def_readwrite("field_name", &ClassName::field_name)

    // Properties from builder pattern (getter returns value, setter returns self&)
    .def_property("prop_name",
        [](const ClassName& self) { return self.prop_name(); },
        [](ClassName& self, int val) { self.prop_name(val); })

    // __repr__ and __len__
    .def("__repr__", [](const ClassName& self) {
        return "<ClassName(...)>";
    })
    .def("__len__", &ClassName::size);
```

### Binding each enum:
```cpp
py::enum_<Namespace::EnumName>(m, "EnumName")
    .value("Value1", Namespace::EnumName::Value1)
    .value("Value2", Namespace::EnumName::Value2)
    .export_values();  // Makes values accessible as module.Value1
```

### Binding template free functions (instantiate for double):
```cpp
m.def("function_name", &Namespace::functionName<double>,
      py::arg("x"), py::arg("y"),
      "Docstring");
```

### Naming convention (CasADi-style consistency)

The goal is cross-language consistency. Use the **same names** as C++ everywhere:

- **Classes**: Keep PascalCase (same as C++): `Problem`, `DataLoader`, `Data`
- **Methods/functions in binding layer**: Keep original C++ names (camelCase): `fillDistanceMatrix`, `distByInd`
- **Methods in sugar layer**: Provide **both** the original camelCase name AND a snake_case alias. The camelCase name is the primary (for cross-language consistency), the snake_case alias is for Pythonic convenience.
- **Enums**: Keep PascalCase for type, keep original value names
- **Module name**: Use lowercase, e.g., `dtwcpp` or project-specific name
- **Free functions**: In binding layer keep C++ names; in sugar layer provide snake_case aliases

## Step 3: Generate Python Sugar Module

Create a pure-Python file (e.g., `_wrapper.py` or `wrapper.py`) that imports the compiled C++ module and re-exports with Pythonic enhancements.

### Pattern for wrapping a class:
```python
"""Pythonic wrapper for the C++ library."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Union, Sequence, Callable

# Import the compiled C++ module
from . import _core as _cpp  # or whatever the compiled module is named


class ClassName:
    """Docstring from C++ Doxygen, expanded for Python users.

    Parameters
    ----------
    param1 : type
        Description.
    param2 : type, optional
        Description. Default: value.

    Examples
    --------
    >>> obj = ClassName(param1="value")
    >>> obj.do_something()
    """

    def __init__(self, param1: str = "", **kwargs):
        # Map Python kwargs to C++ constructor + setter calls
        self._cpp = _cpp.ClassName(param1)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"ClassName(name={self.name!r}, size={len(self)})"

    def __len__(self) -> int:
        return self._cpp.size()

    # Properties for builder-pattern methods
    @property
    def some_prop(self) -> int:
        return self._cpp.some_prop()

    @some_prop.setter
    def some_prop(self, value: int) -> None:
        self._cpp.some_prop(value)

    # Delegate methods with snake_case names
    def do_something(self, x: NDArray[np.float64]) -> float:
        """Compute something.

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        float
            The result.
        """
        return self._cpp.doSomething(x)
```

### Copy and pickle support

For classes that hold meaningful state, implement `__copy__`, `__deepcopy__`, and pickle support:

```python
import copy

class ClassName:
    # ... existing methods ...

    def __copy__(self):
        """Shallow copy — creates a new C++ object with same configuration."""
        new = ClassName.__new__(ClassName)
        new._cpp = _cpp.ClassName()  # new C++ object
        # Copy all configurable properties
        for prop in ['prop1', 'prop2']:
            setattr(new, prop, getattr(self, prop))
        return new

    def __deepcopy__(self, memo):
        """Deep copy — same as shallow since C++ objects own their data."""
        return copy.copy(self)

    def __getstate__(self):
        """Pickle support — serialize to dict of Python-native types."""
        return {
            'prop1': self.prop1,
            'prop2': self.prop2,
            # Only serialize properties that can reconstruct the object
        }

    def __setstate__(self, state):
        """Unpickle — reconstruct C++ object from saved state."""
        self._cpp = _cpp.ClassName()
        for key, value in state.items():
            setattr(self, key, value)
```

**When to implement**: Only for classes where copy/pickle semantics are meaningful (e.g., `Problem`, `DataLoader`). Skip for lightweight wrappers or enum-like types.

### Convenience functions at module level:
```python
def dtw_full(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Compute full Dynamic Time Warping distance.

    Parameters
    ----------
    x, y : array_like
        Input time series.

    Returns
    -------
    float
        DTW distance.
    """
    return _cpp.dtw_full(np.asarray(x, dtype=np.float64).tolist(),
                         np.asarray(y, dtype=np.float64).tolist())


def silhouette(problem: Problem) -> NDArray[np.float64]:
    """Compute silhouette scores for clustering result."""
    return np.array(_cpp.silhouette(problem._cpp))
```

### Settings as module-level functions:
```python
def set_data_path(path: Union[str, Path]) -> None:
    """Set the data directory path."""
    _cpp.set_data_path(str(path))

def set_results_path(path: Union[str, Path]) -> None:
    """Set the results output directory path."""
    _cpp.set_results_path(str(path))
```

## Step 4: Generate `__init__.py`

```python
"""Library Name — Python interface.

Provides Pythonic access to the C++ library with the same class names
and method signatures for seamless cross-language development.
"""

from .wrapper import (
    ClassName,
    OtherClass,
    EnumName,
    function_name,
    # ...
)

__all__ = [
    "ClassName",
    "OtherClass",
    "EnumName",
    "function_name",
    # ...
]

__version__ = "1.0.0"  # Should match C++ library version
```

## Step 5: Generate CMake Build Snippet

Create or update CMake to build the Python module:
```cmake
# Python bindings (optional)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core
        bindings/python/py_bindings.cpp
    )
    target_link_libraries(_core PRIVATE
        library_target
    )
    target_include_directories(_core PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
endif()
```

## Step 6: Generate Tests

Create a `test_bindings.py` using pytest that verifies:

```python
import pytest
import numpy as np

def test_class_construction():
    """Each class can be instantiated with default and parameterized constructors."""
    obj = module.ClassName()
    assert obj is not None

def test_enum_values():
    """All enum values are accessible."""
    assert module.EnumName.Value1 is not None

def test_property_roundtrip():
    """Properties can be set and retrieved."""
    obj = module.ClassName()
    obj.prop = 42
    assert obj.prop == 42

def test_method_call():
    """Methods produce expected results for known inputs."""
    # Use small, deterministic test data
    pass

def test_numpy_conversion():
    """NumPy arrays are correctly passed to/from C++."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    result = module.some_function(x, y)
    assert isinstance(result, (float, np.floating))

def test_exception_propagation():
    """C++ exceptions become Python exceptions."""
    with pytest.raises(RuntimeError):
        module.function_that_throws()
```

Adapt tests to the actual API being wrapped. Cover at least:
- Every class constructor
- Every enum
- Key methods with simple inputs
- Edge cases (empty data, wrong types)

## Step 7: Verify Cross-Language Consistency

Before finishing, verify the mapping is complete and consistent:

1. **Every public C++ class** has a Python binding AND a sugar wrapper
2. **Every public method** is accessible from Python (both original and snake_case names)
3. **Every enum** is exposed with all values
4. **Every free function** is wrapped
5. **Constructor signatures** match between C++ and Python (with Pythonic defaults/kwargs)
6. **Return types** are properly converted (especially vectors → lists/arrays, matrices → numpy)
7. **No raw pointers** leak to Python — use return value policies correctly:
   - `py::return_value_policy::reference_internal` for references to owned data
   - `py::return_value_policy::copy` for value returns
   - Default (automatic) for most cases

## Step 8: Generate Type Stub File (.pyi)

Create a `.pyi` type stub file alongside the compiled module so IDEs provide autocomplete and type checking:

```python
# _core.pyi — Type stubs for the compiled C++ module
from typing import List, Optional, Callable, overload
import numpy.typing as npt

class Problem:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, name: str) -> None: ...

    method: Method
    maxIter: int
    band: int
    N_repetition: int
    name: str

    def cluster(self) -> None: ...
    def fillDistanceMatrix(self) -> None: ...
    def size(self) -> int: ...
    # ... all other methods with type annotations

class DataLoader:
    def __init__(self, path: str = "") -> None: ...
    def load(self) -> Data: ...
    # ... builder methods

class Data:
    def __init__(self, data: List[List[float]], names: List[str]) -> None: ...
    def size(self) -> int: ...

class Method:
    Kmedoids: Method
    MIP: Method

class Solver:
    Gurobi: Solver
    HiGHS: Solver

def dtw_full(x: List[float], y: List[float]) -> float: ...
def dtw_banded(x: List[float], y: List[float], band: int) -> float: ...
def silhouette(problem: Problem) -> List[float]: ...
```

**Rules:**
- One `.pyi` file per compiled module (e.g., `_core.pyi` for `_core.so`)
- Include all classes, functions, enums with full type annotations
- Use `@overload` for methods with multiple signatures
- Place in the same directory as the compiled `.so`/`.pyd` file

## Cross-Language Reference: Side-by-Side Examples

When generating bindings, produce a side-by-side usage example showing the same workflow in C++, Python, and MATLAB. This serves as both documentation and a consistency check.

### Example (DTW clustering workflow):

**C++:**
```cpp
#include "dtwc/dtwc.hpp"
using namespace dtwc;

DataLoader loader;
loader.path("data/ECG200").startColumn(1).delimiter(',');
Data data = loader.load();

Problem prob("ECG200", loader);
prob.method = Method::Kmedoids;
prob.maxIter = 100;
prob.band = 10;
prob.fillDistanceMatrix();
prob.cluster();
auto sil = scores::silhouette(prob);
```

**Python:**
```python
import dtwc

loader = dtwc.DataLoader()
loader.path = "data/ECG200"
loader.startColumn = 1
loader.delimiter = ","
data = loader.load()

prob = dtwc.Problem("ECG200", loader)
prob.method = dtwc.Method.Kmedoids
prob.maxIter = 100
prob.band = 10
prob.fillDistanceMatrix()
prob.cluster()
sil = dtwc.silhouette(prob)
```

**MATLAB:**
```matlab
loader = dtwc.DataLoader();
loader.path = 'data/ECG200';
loader.startColumn = 1;
loader.delimiter = ',';
data = loader.load();

prob = dtwc.Problem('ECG200', loader);
prob.method = dtwc.Method.Kmedoids;
prob.maxIter = 100;
prob.band = 10;
prob.fillDistanceMatrix();
prob.cluster();
sil = dtwc.silhouette(prob);
```

Notice: The three versions are nearly identical — same class names, same method names, same property names. Only language syntax differs. **This is the goal.**

## Common Pitfalls to Avoid

1. **Missing `#include <pybind11/stl.h>`** — needed for automatic STL conversions
2. **Missing `#include <pybind11/functional.h>`** — needed for `std::function` parameters
3. **Forgetting `py::arg()` names** — always name parameters for Python clarity
4. **Not handling overloads** — use `static_cast` to disambiguate
5. **Thread-local storage in C++** — `thread_local` works differently under pybind11; be careful with functions using it
6. **Armadillo column-major vs NumPy row-major** — always transpose or copy correctly
7. **Lifetime issues** — use `py::keep_alive<>()` when Python objects reference C++ data
8. **Not exposing `__init__` with kwargs** — Python users expect keyword arguments
9. **Missing GIL release** — expensive C++ computations freeze all Python threads; use `py::call_guard<py::gil_scoped_release>()`
10. **Dangling references** — methods returning `const&` to internal data need `reference_internal` policy or copy
11. **Global mutable state** — static RNG, global paths etc. are NOT thread-safe with GIL released; document or protect with mutex
12. **Missing .pyi stubs** — without stubs, IDEs cannot provide autocomplete for compiled modules
