---
description: "Analyze C++ headers and generate MEX-based MATLAB bindings with OO wrapper classes (+package). Maintains OOP consistency (CasADi-style) so switching between C++ and MATLAB feels seamless."
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Task
---

# MATLAB Wrapper Generator

Generate complete MEX-based MATLAB bindings from C++ header files, following CasADi's cross-language design philosophy: **the same class names, same method signatures, same feel** — MATLAB code should read naturally while mapping 1:1 to the C++ API.

## Requirements

- **MATLAB** R2018a+ (for interleaved complex / `mxGetDoubles` API)
- **C++17** or newer
- Legacy C MEX API (`mex.h`, `matrix.h`) — do NOT use the C++ MEX API (`mex.hpp`/`mexAdapter.hpp`)

## Input

`$ARGUMENTS` = path(s) to C++ header file(s) to wrap. If empty, scan the project's `include/` and source directories for public headers.

## Step 0: Check for Existing Bindings

Before generating anything, search the project for existing MEX/MATLAB binding files:
```
Glob pattern: **/*mex*  **/+*/**/*.m  **/bindings/matlab/**
```

If existing bindings are found:
- Read them carefully to understand what is already bound
- **Extend** the existing files rather than creating new ones
- Preserve the existing package name and structure

## Step 1: Analyze C++ Headers

Read each specified header and extract a structured inventory (identical analysis to Python wrapper):

### For each class/struct:
- Class name, namespace, base class (if any)
- **Constructors**: parameter types, defaults
- **Public methods**: name, return type, parameter types, const-ness, overloads
- **Public fields**: name, type, default value
- **Builder pattern**: detect methods returning `ClassName&` (fluent setters) — these become MATLAB dependent properties
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

These require special handling in MEX bindings:

1. **`auto` return types** — Cannot take address directly. Call via a wrapper function or inline in the MEX dispatch.

2. **Rvalue-reference constructors** (`Class(T&& arg)`) — Construct temporaries in the MEX gateway and `std::move` them into the C++ constructor.

3. **Mutable lvalue reference parameters** (`void foo(std::vector<int>& v)`) — Copy the MATLAB data into a C++ vector, call the method, then copy back to a new output `mxArray`. The input cannot be modified in-place via MEX.

4. **Type aliases** (e.g., `using data_t = double`) — Check the project's type alias and instantiate templates for the actual underlying type.

5. **`thread_local` scratch buffers** — Common in DTW implementations. MATLAB MEX runs single-threaded by default, so thread_local works correctly. However, if OpenMP is enabled inside the MEX library, each OpenMP thread gets its own buffer. Document this behavior.

6. **`std::function` fields** — Must convert MATLAB function handles to C++ `std::function` via `mexCallMATLAB`/`feval`. Use `mexMakeArrayPersistent` to prevent MATLAB from garbage-collecting the handle.

7. **Overloaded getter/setter (builder pattern)** — Expose as separate `ClassName_get_PropName` and `ClassName_set_PropName` MEX commands, mapped to MATLAB Dependent properties.

8. **1-indexed vs 0-indexed** — All indices crossing the MEX boundary must be converted. MATLAB uses 1-based indexing, C++ uses 0-based. Subtract 1 on input, add 1 on output.

## Step 2: Generate MEX Gateway

Create a single MEX C++ gateway file (e.g., `mex_gateway.cpp`) that dispatches all calls.

### Architecture: Handle-Based Object Management

Use a handle map to track C++ objects from MATLAB:

```cpp
// Use the legacy C MEX API (wider compatibility than C++ MEX API)
#include "mex.h"
#include "matrix.h"

// Project headers
#include "project_header.hpp"

#include <unordered_map>
#include <memory>
#include <string>
#include <cstdint>

namespace {

// Handle registry: maps uint64 handles to C++ objects
// CRITICAL: MATLAB stores handles as double. double has 53 bits of mantissa,
// so handles above 2^53 (~9e15) silently lose precision. The counter below
// will never reach 2^53 in practice, but we add a static_assert guard.
template <typename T>
class HandleManager {
    static std::unordered_map<uint64_t, std::shared_ptr<T>> handles_;
    static uint64_t next_handle_;
public:
    static uint64_t create(std::shared_ptr<T> obj) {
        uint64_t h = ++next_handle_;
        // Guard: double can only represent integers exactly up to 2^53
        static_assert(sizeof(double) == 8, "double must be 64-bit");
        if (h > (1ULL << 53)) {
            mexErrMsgIdAndTxt("wrapper:handleOverflow",
                "Handle counter exceeded 2^53 — double precision limit");
        }
        handles_[h] = std::move(obj);
        return h;
    }

    static std::shared_ptr<T>& get(uint64_t h) {
        auto it = handles_.find(h);
        if (it == handles_.end())
            mexErrMsgIdAndTxt("wrapper:invalidHandle", "Invalid object handle");
        return it->second;
    }

    static void destroy(uint64_t h) {
        handles_.erase(h);
    }

    static void clear() {
        handles_.clear();
    }
};

template <typename T>
std::unordered_map<uint64_t, std::shared_ptr<T>> HandleManager<T>::handles_;
template <typename T>
uint64_t HandleManager<T>::next_handle_ = 0;

} // anonymous namespace
```

### Type conversion rules:

| C++ Type | MATLAB Type | Conversion |
|----------|-------------|-----------|
| `double` | `double` scalar | Direct |
| `int` | `double` scalar | Cast `(int)mxGetScalar(prhs[i])` |
| `bool` | `logical` | `mxIsLogicalScalarTrue(prhs[i])` |
| `std::string` | `char` array or `string` | `mxArrayToString(prhs[i])` |
| `std::string_view` | `char` array | `mxArrayToString(prhs[i])` |
| `std::vector<double>` | `double` column vector | `mxGetDoubles()` + size |
| `std::vector<int>` | `double` vector | Cast each element |
| `std::vector<std::string>` | Cell array of strings | Iterate cells |
| `std::vector<std::vector<double>>` | Cell array of double vectors | Iterate cells |
| `std::filesystem::path` | `char` or `string` | `mxArrayToString()` |
| `arma::Mat<double>` | `double` matrix | Share memory (both column-major!) |
| `arma::Col<double>` | `double` column vector | Share memory |
| `enum class E` | `char`/`string` | String-to-enum lookup map |
| `std::function<R(Args...)>` | Function handle | Create C++ lambda calling `mexCallMATLAB` |

**Key advantage**: Armadillo and MATLAB both use **column-major** storage, so matrix data can be shared without transposing (unlike NumPy).

### Armadillo ↔ MATLAB helpers:
```cpp
// MATLAB matrix → Armadillo (zero-copy, READ-ONLY — do NOT modify through this!)
// WARNING: The const_cast is safe only for read-only access. If any Armadillo
// operation triggers reallocation or modification, it writes into MATLAB memory
// which can corrupt state. Use the copy version for any mutable operations.
arma::Mat<double> mxArray_to_arma_readonly(const mxArray* mx) {
    const double* data = mxGetDoubles(mx);
    size_t rows = mxGetM(mx);
    size_t cols = mxGetN(mx);
    return arma::Mat<double>(const_cast<double*>(data), rows, cols, false, true);
}

// MATLAB matrix → Armadillo (safe copy for mutable use)
arma::Mat<double> mxArray_to_arma(const mxArray* mx) {
    const double* data = mxGetDoubles(mx);
    size_t rows = mxGetM(mx);
    size_t cols = mxGetN(mx);
    return arma::Mat<double>(data, rows, cols); // copies data
}

// Armadillo → MATLAB matrix (copy)
mxArray* arma_to_mxArray(const arma::Mat<double>& mat) {
    mxArray* mx = mxCreateDoubleMatrix(mat.n_rows, mat.n_cols, mxREAL);
    std::memcpy(mxGetDoubles(mx), mat.memptr(), mat.n_elem * sizeof(double));
    return mx;
}

// std::vector<double> → MATLAB column vector
mxArray* vec_to_mxArray(const std::vector<double>& v) {
    mxArray* mx = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
    std::memcpy(mxGetDoubles(mx), v.data(), v.size() * sizeof(double));
    return mx;
}

// MATLAB vector → std::vector<double>
std::vector<double> mxArray_to_vec(const mxArray* mx) {
    const double* data = mxGetDoubles(mx);
    size_t n = mxGetNumberOfElements(mx);
    return std::vector<double>(data, data + n);
}

// std::vector<int> → MATLAB double vector
mxArray* ivec_to_mxArray(const std::vector<int>& v) {
    mxArray* mx = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
    double* data = mxGetDoubles(mx);
    for (size_t i = 0; i < v.size(); ++i) data[i] = static_cast<double>(v[i]);
    return mx;
}
```

### String-dispatched method routing:
```cpp
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("wrapper:badInput", "First argument must be a command string");

    std::string cmd = mxArrayToString(prhs[0]);

    // ===== ClassName =====
    if (cmd == "ClassName_new") {
        // Constructor: handle = mex('ClassName_new', args...)
        auto obj = std::make_shared<Namespace::ClassName>(/* convert args */);
        uint64_t h = HandleManager<Namespace::ClassName>::create(obj);
        plhs[0] = mxCreateDoubleScalar(static_cast<double>(h));
        return;
    }
    if (cmd == "ClassName_delete") {
        // Destructor: mex('ClassName_delete', handle)
        uint64_t h = static_cast<uint64_t>(mxGetScalar(prhs[1]));
        HandleManager<Namespace::ClassName>::destroy(h);
        return;
    }
    if (cmd == "ClassName_methodName") {
        // Apply longjmp-safe pattern to EVERY dispatch (see CRITICAL section below)
        std::string error_msg;
        try {
            uint64_t h = static_cast<uint64_t>(mxGetScalar(prhs[1]));
            auto& obj = HandleManager<Namespace::ClassName>::get(h);
            // Convert input args from prhs[2], prhs[3], ...
            auto result = obj->methodName(/* converted args */);
            // Convert result to plhs[0]
        } catch (const std::exception& e) {
            error_msg = e.what();
        } catch (...) {
            error_msg = "Unknown C++ exception";
        }
        if (!error_msg.empty())
            mexErrMsgIdAndTxt("wrapper:cppException", "%s", error_msg.c_str());
        return;
    }

    // ===== Free functions =====
    if (cmd == "function_name") {
        // Convert args, call, convert result
        return;
    }

    mexErrMsgIdAndTxt("wrapper:unknownCommand", "Unknown command: %s", cmd.c_str());
}
```

### CRITICAL: Exception handling (longjmp-safe pattern)

**Every** MEX command dispatch must be wrapped in try/catch. However, `mexErrMsgIdAndTxt` calls `longjmp` internally, which **skips C++ destructors** (stack unwinding). If you call it inside a try/catch block, any RAII objects (shared_ptr, string, vector) in that scope will leak.

**WRONG — destructors skipped:**
```cpp
try {
    auto result = obj->expensiveMethod();  // RAII objects on stack
    mexErrMsgIdAndTxt("...", "...");  // longjmp skips destructors!
} catch (const std::exception& e) {
    mexErrMsgIdAndTxt("...", "%s", e.what());  // longjmp skips catch-scope destructors!
}
```

**CORRECT — capture error, exit scope, then call mexErrMsgIdAndTxt:**
```cpp
std::string error_msg;
try {
    // All C++ objects with destructors live here
    auto result = obj->expensiveMethod();
    // ... convert result to plhs[0] ...
} catch (const std::exception& e) {
    error_msg = e.what();
} catch (...) {
    error_msg = "Unknown C++ exception";
}
// Now OUTSIDE the try/catch — all C++ destructors have run
if (!error_msg.empty()) {
    mexErrMsgIdAndTxt("wrapper:cppException", "%s", error_msg.c_str());
}
```

**Apply this pattern to EVERY command dispatch in the MEX gateway.** The pattern ensures:
1. All C++ RAII objects are destroyed before longjmp
2. The error message is captured by value (not a dangling pointer)
3. `mexErrMsgIdAndTxt` is only called after normal scope exit

### Cleanup on MEX unload

Register a `mexAtExit` callback to destroy all handles when the MEX file is cleared (`clear mex`):
```cpp
static bool cleanup_registered = false;

void cleanup_all_handles() {
    // Destroy all managed objects
    // For each HandleManager<T>, call a static clear() method
}

// Call this once at the top of mexFunction:
if (!cleanup_registered) {
    mexAtExit(cleanup_all_handles);
    cleanup_registered = true;
}
```

### Enum string-to-C++ lookup:
```cpp
// NOTE: Throw C++ exceptions from helper functions — NOT mexErrMsgIdAndTxt.
// The outer try/catch will convert them to MEX errors via the longjmp-safe pattern.
Namespace::EnumName string_to_EnumName(const std::string& s) {
    if (s == "Value1") return Namespace::EnumName::Value1;
    if (s == "Value2") return Namespace::EnumName::Value2;
    throw std::invalid_argument("Unknown EnumName value: " + s);
}

std::string EnumName_to_string(Namespace::EnumName e) {
    switch (e) {
        case Namespace::EnumName::Value1: return "Value1";
        case Namespace::EnumName::Value2: return "Value2";
    }
    return "Unknown";
}
```

### std::function from MATLAB function handle:
```cpp
// For callbacks like std::function<void(ClassName&)>:
// Store the mxArray* (function handle) and use mexCallMATLAB
// Note: this requires the function handle to persist — use mxDuplicateArray + mexMakeArrayPersistent

std::function<void(Namespace::ClassName&)> make_callback(const mxArray* fh) {
    mxArray* persistent_fh = mxDuplicateArray(fh);
    mexMakeArrayPersistent(persistent_fh);

    return [persistent_fh](Namespace::ClassName& obj) {
        // Create a temporary handle for the object
        uint64_t temp_h = HandleManager<Namespace::ClassName>::create(
            std::shared_ptr<Namespace::ClassName>(&obj, [](auto*){})); // non-owning
        mxArray* arg = mxCreateDoubleScalar(static_cast<double>(temp_h));
        mxArray* rhs[2] = { const_cast<mxArray*>(persistent_fh), arg };
        mexCallMATLAB(0, nullptr, 2, rhs, "feval");
        HandleManager<Namespace::ClassName>::destroy(temp_h);
        mxDestroyArray(arg);
    };
}
```

## Step 3: Generate MATLAB OO Wrapper Classes

Create a `+packagename/` directory (e.g., `+dtwc/`) with one `.m` file per C++ class.

### Template for each class:
```matlab
classdef ClassName < handle
    %CLASSNAME Brief description from C++ Doxygen.
    %   Detailed description.
    %
    %   Example:
    %       obj = packagename.ClassName('param1', value1);
    %       obj.doSomething(data);

    properties (Access = private, Hidden)
        ObjectHandle  % uint64 handle to C++ object
    end

    properties (Dependent)
        % Properties mapped from C++ builder-pattern getters/setters
        PropName
    end

    methods
        function obj = ClassName(varargin)
            %CLASSNAME Construct a ClassName object.
            %   obj = ClassName() creates default object.
            %   obj = ClassName(arg1, arg2) creates with positional args.
            %   obj = ClassName('Name', Value, ...) creates with options.

            % Parse inputs
            p = inputParser;
            p.addOptional('arg1', default1);
            p.addParameter('ParamName', defaultVal);
            p.parse(varargin{:});

            % Call MEX constructor
            obj.ObjectHandle = packagename_mex('ClassName_new', ...
                p.Results.arg1, p.Results.ParamName);
        end

        function delete(obj)
            %DELETE Destructor — releases C++ object.
            if ~isempty(obj.ObjectHandle)
                packagename_mex('ClassName_delete', obj.ObjectHandle);
                obj.ObjectHandle = [];
            end
        end

        % --- Dependent property access ---
        function val = get.PropName(obj)
            val = packagename_mex('ClassName_get_PropName', obj.ObjectHandle);
        end

        function set.PropName(obj, val)
            packagename_mex('ClassName_set_PropName', obj.ObjectHandle, val);
        end

        % --- Methods ---
        function result = doSomething(obj, x, y)
            %DOSOMETHING Brief description.
            %   result = obj.doSomething(x, y)
            result = packagename_mex('ClassName_doSomething', ...
                obj.ObjectHandle, x, y);
        end

        % --- Display ---
        function disp(obj)
            fprintf('  ClassName with %d elements\n', obj.size());
            fprintf('    Name: %s\n', obj.name);
        end

        function n = numel(obj)
            %NUMEL Number of elements.
            n = packagename_mex('ClassName_size', obj.ObjectHandle);
        end

        function n = length(obj)
            %LENGTH Alias for numel.
            n = numel(obj);
        end
    end

    methods (Static)
        % Static factory methods if applicable
    end
end
```

### Template for each enum:
```matlab
classdef EnumName
    %ENUMNAME C++ enum class wrapper.
    %   Use as: packagename.EnumName.Value1

    enumeration
        Value1 ('Value1')
        Value2 ('Value2')
    end

    properties (SetAccess = immutable, Hidden)
        CppString  % String passed to MEX for C++ conversion
    end

    methods
        function obj = EnumName(s)
            obj.CppString = s;
        end
    end
end
```

### Template for free functions:
```matlab
function result = function_name(x, y, varargin)
    %FUNCTION_NAME Brief description.
    %   result = packagename.function_name(x, y)
    %   result = packagename.function_name(x, y, 'Band', 10)
    %
    %   Parameters
    %   ----------
    %   x : double vector
    %       First time series.
    %   y : double vector
    %       Second time series.
    %
    %   Returns
    %   -------
    %   result : double
    %       The computed distance.

    p = inputParser;
    p.addOptional('extra_param', default);
    p.parse(varargin{:});

    result = packagename_mex('function_name', x, y, p.Results.extra_param);
end
```

## Step 4: Generate Build Script

Create a `compile_mex.m` script:

```matlab
function compile_mex()
    %COMPILE_MEX Build the MEX gateway for the library.
    %   compile_mex() compiles the MEX file with proper includes and libraries.

    % Paths — adjust for your installation
    cpp_include = fullfile(fileparts(mfilename('fullpath')), '..', 'include');
    cpp_lib_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'build');

    % Source file
    src = fullfile(fileparts(mfilename('fullpath')), 'mex_gateway.cpp');

    % Compiler flags
    cxx_flags = 'CXXFLAGS="$CXXFLAGS -std=c++17"';

    % Include directories
    includes = ['-I' cpp_include];

    % Library directories and libraries
    lib_dir = ['-L' cpp_lib_dir];
    libs = '-llibrary_name';  % Adjust to actual library name

    % Optional: Armadillo
    % arma_include = '-I/path/to/armadillo/include';
    % arma_lib = '-larmadillo';

    % Compile
    fprintf('Compiling MEX gateway...\n');
    mex(cxx_flags, includes, lib_dir, libs, src, '-output', 'packagename_mex');
    fprintf('Done. MEX file created: %s\n', ...
        fullfile(pwd, ['packagename_mex.' mexext]));
end
```

## Step 5: Generate Package Layout

Ensure the `+packagename/` directory structure is:

```text
+packagename/
    ClassName.m          % OO wrapper class
    OtherClass.m         % Another wrapper class
    EnumName.m           % Enum wrapper
    function_name.m      % Free function wrapper
    private/
        packagename_mex.mexw64  % Compiled MEX (or .mexa64 on Linux)
```

Or alternatively, place the MEX file alongside the package and reference it without the package prefix.

## Step 6: Generate Tests

Create a `test_bindings.m` test script using MATLAB's unit testing framework:

```matlab
classdef test_bindings < matlab.unittest.TestCase
    methods (Test)
        function testClassConstruction(testCase)
            obj = packagename.ClassName();
            testCase.verifyNotEmpty(obj);
        end

        function testEnumValues(testCase)
            e = packagename.EnumName.Value1;
            testCase.verifyNotEmpty(e);
        end

        function testPropertyRoundtrip(testCase)
            obj = packagename.ClassName();
            obj.PropName = 42;
            testCase.verifyEqual(obj.PropName, 42);
        end

        function testMethodCall(testCase)
            % Use small, deterministic test data
            x = [1.0; 2.0; 3.0];
            y = [1.0; 2.0; 3.0];
            result = packagename.function_name(x, y);
            testCase.verifyClass(result, 'double');
        end

        function testExceptionPropagation(testCase)
            testCase.verifyError(@() packagename.function_that_throws(), ...
                'wrapper:cppException');
        end

        function testIndexConversion(testCase)
            % Verify 1-based MATLAB indices map correctly to 0-based C++
            obj = packagename.ClassName();
            % ... test index-dependent methods
        end
    end
end
```

Adapt tests to the actual API. Cover at least:
- Every class constructor
- Every enum
- Key methods with simple inputs
- Index conversion correctness
- Edge cases (empty data, wrong types)

### Naming convention (CasADi-style consistency)

Use the **same method names** as C++ for cross-language consistency:
- **Classes**: Keep PascalCase: `Problem`, `DataLoader`, `Data`
- **Methods**: Keep original C++ names (camelCase): `fillDistanceMatrix`, `distByInd`
- **Properties**: Map builder getters/setters using MATLAB conventions but keep C++ names where possible
- **Enums**: Keep PascalCase for type and original value names
- **Free functions**: Keep C++ names; MATLAB conventions allow camelCase

## Step 7: Verify Cross-Language Consistency

Before finishing, verify the mapping is complete and consistent:

1. **Every public C++ class** has a MATLAB classdef wrapper
2. **Every public method** is accessible from MATLAB
3. **Every enum** is exposed with all values
4. **Every free function** is wrapped as a package function
5. **Constructor signatures** support both positional and name-value pair syntax
6. **Matrix data** uses MATLAB's column-major convention (matches Armadillo — no transposition needed)
7. **Handle cleanup** — destructors are called when MATLAB objects go out of scope
8. **Error handling** — C++ exceptions become `mexErrMsgIdAndTxt` with descriptive IDs

## Step 8: MATLAB save/load Support

For handle classes wrapping C++ objects, implement `saveobj`/`loadobj` so users can `save`/`load` objects to `.mat` files:

```matlab
methods
    function s = saveobj(obj)
        %SAVEOBJ Serialize to struct for save().
        s.prop1 = obj.PropName;
        s.prop2 = obj.OtherProp;
        % Only save properties that can reconstruct the object.
        % Do NOT save the ObjectHandle — it is transient.
    end
end

methods (Static)
    function obj = loadobj(s)
        %LOADOBJ Reconstruct from saved struct.
        obj = packagename.ClassName();
        obj.PropName = s.prop1;
        obj.OtherProp = s.prop2;
    end
end
```

**Rules:**
- Save only MATLAB-native types (double, string, cell, struct) — not the C++ handle
- The loaded object creates a fresh C++ object and re-applies saved properties
- Document which properties survive save/load in the class help text

## IMPORTANT: OpenMP Restrictions in MEX

If the C++ library uses OpenMP internally (e.g., for parallel distance matrix computation):

1. **MATLAB manages its own thread pool** — OpenMP threads inside MEX can conflict. Set thread count conservatively:
```cpp
// At the top of mexFunction, limit OpenMP threads:
#ifdef _OPENMP
    omp_set_num_threads(std::min(omp_get_max_threads(), 4));  // conservative default
#endif
```

2. **Do NOT use `omp_set_nested(true)`** — nested parallelism in MEX is unreliable

3. **On macOS**: MATLAB ships its own `libomp.dylib` which may conflict with the system/Homebrew one. Link statically or use `-Xclang -fopenmp` with MATLAB's own copy.

4. **Thread-local storage**: With OpenMP enabled, each OpenMP worker gets its own `thread_local` buffer. This is correct behavior but increases memory usage proportionally to thread count.

5. **Graceful fallback**: Always compile MEX with OpenMP **optional**. The C++ code should have serial fallbacks when OpenMP is unavailable:
```cpp
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n; ++i) { /* ... */ }
```

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

1. **Memory leaks** — Always pair `ClassName_new` with `ClassName_delete` in the destructor
2. **Handle invalidation** — Check handle validity before every method call
3. **Armadillo memory sharing** — The zero-copy `arma::Mat` from `mxArray` must not outlive the `mxArray`
4. **MATLAB string vs char** — Support both `char` arrays and `string` scalars for string inputs
5. **Integer types** — MATLAB passes everything as `double`; cast properly in MEX
6. **Function handles for callbacks** — Must use `mexMakeArrayPersistent` to prevent GC
7. **Thread safety** — MATLAB is single-threaded; MEX calls must not hold locks
8. **Name collisions** — Use `+packagename` namespace to avoid conflicts with MATLAB built-ins
9. **Copy semantics** — MATLAB uses value semantics by default; use `handle` base class for C++ object wrappers
10. **1-indexed vs 0-indexed** — Convert all indices at the MEX boundary (MATLAB is 1-based, C++ is 0-based)
11. **mexErrMsgIdAndTxt inside try/catch** — Calls `longjmp`, skipping C++ destructors. Always capture error, exit scope, THEN call
12. **Handle stored as double** — `uint64_t` handles above 2^53 lose precision when stored in MATLAB `double`. Use the guarded HandleManager
13. **OpenMP in MEX** — Limit thread count; avoid nested parallelism; handle macOS `libomp` conflicts
