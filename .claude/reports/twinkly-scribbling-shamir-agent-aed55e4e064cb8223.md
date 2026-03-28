# Deep Research Report: Parallel Computing, Autodiff, mdspan, GPU DTW, and CUDA Redistribution

**Date:** 2026-03-27
**Purpose:** Technology survey for DTWC++ project decisions

---

## 1. C++ Parallel Computing Libraries for HPC/SLURM

### 1.1 Thrust (NVIDIA CCCL)

- **Repository:** https://github.com/NVIDIA/cccl (unified repo since ~2020; formerly separate NVIDIA/thrust)
- **License:** Apache 2.0
- **Stars:** ~2,242
- **Status:** Active, part of CUDA Core Compute Libraries (CCCL) alongside CUB and libcudacxx
- **CPU Backends:** Yes -- Thrust explicitly supports **OpenMP** and **TBB** as configurable backends alongside CUDA. From the README: "Thrust's high-level interface greatly enhances programmer productivity while enabling performance portability between GPUs and multicore CPUs via configurable backends that allow using multiple parallel programming frameworks (such as CUDA, TBB, and OpenMP)."
- **Can run without CUDA?** Technically yes (CPU backends), but the build system and headers are deeply CUDA-centric. Thrust is shipped *with* the CUDA Toolkit. Building Thrust without nvcc requires careful configuration. It is not designed as a standalone CPU library.
- **CMake integration:** Comes with CUDA Toolkit; can also be fetched via CPM/FetchContent from the CCCL repo.
- **DTW suitability:** Thrust's high-level parallel primitives (sort, reduce, transform, scan) are well-suited for distance matrix computation but not for the DTW recurrence itself (which has diagonal dependencies). CUB (also in CCCL) provides lower-level block/warp primitives that could be used for anti-diagonal kernel parallelism.
- **Recommendation for DTWC++:** Not ideal as the primary parallelism layer due to tight CUDA coupling. Better used as a GPU-specific backend if CUDA support is added.

### 1.2 oneTBB (Intel Threading Building Blocks)

- **Repository:** https://github.com/uxlfoundation/oneTBB (moved from oneapi-src to UXL Foundation)
- **License:** Apache 2.0
- **Stars:** ~6,600
- **Status:** Very mature, actively maintained, part of UXL Foundation / oneAPI ecosystem
- **CPU+GPU:** CPU only (multi-core). No GPU support. Designed for task-based parallelism on CPUs.
- **vs. OpenMP:** oneTBB excels at **dynamic task parallelism** and work-stealing. OpenMP is better for simple data-parallel loops but struggles with irregular/nested parallelism. oneTBB is more composable -- nested parallelism works correctly without oversubscription. OpenMP can oversubscribe threads in nested parallel regions unless carefully managed.
- **CMake integration:** Excellent. Standard `find_package(TBB)` or FetchContent/CPM. Widely packaged in system package managers.
- **Community:** Very large; used by GCC's libstdc++ for `std::execution` parallel algorithms.
- **DTW suitability:** Excellent for CPU distance matrix computation. `tbb::parallel_for` with automatic load balancing is ideal for the triangular distance matrix (where work per row varies). Task-based approach handles the "compute DTW for pair (i,j)" pattern well.
- **Recommendation for DTWC++:** **Strong candidate for CPU parallelism backend.** More portable and composable than OpenMP. Could serve as both a direct parallel_for provider and as the backend for std::execution policies.

### 1.3 Kokkos (Sandia National Labs)

- **Repository:** https://github.com/kokkos/kokkos
- **License:** Apache 2.0 with LLVM Exceptions
- **Stars:** ~2,489
- **Status:** Very mature, Linux Foundation project, current release 4.7.01
- **CPU+GPU:** Yes -- supports CUDA, HIP (AMD), SYCL, HPX, OpenMP, and C++ threads as backends. True performance portability across CPU/GPU/accelerator.
- **Complexity to integrate:** Moderate-to-high. Kokkos is a full programming model (Kokkos::View for data, Kokkos::parallel_for/reduce/scan for execution, memory spaces, execution spaces). Requires restructuring data layouts around Kokkos::View. Non-trivial for an existing codebase. Typically integrated as a CMake subproject or pre-installed.
- **CMake integration:** Native CMake support. Can be used as a subdirectory or installed package.
- **Community:** Large HPC community (~500 forks). Many DOE national lab codes use it.
- **DTW suitability:** Kokkos::parallel_for is excellent for distance matrix computation. Kokkos::View with appropriate memory layout (LayoutLeft for column-major, LayoutRight for row-major) can hold series data efficiently. GPU execution requires porting DTW kernels to Kokkos functors.
- **Recommendation for DTWC++:** **Overkill for current scope** but future-proof if GPU portability across vendors (NVIDIA + AMD + Intel) is desired. The integration cost is high for what DTWC++ currently needs. Better suited if the project grows to target exascale or multi-vendor GPU systems.

### 1.4 RAJA (Lawrence Livermore)

- **Repository:** https://github.com/LLNL/RAJA
- **License:** BSD 3-Clause
- **Stars:** ~571
- **Status:** Active, used in LLNL production codes
- **CPU+GPU:** Yes -- supports OpenMP, CUDA, HIP, SYCL backends. Similar goals to Kokkos.
- **vs. Kokkos:** RAJA focuses more on the loop abstraction (portable parallel for) while Kokkos provides a more complete ecosystem (views, memory spaces, team policies). RAJA is somewhat easier to adopt incrementally since it primarily wraps loops. Kokkos has a larger community and broader adoption.
- **CMake integration:** Native CMake. Requires `--recursive` clone due to submodules (camp, blt).
- **DTW suitability:** Similar to Kokkos -- good for distance matrix parallel_for, but requires kernel porting.
- **Recommendation for DTWC++:** **Not recommended.** Smaller community than Kokkos with similar integration cost. If you want this level of portability, Kokkos is the better choice.

### 1.5 HPX (STEllAR Group)

- **Repository:** https://github.com/STEllAR-GROUP/hpx
- **License:** Boost Software License 1.0 (BSL-1.0) -- very permissive
- **Stars:** ~2,815
- **Status:** Mature, member of High-Performance Software Foundation (HPSF)
- **CPU+GPU:** Primarily CPU. Implements the C++ Standard Library for parallelism and concurrency. Can serve as a backend for Kokkos. No direct GPU kernel support.
- **Key features:** Implements `std::execution` policies, futures, dataflow, distributed computing (like MPI but with C++ standard semantics). Supports "hundreds of millions of threads" via lightweight tasks.
- **DTW-relevant features:** Implements C++ standard parallel algorithms with proper futures/dataflow semantics. Could express the distance matrix computation as a DAG of async tasks. Distributed computing support could be interesting for massive datasets across cluster nodes.
- **CMake integration:** Standard CMake. Relatively heavyweight dependency.
- **Recommendation for DTWC++:** **Not recommended for current scope.** HPX shines for distributed computing and extreme concurrency. DTWC++ is primarily node-level compute. OpenMP or oneTBB are simpler choices for shared-memory parallelism.

### 1.6 Taskflow

- **Repository:** https://github.com/taskflow/taskflow
- **License:** MIT
- **Stars:** ~11,844 (highest among all candidates!)
- **Status:** Very active, excellent documentation and community
- **CPU+GPU:** CPU task parallelism + GPU (cudaFlow for CUDA kernel orchestration). GPU support is for launching kernels, not for writing portable kernels.
- **Key features:** Header-only. Modern C++17. Work-stealing scheduler. Task DAG programming model with conditional tasking, subflows, composition. CUDA kernel graph support via cudaFlow. Profiler (TFProf).
- **CMake integration:** Header-only -- trivially integrable via CPM/FetchContent or just include the header directory.
- **DTW suitability:** Very well-suited for **task-graph based distance matrix computation**. Can express "compute DTW(i,j) for all pairs" as a task graph with natural load balancing. cudaFlow could orchestrate GPU DTW kernels. The task DAG model is perfect for expressing dependencies in the anti-diagonal DTW parallelism.
- **Recommendation for DTWC++:** **Top recommendation for CPU parallelism.** Header-only, MIT license, massive community, trivial to integrate, supports both CPU and CUDA task orchestration. The work-stealing scheduler is ideal for the irregular workload of DTW distance matrix computation (where different pairs may have different series lengths).

### 1.7 std::execution (C++17/20/23/26)

- **C++17:** Introduced execution policies (`std::execution::par`, `std::execution::par_unseq`). Parallel versions of standard algorithms (sort, for_each, transform, reduce, etc.).
- **C++23:** Added `std::execution` sender/receiver model (P2300) -- but this is the *asynchronous execution* model, not the parallel algorithms.
- **C++26:** P2300 `std::execution` (senders/receivers) accepted into C++26 standard.
- **Reference implementation:** NVIDIA/stdexec (2,281 stars), Apache 2.0 license.

**Compiler support for C++17 parallel algorithms:**
- **GCC (libstdc++):** Supported since GCC 9, but **requires oneTBB as external dependency** (uses Intel oneDPL/TBB under the hood). Without TBB installed, `std::execution::par` silently falls back to sequential execution.
- **MSVC:** Full support since VS 2017 15.7. Uses a built-in thread pool. **Best out-of-the-box support.**
- **Clang (libc++):** As of 2025/2026, libc++ support for parallel algorithms is still **incomplete/experimental**. Partial support via PSTL (Parallel STL) with TBB backend.
- **Intel (oneAPI DPC++/C++):** Full support via oneDPL.

**Recommendation for DTWC++:** std::execution parallel algorithms are useful for simple parallel-for patterns but have inconsistent compiler support. The DTWC++ project already targets MSVC (Windows) where support is good, but for cross-platform reliability, wrapping parallelism behind an abstraction that can use OpenMP, TBB, or Taskflow is safer.

### Summary Table: Parallel Libraries

| Library | License | Stars | Header-only | CPU | GPU | CMake Ease | DTW Fit | Recommended |
|---------|---------|-------|-------------|-----|-----|------------|---------|-------------|
| Thrust/CCCL | Apache 2.0 | 2,242 | No | Via backends | CUDA | Medium | Medium | GPU only |
| oneTBB | Apache 2.0 | 6,600 | No | Yes | No | Easy | High | Yes (CPU) |
| Kokkos | Apache 2.0 + LLVM | 2,489 | No | Yes | Multi-vendor | Medium | High | Future |
| RAJA | BSD 3-Clause | 571 | No | Yes | Multi-vendor | Medium | Medium | No |
| HPX | BSL 1.0 | 2,815 | No | Yes | No | Medium | Low | No |
| **Taskflow** | **MIT** | **11,844** | **Yes** | **Yes** | **CUDA orchestration** | **Trivial** | **High** | **Yes (top pick)** |
| std::execution | N/A (standard) | N/A | Yes | Yes | No | N/A | Medium | Supplement |

### Top Recommendations for DTWC++

1. **Primary CPU parallelism: Taskflow** -- header-only, MIT, trivial integration, work-stealing, CUDA orchestration support. Use `tf::Taskflow` and `tf::Executor` for distance matrix computation.
2. **Keep OpenMP as optional fallback** -- for HPC/SLURM environments where OpenMP is ubiquitous and Taskflow may not be installed.
3. **Future GPU portability: Kokkos** -- if multi-vendor GPU support becomes important.
4. **GPU-specific: Write CUDA kernels directly** for DTW, orchestrated by Taskflow's cudaFlow.

---

## 2. Autodiff Libraries for C++

### 2.1 Enzyme (MIT / LLVM)

- **Repository:** https://github.com/EnzymeAD/Enzyme
- **License:** Apache 2.0 with LLVM Exceptions (same as LLVM)
- **Stars:** ~1,568
- **How it works:** LLVM compiler plugin that differentiates arbitrary LLVM IR. Works *after* optimization, producing faster derivatives than source-transformation tools. Supports C, C++, Rust, Julia, Fortran, CUDA, and more.
- **Can it differentiate DTW?** Yes, in principle. Enzyme handles loops, branches, and memory operations at the IR level. The DTW recurrence (nested loops with min operations) is differentiable in the Soft-DTW sense (replacing min with softmin). Enzyme has been specifically shown to differentiate **GPU kernels** (SC '21 paper) and **parallel programs** including OpenMP and RAJA (SC '22 paper).
- **Soft-DTW applicability:** For Soft-DTW (where `min` is replaced by `softmin(a,b,c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))`), Enzyme can automatically compute gradients of the forward pass. This avoids manually implementing the backward pass.
- **Integration complexity:** Requires Clang/LLVM as the compiler and the Enzyme plugin. Not compatible with MSVC. Would need a separate build path for Enzyme-enabled builds. CMake integration exists but is non-trivial.
- **Performance:** Excellent -- AD after optimization means the derivatives benefit from compiler optimizations applied to the primal code.
- **Recommendation for DTWC++:** **Best-in-class for automatic Soft-DTW gradients**, but the LLVM-only requirement limits portability. Ideal for a Python binding path where the C++ code is compiled with Clang. Not suitable as a mandatory dependency.

### 2.2 autodiff (autodiff.github.io)

- **Repository:** https://github.com/autodiff/autodiff
- **License:** MIT
- **Stars:** ~1,924
- **How it works:** C++17 operator-overloading AD library. Provides `autodiff::dual` (forward mode) and `autodiff::var` (reverse mode) types. Uses template metaprogramming for efficiency.
- **Soft-DTW applicability:** Replace `double` with `autodiff::dual` or `autodiff::var` in the Soft-DTW computation. The softmin recurrence would work because all operations (exp, log, addition) are overloaded. However, the DTW cost matrix would need to be templated or use `autodiff::var` types throughout.
- **Performance:** Good for forward mode (dual numbers have ~2-3x overhead). Reverse mode requires building an expression graph, which has memory overhead proportional to the number of operations (O(N*M) for DTW).
- **Integration:** Very easy -- header-only, CMake/CPM compatible. Depends on Eigen for some features.
- **Recommendation for DTWC++:** **Best practical choice for Soft-DTW gradients.** MIT license, header-only, easy to integrate. Template the DTW function on scalar type, then instantiate with `autodiff::var` for gradient computation. The main challenge is the memory overhead of reverse-mode AD for large DTW matrices.

### 2.3 CppAD (COIN-OR)

- **Repository:** https://github.com/coin-or/CppAD
- **License:** EPL 2.0 / GPL 2.0 (dual license)
- **Stars:** ~572
- **How it works:** Operator-overloading AD using `CppAD::AD<double>` type. Records a "tape" of operations for reverse-mode differentiation.
- **Soft-DTW applicability:** Similar to autodiff -- replace `double` with `CppAD::AD<double>`. The tape-based approach records all operations in the DTW recurrence. Works but the tape can be very large for DTW (O(N*M) operations).
- **Performance:** Mature and well-optimized tape. Supports sparsity detection and exploitation. Conditional expressions (`CppAD::CondExpLt`) handle the `min` in DTW.
- **Integration:** CMake compatible but heavier than autodiff. Not header-only.
- **Recommendation for DTWC++:** **Viable but heavier than autodiff.** The EPL/GPL dual license is more restrictive. CppAD is battle-tested in optimization (used by CasADi, Ipopt ecosystem) but autodiff is simpler for DTWC++'s needs.

### 2.4 Stan Math

- **Repository:** https://github.com/stan-dev/math
- **License:** BSD 3-Clause
- **Stars:** ~814
- **How it works:** Reverse-mode AD with forward-mode support. Designed for probabilistic programming (Stan language). Uses `stan::math::var` type. Depends on oneTBB, Boost, Eigen, Sundials.
- **Soft-DTW applicability:** Would work for the recurrence but Stan Math is designed for statistical model gradients, not general-purpose AD. The dependency chain (TBB + Boost + Eigen + Sundials) is extremely heavy.
- **Recommendation for DTWC++:** **Not recommended.** Far too many dependencies for what DTWC++ needs. Better suited for probabilistic programming applications.

### 2.5 JAX-style AD in C++

There is no direct C++ equivalent of JAX's tracing-based AD. The closest approaches are:
- **Enzyme** (compiler-level, like XLA's AD)
- **autodiff** (operator overloading, like PyTorch's autograd)
- **NVIDIA's stdexec + cutlass** ecosystem doesn't include AD

For a JAX-like experience in C++, the combination of **Enzyme** (for compiler-level AD) + **Kokkos** (for portable execution) is the closest analog, but this is a research-level stack.

### Summary Table: Autodiff Libraries

| Library | License | Stars | Header-only | Reverse Mode | GPU AD | Integration | Soft-DTW Fit |
|---------|---------|-------|-------------|-------------|--------|-------------|-------------|
| **Enzyme** | Apache 2.0+LLVM | 1,568 | N/A (plugin) | Yes | **Yes** | Hard (LLVM only) | Excellent |
| **autodiff** | **MIT** | **1,924** | **Yes** | **Yes** | No | **Easy** | **Good** |
| CppAD | EPL 2.0/GPL | 572 | No | Yes | No | Medium | Good |
| Stan Math | BSD 3-Clause | 814 | No | Yes | No | Hard (deps) | Poor fit |

### Recommendations for DTWC++

1. **Primary AD: autodiff library** -- MIT, header-only, easy integration. Template DTW on scalar type, use `autodiff::var` for gradients.
2. **Advanced/GPU AD: Enzyme** -- for high-performance GPU Soft-DTW gradients in a Clang-only build path.
3. **Practical approach:** Implement Soft-DTW forward pass manually (it's well-documented in Cuturi & Blondel 2017), then either:
   - Use autodiff for automatic backward pass, or
   - Implement the backward pass manually (it's a well-known algorithm, ~50 lines of code)

---

## 3. mdspan Standalone Implementations

### 3.1 kokkos/mdspan (Reference Implementation)

- **Repository:** https://github.com/kokkos/mdspan
- **License:** Apache 2.0 with LLVM Exceptions (same as Kokkos)
- **Stars:** ~494
- **Status:** Production-quality reference implementation targeting C++23 standard
- **C++17/20 compatibility:** **Yes!** Provides C++17 and C++14 backports. The C++17 backport works well with minor caveats (no deduction guides, implicit conversions that should be explicit). C++14 backport has slower compile times.
- **Header-only:** Yes
- **CUDA compatible:** Yes -- has macros to enable `__device__` marking for all functions.
- **CMake integration:** Trivially integrable via CPM/FetchContent. Standard CMake project.
- **Features:** Full `mdspan`, `extents`, `layout_left`, `layout_right`, `layout_stride`, `mdarray`. Also has `submdspan` (P2630).
- **Compiler support:** Tested with clang-15, gcc-11, CUDA 11.x. Warning-free with `-Wall -Wextra -pedantic` in C++23/20 modes.

### 3.2 std::mdspan in Compilers (Native)

- **GCC (libstdc++):** `std::mdspan` available since GCC 14 (C++23 mode). Experimental in GCC 13.
- **MSVC:** `std::mdspan` available since VS 2022 17.8 (C++23 mode).
- **Clang (libc++):** Partial support as of Clang 18. Full support expected in Clang 19+.

### 3.3 Other Polyfills

No other significant standalone mdspan implementations exist. The kokkos/mdspan repo *is* the reference implementation that compiler vendors are using to implement their `std::mdspan`.

### Recommendation for DTWC++

**Use kokkos/mdspan as a CPM dependency.** It provides a production-quality, header-only mdspan implementation that works with C++17 (which DTWC++ targets). When compilers catch up, it can be swapped for `std::mdspan` with minimal code changes. Key use case in DTWC++:
- `mdspan<double, dextents<size_t, 2>>` for the DTW cost matrix (avoids owning the memory, works with any allocator/buffer strategy)
- `mdspan<const double, dextents<size_t, 1>>` for time series views (replaces raw pointer + length)
- Layout flexibility (`layout_left` for Armadillo/MATLAB compatibility, `layout_right` for C/NumPy)

---

## 4. GPU DTW State of the Art

### 4.1 Key Implementations

#### cuDTW++ (Euro-Par 2020)
- **Repository:** https://github.com/asbschmidt/cuDTW (33 stars)
- **Paper:** "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs" by Bertil Schmidt and Christian Hundt, Euro-Par 2020
- **Technique:** Anti-diagonal parallelism with shared memory tiling. The key insight: cells along the same anti-diagonal of the DTW cost matrix are independent (each depends only on cells from the previous two anti-diagonals). A CUDA thread block processes one anti-diagonal at a time, with threads computing independent cells in parallel.
- **Performance:** Achieves near-memory-bandwidth-limited performance for long sequences.
- **License:** Not clearly stated (check repo)

#### pytorch-softdtw-cuda (Maghoumi)
- **Repository:** https://github.com/Maghoumi/pytorch-softdtw-cuda (733 stars -- most popular!)
- **Technique:** Diagonal-based Bellman recursion in CUDA via Numba. Both forward and backward passes on GPU. Uses the anti-diagonal approach where each CUDA thread processes one element of an anti-diagonal.
- **Limitation:** Sequence length limited to 1024 (one thread per element, limited by CUDA block size).
- **License:** Not explicitly stated

#### sdtw-cuda-torch (BGU-CS-VIL, 2025-2026)
- **Repository:** https://github.com/BGU-CS-VIL/sdtw-cuda-torch (18 stars, very recent)
- **Technique:** **Tiled anti-diagonal execution** -- overcomes the 1024 sequence length limit by tiling. Both fused and unfused modes for memory/speed tradeoff.
- **Key benchmarks (B=32, N=512, D=64):**
  - vs. Maghoumi: **67x faster** (unfused), **98% less memory** (fused)
  - Peak memory: 161 MB (fused) vs 8,256 MB (Maghoumi)
- **Features:** Log-space numerical stability for backward pass, unbounded sequence lengths, fused distance computation (no intermediate (B,N,M) tensor)
- **License:** Not explicitly stated

#### OpenDBA
- **Repository:** https://github.com/nodrogluap/OpenDBA (69 stars)
- **Technique:** Novel "stripe" mode for full DTW on sequences up to millions of elements. GPU-accelerated DTW Barycenter Averaging. Computes all-vs-all DTW distance matrix as a side effect.
- **License:** GPL v3 (incompatible with DTWC++ if it's not GPL)

### 4.2 GPU DTW Parallelism Techniques

The **anti-diagonal parallelism** approach is the dominant technique:

1. **Anti-diagonal sweep:** The DTW cost matrix C[i,j] depends on C[i-1,j], C[i,j-1], and C[i-1,j-1]. All cells on the same anti-diagonal (where i+j = constant) are independent. Process anti-diagonals 0, 1, 2, ..., N+M-2 sequentially, with cells within each anti-diagonal computed in parallel.

2. **Shared memory tiling:** For sequences longer than CUDA block size, tile the cost matrix into blocks. Within each tile, use shared memory for the three dependency rows/columns. Process tiles along anti-diagonals of the tile grid.

3. **Warp-level optimizations:** Use warp shuffle operations (`__shfl_up_sync`, `__shfl_down_sync`) to share values between adjacent threads within a warp, avoiding shared memory bank conflicts.

4. **Multiple DTW pairs per kernel launch:** For distance matrix computation, batch many DTW pair computations into a single kernel launch. Each thread block computes one DTW pair. This maximizes GPU occupancy.

5. **Register-level tiling:** Keep the current and previous anti-diagonal values in registers rather than shared memory for short sequences.

### 4.3 Production GPU DTW Libraries

There are **no production-ready, general-purpose GPU DTW libraries** as of early 2026. All existing implementations are either:
- Research code tied to specific papers (cuDTW++)
- PyTorch-specific (Maghoumi, sdtw-cuda-torch)
- Domain-specific (OpenDBA for nanopore sequencing)

This represents an **opportunity for DTWC++** to provide the first production-quality, standalone C++ GPU DTW library.

### Recommendations for DTWC++ GPU Support

1. **Anti-diagonal parallelism** is the proven approach. Implement it as a CUDA kernel.
2. **Start with the distance matrix kernel:** One thread block per DTW pair. This is simpler than single-pair GPU acceleration and provides the most value (the distance matrix is the bottleneck).
3. **Use tiled anti-diagonal execution** (as in sdtw-cuda-torch) for arbitrary sequence lengths.
4. **Shared memory strategy:** Store two anti-diagonals in shared memory. Each thread computes one cell. Synchronize between anti-diagonals with `__syncthreads()`.
5. **Consider Soft-DTW from the start:** The GPU kernel structure is nearly identical for DTW and Soft-DTW (replace `min` with `softmin`). Design the kernel to support both.
6. **Orchestration:** Use Taskflow's cudaFlow to manage kernel launches for the distance matrix (submit all pairs as a graph of kernel launches).

---

## 5. CUDA Toolkit Redistribution

### 5.1 Key Rules

From the NVIDIA CUDA EULA (https://docs.nvidia.com/cuda/eula/index.html):

**Redistributable components (at no charge):**
- **CUDA Runtime:** `cudart.dll` (Windows), `libcudart.so` (Linux), `libcudart.dylib` (macOS) -- both shared and static versions
- **CUDA Runtime Compilation (NVRTC):** `nvrtc.dll` / `libnvrtc.so` -- for runtime kernel compilation
- **All CUDA math libraries:** cuBLAS, cuFFT, cuSPARSE, cuSOLVER, cuRAND
- **CUDA headers:** `cuda_runtime_api.h`, `cuda.h`, `vector_types.h`, etc.

**NOT redistributable:**
- The NVIDIA GPU **driver** itself -- users must install this separately
- The CUDA **compiler** (nvcc) -- users need the CUDA Toolkit to compile CUDA code
- CUDA profiling tools (nsight, nvprof)

### 5.2 Runtime vs. Driver API

- **CUDA Runtime API** (`cudart`): Higher-level, manages contexts automatically. **Redistributable.** This is what most applications use. Users need only the NVIDIA driver installed.
- **CUDA Driver API** (`cuda.h`, `libcuda.so`): Lower-level, provided by the **NVIDIA driver** (not the toolkit). NOT redistributable separately -- it comes with the driver installation. But applications can *link against* it if the driver is installed.

### 5.3 Do End Users Need to Install CUDA?

**For running pre-compiled binaries:**
- Users need the **NVIDIA GPU driver** (which includes `libcuda.so` / `nvcuda.dll`)
- Users do **NOT** need the full CUDA Toolkit if the application ships with the redistributable CUDA runtime (`libcudart.so` / `cudart.dll`)
- The CUDA runtime can be **statically linked** (`cudart_static.lib` / `libcudart_static.a`), eliminating even the runtime DLL dependency. Users then only need the GPU driver.

**For building from source:**
- Users need the **CUDA Toolkit** (nvcc compiler, headers, libraries)

### 5.4 Practical Strategy for DTWC++

1. **Build time:** Require CUDA Toolkit for building with GPU support (standard CMake `find_package(CUDA)` or `enable_language(CUDA)`)
2. **Runtime:** Statically link `cudart_static` so users only need the NVIDIA driver
3. **Python wheels:** Ship pre-compiled CUDA kernels in the wheel. The CUDA runtime can be bundled (redistributable). Users need only the NVIDIA driver. This is exactly what PyTorch and CuPy do.
4. **Version compatibility:** CUDA has forward compatibility -- a newer driver supports older CUDA runtime versions. Target CUDA 11.8 or 12.x for broad compatibility.
5. **Detection:** At runtime, check for CUDA availability with `cudaGetDeviceCount()`. Fall back to CPU if no GPU is available. This allows a single binary to work on both GPU and non-GPU systems.

### 5.5 License Implications

- The CUDA EULA is **not open source** -- it's a proprietary license with redistribution rights
- DTWC++ can link against CUDA and redistribute the runtime without licensing issues
- The EULA prohibits reverse engineering CUDA output "for the purpose of translating such output artifacts to target a non-NVIDIA platform" -- this means you cannot use CUDA output to create AMD/Intel GPU code
- No fee for redistribution

---

## Citations and References

### Papers
1. Schmidt, B. and Hundt, C. "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs." Euro-Par 2020.
2. Cuturi, M. and Blondel, M. "Soft-DTW: a Differentiable Loss Function for Time-Series." ICML 2017.
3. Moses, W. and Churavy, V. "Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients." NeurIPS 2020. (Enzyme)
4. Moses, W. et al. "Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme." SC '21.
5. Moses, W. et al. "Scalable Automatic Differentiation of Multiple Parallel Paradigms through Compiler Augmentation." SC '22.
6. Maghoumi, M. "Deep Recurrent Networks for Gesture Recognition and Synthesis." PhD thesis, UCF, 2020. (pytorch-softdtw-cuda)

### Repositories
- NVIDIA CCCL (Thrust/CUB/libcudacxx): https://github.com/NVIDIA/cccl
- oneTBB: https://github.com/uxlfoundation/oneTBB
- Kokkos: https://github.com/kokkos/kokkos
- RAJA: https://github.com/LLNL/RAJA
- HPX: https://github.com/STEllAR-GROUP/hpx
- Taskflow: https://github.com/taskflow/taskflow
- NVIDIA stdexec (P2300): https://github.com/NVIDIA/stdexec
- Enzyme: https://github.com/EnzymeAD/Enzyme
- autodiff: https://github.com/autodiff/autodiff
- CppAD: https://github.com/coin-or/CppAD
- Stan Math: https://github.com/stan-dev/math
- kokkos/mdspan: https://github.com/kokkos/mdspan
- cuDTW++: https://github.com/asbschmidt/cuDTW
- pytorch-softdtw-cuda: https://github.com/Maghoumi/pytorch-softdtw-cuda
- sdtw-cuda-torch: https://github.com/BGU-CS-VIL/sdtw-cuda-torch
- OpenDBA: https://github.com/nodrogluap/OpenDBA
- CUDA EULA: https://docs.nvidia.com/cuda/eula/index.html
