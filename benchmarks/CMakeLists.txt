cmake_minimum_required(VERSION 3.26)

# Google Benchmark microbenchmarks
add_executable(bench_dtw_baseline bench_dtw_baseline.cpp)
target_link_libraries(bench_dtw_baseline
  PRIVATE
    dtwc++
    benchmark::benchmark
    benchmark::benchmark_main
)

# Link Highway for SIMD benchmarks if available
if(DTWC_ENABLE_SIMD AND (TARGET hwy::hwy OR TARGET hwy))
  set(_hwy_target "$<IF:$<TARGET_EXISTS:hwy::hwy>,hwy::hwy,hwy>")
  target_link_libraries(bench_dtw_baseline PRIVATE "${_hwy_target}")
endif()

# Mmap access pattern benchmarks
add_executable(bench_mmap_access bench_mmap_access.cpp)
target_link_libraries(bench_mmap_access
  PRIVATE
    dtwc++
    benchmark::benchmark
    benchmark::benchmark_main
)

# CUDA benchmarks (Google Benchmark, requires CUDA).
# Defines its own main() to inject GPU device info into the JSON context.
add_executable(bench_cuda_dtw bench_cuda_dtw.cpp)
target_link_libraries(bench_cuda_dtw
  PRIVATE
    dtwc++
    benchmark::benchmark
)

# Metal benchmarks (Google Benchmark, requires macOS Metal).
# Target is always added; kernel code is gated on DTWC_HAS_METAL inside the .cpp.
# Defines its own main() to inject GPU device info into the JSON context.
add_executable(bench_metal_dtw bench_metal_dtw.cpp)
target_link_libraries(bench_metal_dtw
  PRIVATE
    dtwc++
    benchmark::benchmark
)

# MPI benchmarks (standalone, requires MPI)
add_executable(bench_mpi_dtw bench_mpi_dtw.cpp)
target_link_libraries(bench_mpi_dtw PRIVATE dtwc++)

# float32 vs float64 DTW speed benchmark
add_executable(bench_f32_vs_f64 bench_f32_vs_f64.cpp)
target_link_libraries(bench_f32_vs_f64
  PRIVATE
    dtwc++
    benchmark::benchmark
    benchmark::benchmark_main
)

# Legacy UCR benchmark
add_executable(benchmark_UCR UCR_dtwc.cpp)
target_link_libraries(benchmark_UCR PRIVATE dtwc++)
