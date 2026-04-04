cmake_minimum_required(VERSION 3.21)

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

# CUDA benchmarks (Google Benchmark, requires CUDA)
add_executable(bench_cuda_dtw bench_cuda_dtw.cpp)
target_link_libraries(bench_cuda_dtw
  PRIVATE
    dtwc++
    benchmark::benchmark
    benchmark::benchmark_main
)

# MPI benchmarks (standalone, requires MPI)
add_executable(bench_mpi_dtw bench_mpi_dtw.cpp)
target_link_libraries(bench_mpi_dtw PRIVATE dtwc++)

# Legacy UCR benchmark
add_executable(benchmark_UCR UCR_dtwc.cpp)
target_link_libraries(benchmark_UCR PRIVATE dtwc++)
