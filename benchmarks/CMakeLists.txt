cmake_minimum_required(VERSION 3.21)

<<<<<<< HEAD
add_executable(bench_dtw_baseline bench_dtw_baseline.cpp)
target_link_libraries(bench_dtw_baseline
  PRIVATE
    dtwc++
    benchmark::benchmark
    benchmark::benchmark_main
)
=======
# Google Benchmark dependency
if(NOT TARGET benchmark::benchmark)
  CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    VERSION 1.8.3
    OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
    "BENCHMARK_ENABLE_INSTALL OFF"
  )
endif()

add_executable(bench_dtw_baseline bench_dtw_baseline.cpp)
target_link_libraries(bench_dtw_baseline
  PRIVATE
  dtwc++
  benchmark::benchmark
  armadillo
)
target_compile_features(bench_dtw_baseline PRIVATE cxx_std_17)
>>>>>>> e59a35c (Fix documentation errors and add roofline-aware benchmarks)
