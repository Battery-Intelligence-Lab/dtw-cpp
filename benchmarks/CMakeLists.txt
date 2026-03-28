cmake_minimum_required(VERSION 3.21)

# Google Benchmark microbenchmarks
add_executable(bench_dtw_baseline bench_dtw_baseline.cpp)
target_link_libraries(bench_dtw_baseline
  PRIVATE
    dtwc++
    benchmark::benchmark
    benchmark::benchmark_main
)

# Legacy UCR benchmark
add_executable(benchmark_UCR UCR_dtwc.cpp)
target_link_libraries(benchmark_UCR PRIVATE dtwc++)
