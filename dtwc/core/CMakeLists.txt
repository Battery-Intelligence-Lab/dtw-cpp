# @file CMakeLists.txt
# @brief Source and header registration for core DTW kernels, SIMD lanes, and distance matrix storage.
# @author Volkan Kumtepeli
# @date 01 Oct 2026

target_sources(dtwc_core
  PRIVATE
    distance_matrix.cpp
    dtw_dispatch.cpp
    dtw_lanes.cpp
    machine.cpp
    thread_scratch.cpp
  PUBLIC FILE_SET HEADERS FILES
    clustering_result.hpp
    distance_matrix.hpp
    distance_sampling_weights.hpp
    dtw_cost.hpp
    dtw_dispatch.hpp
    dtw_kernel.hpp
    dtw_options.hpp
    gpu_dtw_common.hpp
    gpu_policy.hpp
    lower_bound_impl.hpp
    machine.hpp
    matrix_io.hpp
    medoid_assignment_policy.hpp
    msm.hpp
    portable_random.hpp
    public_distance.hpp
    series.hpp
    sha256.hpp
    storage.hpp
    thread_scratch.hpp
    twe.hpp
    z_normalize.hpp
  )

# lld-link's LTO backend lacks the SLP vectoriser dtw_kernel_lanes needs, so clang-cl compiles
# dtw_lanes.cpp to native code. Under GCC on x86-64, raising max-completely-peel-times to 8 lets
# the W-lane inner loop peel completely into packed SIMD registers instead of scalar stack traffic.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  set_source_files_properties(dtw_lanes.cpp TARGET_DIRECTORY dtwc_core PROPERTIES COMPILE_OPTIONS -fno-lto)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set_source_files_properties(dtw_lanes.cpp TARGET_DIRECTORY dtwc_core PROPERTIES
    COMPILE_OPTIONS "--param=max-completely-peel-times=8")
endif()
