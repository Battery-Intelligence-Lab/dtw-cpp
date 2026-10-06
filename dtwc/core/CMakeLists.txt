target_sources(dtwc_core
  PRIVATE
    distance_matrix.cpp
    dtw_dispatch.cpp
    dtw_lanes.cpp
  PUBLIC FILE_SET HEADERS FILES
    clustering_result.hpp
    distance_matrix.hpp
    distance_sampling_weights.hpp
    dtw_cost.hpp
    dtw_dispatch.hpp
    dtw_kernel.hpp
    dtw_options.hpp
    gpu_dtw_common.hpp
    lower_bound_impl.hpp
    matrix_io.hpp
    medoid_assignment_policy.hpp
    msm.hpp
    portable_random.hpp
    public_distance.hpp
    sha256.hpp
    storage.hpp
    twe.hpp
    z_normalize.hpp
  )

# lld-link's LTO backend lacks the SLP vectoriser dtw_kernel_lanes needs, so clang-cl compiles
# dtw_lanes.cpp to native code. dtwc_core is defined in the parent directory, hence TARGET_DIRECTORY.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  set_source_files_properties(dtw_lanes.cpp TARGET_DIRECTORY dtwc_core PROPERTIES COMPILE_OPTIONS -fno-lto)
endif()
