target_sources(dtwc++
  PRIVATE
    distance_matrix.cpp
    dtw_dispatch.cpp
    dtw_lanes.cpp
  PUBLIC
    clustering_result.hpp
    distance_matrix.hpp
    dtw_options.hpp
    lower_bound_impl.hpp
    sha256.hpp
    z_normalize.hpp
  )

# lld-link's LTO backend lacks the SLP vectoriser dtw_kernel_lanes needs, so clang-cl compiles
# dtw_lanes.cpp to native code. dtwc++ is defined in the parent directory, hence TARGET_DIRECTORY.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  set_source_files_properties(dtw_lanes.cpp TARGET_DIRECTORY dtwc++ PROPERTIES COMPILE_OPTIONS -fno-lto)
endif()
