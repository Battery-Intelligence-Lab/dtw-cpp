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
    scratch_matrix.hpp
    time_series.hpp
    z_normalize.hpp
  )

# lld-link's LTO backend (LLVM 21) runs the loop vectoriser but not the SLP
# vectoriser, which is what turns dtw_kernel_lanes' unrolled lane loop into
# packed vminpd/vaddpd. Clang targeting the MSVC ABI links through lld-link, so
# there the fill's lane binding is compiled to native code, not ThinLTO bitcode.
# dtwc++ is defined in the parent directory, hence TARGET_DIRECTORY.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  set_source_files_properties(dtw_lanes.cpp TARGET_DIRECTORY dtwc++ PROPERTIES COMPILE_OPTIONS -fno-lto)
endif()
