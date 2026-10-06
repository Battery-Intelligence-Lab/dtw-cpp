# The architectures are set in the root CMakeLists.txt, before enable_language(CUDA).
# The headers are listed in every build, as every folder's are; the source needs the switch.
target_sources(dtwc++
  PUBLIC FILE_SET HEADERS FILES
    cuda_dtw.cuh
    cuda_memory.cuh
    kernel_selection.hpp
    launch_prep.hpp
  )
if(DTWC_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  target_sources(dtwc++
    PRIVATE
      cuda_dtw.cu
    )
  target_link_libraries(dtwc++ PUBLIC CUDA::cudart)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_CUDA)
  message(STATUS "CUDA enabled for GPU DTW acceleration")
endif()
