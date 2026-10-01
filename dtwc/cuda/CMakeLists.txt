# The architectures are set in the root CMakeLists.txt, before enable_language(CUDA).
if(DTWC_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  target_sources(dtwc++
    PRIVATE
      cuda_dtw.cu
    )
  target_link_libraries(dtwc++ PUBLIC CUDA::cudart)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_CUDA)
  message(STATUS "CUDA enabled for GPU DTW acceleration")
endif()
