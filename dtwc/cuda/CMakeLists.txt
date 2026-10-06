# The architectures are set in the root CMakeLists.txt, before enable_language(CUDA).
# The headers are listed in every build, as every folder's are; the source needs the switch.
target_sources(dtwc_core
  PUBLIC FILE_SET HEADERS FILES
    cuda_dtw.cuh
    cuda_memory.cuh
    kernel_selection.hpp
    launch_prep.hpp
  )
if(DTWC_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  target_sources(dtwc_core
    PRIVATE
      cuda_dtw.cu
    )
  target_link_libraries(dtwc_core PUBLIC CUDA::cudart)
  target_compile_definitions(dtwc_core PUBLIC DTWC_HAS_CUDA)
  message(STATUS "CUDA enabled for GPU DTW acceleration")
endif()
