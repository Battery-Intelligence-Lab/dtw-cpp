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
  target_compile_options(dtwc_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
  )
  # Pass host CPU SIMD flag to nvcc's host compiler so host-side staging
  # (e.g. pad_series FP64->FP32 conversion) auto-vectorizes.
  if(DEFINED _arch_flag AND NOT "${_arch_flag}" STREQUAL "")
    target_compile_options(dtwc_core PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release,RelWithDebInfo>:-Xcompiler=${_arch_flag}>>
    )
  elseif(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    target_compile_options(dtwc_core PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release,RelWithDebInfo>:-Xcompiler=/arch:AVX2>>
    )
  endif()
  message(STATUS "CUDA enabled for GPU DTW acceleration")
endif()
