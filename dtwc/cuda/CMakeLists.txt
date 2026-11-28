# @file CMakeLists.txt
# @brief CUDA kernel compilation and device runtime linkage for dtwc_core.
# @author Volkan Kumtepeli
# @date 01 Oct 2026

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
  # (e.g. pad_series FP64->FP32 conversion) auto-vectorizes. Match the flag syntax
  # to nvcc's actual host compiler (e.g. cl.exe on Windows needs /arch:..., not -march=
  # when CXX is clang-cl).
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(_cuda_msvc_host FALSE)
    if(CMAKE_CUDA_HOST_SIMULATE_ID STREQUAL "MSVC")
      set(_cuda_msvc_host TRUE)
    elseif(WIN32 AND NOT (DEFINED CMAKE_CUDA_HOST_COMPILER AND CMAKE_CUDA_HOST_COMPILER MATCHES "clang|gcc|g\\+\\+"))
      set(_cuda_msvc_host TRUE)
    endif()
    if(_cuda_msvc_host)
      if(DEFINED _arch_flag AND _arch_flag MATCHES "^/arch:")
        target_compile_options(dtwc_core PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release,RelWithDebInfo>:-Xcompiler=${_arch_flag}>>
        )
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        if(DEFINED _dtwc_arch_level AND _dtwc_arch_level STREQUAL "v4")
          set(_cuda_host_arch_flag /arch:AVX512)
        else()
          set(_cuda_host_arch_flag /arch:AVX2)
        endif()
        target_compile_options(dtwc_core PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release,RelWithDebInfo>:-Xcompiler=${_cuda_host_arch_flag}>>
        )
        unset(_cuda_host_arch_flag)
      endif()
    elseif(DEFINED _arch_flag AND _arch_flag MATCHES "^-")
      target_compile_options(dtwc_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release,RelWithDebInfo>:-Xcompiler=${_arch_flag}>>
      )
    endif()
    unset(_cuda_msvc_host)
  endif()
  message(STATUS "CUDA enabled for GPU DTW acceleration")
endif()
