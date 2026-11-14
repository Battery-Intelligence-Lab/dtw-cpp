# nanoarrow.h renames every symbol into the DtwcNanoarrow namespace, so this copy
# cannot clash with an Arrow library loaded into the same process; nanoarrow_ipc.h,
# which includes it, renames the IPC extension's alike. The IPC extension and flatcc
# are third-party C the Arrow IPC reader runs, compiled in dtwc_third_party
# (dtwc/CMakeLists.txt), with its include directories.
target_sources(dtwc_core PRIVATE nanoarrow.c)
target_sources(dtwc_third_party PRIVATE nanoarrow_ipc.c flatcc.c)
# flatcc's paligned_alloc.h calls _aligned_malloc/_aligned_free on MSVC/MinGW without including
# <malloc.h>; force-include it via CMake so vendored nanoarrow sources stay unmodified upstream.
if(MSVC)
  set_source_files_properties(nanoarrow_ipc.c flatcc.c PROPERTIES COMPILE_OPTIONS "/FImalloc.h")
elseif(MINGW)
  set_source_files_properties(nanoarrow_ipc.c flatcc.c PROPERTIES COMPILE_OPTIONS "-include;malloc.h")
endif()
