# nanoarrow.h renames every symbol into the DtwcNanoarrow namespace, so this copy
# cannot clash with an Arrow library loaded into the same process; nanoarrow_ipc.h,
# which includes it, renames the IPC extension's alike.
target_sources(dtwc_core
  PRIVATE
    nanoarrow.c
    nanoarrow_ipc.c
    flatcc.c
  )
# flatcc's headers include each other as "flatcc/...", from this folder.
set_source_files_properties(nanoarrow_ipc.c flatcc.c TARGET_DIRECTORY dtwc_core
  PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")
