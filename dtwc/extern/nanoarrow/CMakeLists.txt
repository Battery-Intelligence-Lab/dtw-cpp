# nanoarrow.h renames every symbol into the DtwcNanoarrow namespace, so this copy
# cannot clash with an Arrow library loaded into the same process.
target_sources(dtwc_core
  PRIVATE
    nanoarrow.c
  )
