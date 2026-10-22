# nanoarrow.h renames every symbol into the DtwcNanoarrow namespace, so this copy
# cannot clash with an Arrow library loaded into the same process; nanoarrow_ipc.h,
# which includes it, renames the IPC extension's alike. The IPC extension and flatcc
# are third-party C the Arrow IPC reader runs, compiled in dtwc_third_party
# (dtwc/CMakeLists.txt), with its include directories.
target_sources(dtwc_core PRIVATE nanoarrow.c)
target_sources(dtwc_third_party PRIVATE nanoarrow_ipc.c flatcc.c)
