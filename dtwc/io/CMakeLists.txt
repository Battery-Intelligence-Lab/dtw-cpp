# @file CMakeLists.txt
# @brief Source and header registration for CSV, Arrow IPC, and Parquet readers/writers.
# @author Volkan Kumtepeli
# @date 01 Oct 2026

target_sources(dtwc_core
  PRIVATE
    arrow_c_data.cpp
    arrow_ipc.cpp
    read_data.cpp
    parse_number.cpp
  PUBLIC FILE_SET HEADERS FILES
    arrow_c_data.hpp
    arrow_ipc.hpp
    read_data.hpp
    parse_number.hpp
  )
# The Parquet readers, which need Arrow C++, in a build with Arrow.
if(TARGET dtwc_io)
  target_sources(dtwc_io
    PRIVATE
      read_arrow.cpp
    PUBLIC FILE_SET HEADERS FILES
      read_arrow.hpp
      parquet_chunk_reader.hpp
      parquet_schema.hpp
    )
endif()
