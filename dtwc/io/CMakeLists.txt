target_sources(dtwc_core
  PRIVATE
    arrow_c_data.cpp
    read_data.cpp
    parse_number.cpp
  PUBLIC FILE_SET HEADERS FILES
    arrow_c_data.hpp
    read_data.hpp
    parse_number.hpp
  )
# The readers that need Arrow, in a build with Arrow.
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
