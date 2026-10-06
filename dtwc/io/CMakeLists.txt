target_sources(dtwc_core
  PRIVATE
    arrow_c_data.cpp
    read_data.cpp
    parse_number.cpp
  PUBLIC FILE_SET HEADERS FILES
    arrow_c_data.hpp
    read_data.hpp
    parse_number.hpp
    parquet_chunk_reader.hpp
    parquet_schema.hpp
  )
