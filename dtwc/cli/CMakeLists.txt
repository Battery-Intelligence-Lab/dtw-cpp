target_sources(dtwc++
  PRIVATE
    config.cpp
    run.cpp
  PUBLIC FILE_SET HEADERS FILES
    config.hpp
    config_file.hpp
    run.hpp
  )
