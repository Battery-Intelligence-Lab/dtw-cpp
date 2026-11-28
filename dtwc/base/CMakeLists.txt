# @file CMakeLists.txt
# @brief Source and header registration for base error, parallelisation, and settings utilities.
# @author Volkan Kumtepeli
# @date 01 Oct 2026

target_sources(dtwc_core
  PRIVATE
    env.cpp
  PUBLIC FILE_SET HEADERS FILES
    env.hpp
    error.hpp
    missing_utils.hpp
    names.hpp
    parallelisation.hpp
    settings.hpp
    timing.hpp
  )
