target_sources(dtwc++
  PRIVATE
    env.cpp
  PUBLIC FILE_SET HEADERS FILES
    env.hpp
    error.hpp
    missing_utils.hpp
    names.hpp
    parallelisation.hpp
    random_engine.hpp
    settings.hpp
    timing.hpp
  )
