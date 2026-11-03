target_sources(dtwc_cli
  PRIVATE
    config.cpp
    report.cpp
    run.cpp
  PUBLIC FILE_SET HEADERS FILES
    config.hpp
    config_file.hpp
    report.hpp
    run.hpp
  )
# The run report names the floating-point model and the arch level the build chose, which only CMake knows
# (cmake/StandardProjectSettings.cmake): two definitions on its one source.
set_source_files_properties(report.cpp TARGET_DIRECTORY dtwc_cli PROPERTIES COMPILE_DEFINITIONS
  "DTWC_FP_MODEL_NAME=\"${DTWC_FP_MODEL}\";DTWC_ARCH_LEVEL_NAME=\"${_dtwc_arch_level}\"")
