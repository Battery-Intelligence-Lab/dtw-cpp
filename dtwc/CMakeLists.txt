add_subdirectory(solver)

add_library(dtwc++ STATIC)

target_sources(dtwc++
  PRIVATE
  Problem.cpp
  mip.cpp
  mip_Gurobi.cpp
  parallelisation.hpp
)

target_include_directories(dtwc++ PUBLIC .)

# if(DTWC_ENABLE_IWYU)
#   find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)
#   set_property(TARGET dtwc++ PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
# endif()

target_link_libraries(dtwc++
  PRIVATE
  dtwc_project_warnings
  dtwc_project_options
  solvers
  eigen
  range-v3
  fmt
  highs::highs
)

if(DTWC_ENABLE_GUROBI AND Gurobi_FOUND)
  message("Gurobi is found!")
  add_definitions(-DDTWC_ENABLE_GUROBI) 
  target_link_libraries(dtwc++ PRIVATE Gurobi::GurobiCXX)
endif()

target_include_directories(dtwc++ PRIVATE ${HIGHS_INCLUDE_DIRS} ${range-v3_INCLUDE_DIRS})
target_compile_definitions(dtwc++ PRIVATE ROOT_FOLDER="${PROJECT_SOURCE_DIR}")

add_executable(dtwc_main
  main.cpp
)