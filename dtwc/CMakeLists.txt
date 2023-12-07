
add_library(dtwc++ OBJECT)
add_subdirectory(solver)

target_sources(dtwc++
  PRIVATE
  Problem.cpp
  mip.cpp
  mip_Gurobi.cpp
  parallelisation.hpp
)

target_include_directories(dtwc++ PUBLIC .)

target_link_libraries(dtwc++
  PRIVATE
  project_warnings
  project_options
  eigen
  range-v3
  fmt
  highs::highs
)

if(DTWC_ENABLE_GUROBI)
  message(STATUS "Gurobi is enabled!")
  if(Gurobi_FOUND)
    message(STATUS "Gurobi is found!")
    add_definitions(-DDTWC_ENABLE_GUROBI) 
    target_link_libraries(dtwc++ PRIVATE Gurobi::GurobiCXX)
  endif()
endif()


target_include_directories(dtwc++ PRIVATE ${range-v3_INCLUDE_DIRS} )
target_compile_definitions(dtwc++ PRIVATE ROOT_FOLDER="${PROJECT_SOURCE_DIR}")

add_executable(dtwc_main
  main.cpp
)

target_link_libraries(dtwc_main
  PRIVATE
  dtwc++
  project_warnings
  project_options
)