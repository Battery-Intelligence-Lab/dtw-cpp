# MATLAB MEX bindings for DTWC++
#
# Build with: cmake .. -DDTWC_BUILD_MATLAB=ON
# Requires MATLAB to be installed and findable by CMake.

find_package(Matlab COMPONENTS MX_LIBRARY)

if(NOT Matlab_FOUND)
    message(WARNING "MATLAB not found -- skipping MATLAB MEX bindings.")
    return()
endif()

message(STATUS "MATLAB found: ${Matlab_ROOT_DIR}")
message(STATUS "Building MATLAB MEX bindings for DTWC++")

# Build the MEX gateway using the C++ MEX API (R2018a+)
matlab_add_mex(
    NAME dtwc_mex
    SRC dtwc_mex.cpp
    R2018a          # Use interleaved complex / C++ MEX API
    LINK_TO dtwc++
)

target_compile_features(dtwc_mex PRIVATE cxx_std_17)

# Install MEX binary alongside the +dtwc package
install(TARGETS dtwc_mex
    DESTINATION matlab
)

# Install the MATLAB package directory
install(DIRECTORY +dtwc
    DESTINATION matlab
)
