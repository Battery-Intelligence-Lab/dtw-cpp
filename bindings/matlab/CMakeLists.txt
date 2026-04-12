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

# Build the MEX gateway using the legacy C MEX API (mex.h / matrix.h)
# with R2018a interleaved complex support (enables mxGetDoubles).
# R2024b/macOS: FindMatlab.cmake unconditionally appends cppMexFunction.map
# to the exported-symbols list whenever Matlab_HAS_CPP_API is set. That map
# requires _mexCreateMexFunction / _mexDestroyMexFunction / _mexFunctionAdapter
# to be defined — but we use the legacy C mexFunction() API and don't provide
# them. Clear the flag so only c_exportsmexfileversion.map is used.
if(APPLE)
  # Clear the regular variable that FindMatlab.cmake sets at line ~1777.
  # matlab_add_mex() reads this inside its own scope.
  set(Matlab_HAS_CPP_API FALSE)
  unset(Matlab_HAS_CPP_API CACHE)
endif()

matlab_add_mex(
    NAME dtwc_mex
    SRC dtwc_mex.cpp
    R2018a          # Use interleaved complex (mxGetDoubles); keeps legacy C API
    LINK_TO dtwc++
)

target_compile_features(dtwc_mex PRIVATE cxx_std_20)

# Install MEX binary alongside the +dtwc package
install(TARGETS dtwc_mex
    DESTINATION matlab
)

# Install the MATLAB package directory
install(DIRECTORY +dtwc
    DESTINATION matlab
)
