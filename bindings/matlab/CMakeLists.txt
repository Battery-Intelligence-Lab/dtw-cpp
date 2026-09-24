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

# macOS: one OpenMP runtime per process. MATLAB ships libomp beside libmex and has
# it loaded at start-up (its BLAS links it); a second copy aborts MATLAB at the
# MEX's first parallel region ("OMP: Error #15"), and with KMP_DUPLICATE_LIB_OK
# it segfaults instead, since dyld coalesces the copies' weak symbols (R2026a,
# 2026-09-24). So the MEX's libomp load command is repointed at @rpath/libomp.dylib,
# the install name of MATLAB's copy, which dyld finds already loaded. The old name
# is the install name the linker copied (line 2 of `otool -D`), which need not be
# the path CMake found: install_name_tool ignores a name that does not match.
# DTWC_MEX_MATLAB_LIBOMP: see dtwc_mex.cpp.
get_filename_component(_dtwc_matlab_bin "${Matlab_MEX_LIBRARY}" DIRECTORY)
if(APPLE AND EXISTS "${_dtwc_matlab_bin}/libomp.dylib")
  foreach(_dtwc_omp_name IN LISTS OpenMP_CXX_LIB_NAMES)
    if(OpenMP_${_dtwc_omp_name}_LIBRARY MATCHES "libomp\\.dylib$")
      execute_process(COMMAND otool -D "${OpenMP_${_dtwc_omp_name}_LIBRARY}"
                      OUTPUT_VARIABLE _dtwc_omp_id OUTPUT_STRIP_TRAILING_WHITESPACE
                      COMMAND_ERROR_IS_FATAL ANY)
      string(REGEX REPLACE "^[^\n]*\n" "" _dtwc_omp_id "${_dtwc_omp_id}")
      target_compile_definitions(dtwc_mex PRIVATE DTWC_MEX_MATLAB_LIBOMP)
      add_custom_command(TARGET dtwc_mex POST_BUILD
        COMMAND install_name_tool -change "${_dtwc_omp_id}" "@rpath/libomp.dylib"
                "$<TARGET_FILE:dtwc_mex>"
        VERBATIM)
      break()
    endif()
  endforeach()
endif()

# FindMatlab (CMake 4.2) exports the MEX entry points only under `if(MSVC)`.
# A GNU-frontend Clang build on Windows therefore links a .mexw64 with no
# exported mexFunction, and MATLAB rejects it at load time with
# "Invalid MEX-file ...: Gateway function is missing". Export them explicitly;
# `LINKER:` lets CMake emit the right driver syntax for either frontend.
if(WIN32 AND NOT MSVC)
    target_link_options(
        dtwc_mex
        PRIVATE
            "LINKER:/EXPORT:mexFunction"
            "LINKER:/EXPORT:mexfilerequiredapiversion")
endif()

# Install MEX binary alongside the +dtwc package
install(TARGETS dtwc_mex
    DESTINATION matlab
)

# Install the MATLAB package directory
install(DIRECTORY +dtwc
    DESTINATION matlab
)
