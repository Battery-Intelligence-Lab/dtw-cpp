# This file is adapted from https://gitlab.inf.unibe.ch/CGG-public/cmake-library/-/blob/master/finders/FindGurobi.cmake
# Once done this will define
#  Gurobi_FOUND - System has Gurobi
#  Targets:
#       Gurobi::GurobiC   - only the C interface
#       Gurobi::GurobiCXX - C and C++ interface

# ── Locate GUROBI_HOME ──────────────────────────────────────────────────────
# Priority: user cache → GUROBI_HOME env-var → well-known install paths.
# On Windows, Gurobi installs to C:/gurobiXYZ/win64. Scan for the newest.
set(_gurobi_search_paths "$ENV{GUROBI_HOME}" "/opt/gurobi/linux64/")
if(WIN32)
  # Scan C:/gurobi*/win64 for installations, sorted descending to prefer newest
  file(GLOB _gurobi_win_candidates "C:/gurobi*/win64")
  list(SORT _gurobi_win_candidates ORDER DESCENDING)
  list(APPEND _gurobi_search_paths ${_gurobi_win_candidates})
  unset(_gurobi_win_candidates)
endif()

find_path(GUROBI_HOME
          NAMES include/gurobi_c++.h
          PATHS ${_gurobi_search_paths}
          NO_DEFAULT_PATH # avoid finding /usr
          )
unset(_gurobi_search_paths)

find_path(GUROBI_INCLUDE_DIR
    NAMES gurobi_c++.h
    HINTS "${GUROBI_HOME}/include"
    )
mark_as_advanced(GUROBI_INCLUDE_DIR)

set(GUROBI_BIN_DIR "${GUROBI_HOME}/bin")
set(GUROBI_LIB_DIR "${GUROBI_HOME}/lib")

# ── Detect library version ──────────────────────────────────────────────────
# Scan for versioned libraries to determine the version suffix (e.g., "130").
if (WIN32)
    # On Windows, the import lib in lib/ is authoritative.
    file(GLOB GUROBI_LIBRARY_LIST
        RELATIVE ${GUROBI_LIB_DIR}
        ${GUROBI_LIB_DIR}/gurobi[0-9]*.lib
        )
    # Filter out C++ wrapper libs (gurobi_c++*.lib)
    list(FILTER GUROBI_LIBRARY_LIST EXCLUDE REGEX "gurobi_c\\+\\+")
else()
    file(GLOB GUROBI_LIBRARY_LIST
        RELATIVE ${GUROBI_LIB_DIR}
        ${GUROBI_LIB_DIR}/libgurobi*.so
        )
endif()

# Extract version numbers, ignoring _light variants
string(REGEX MATCHALL
    "gurobi([0-9]+)\\..*"
    GUROBI_LIBRARY_LIST
    "${GUROBI_LIBRARY_LIST}"
    )

string(REGEX REPLACE
    ".*gurobi([0-9]+)\\..*"
    "\\1"
    GUROBI_LIBRARY_VERSIONS
    "${GUROBI_LIBRARY_LIST}")
list(LENGTH GUROBI_LIBRARY_VERSIONS GUROBI_NUMVER)

if (GUROBI_NUMVER EQUAL 0)
    message(STATUS "Found no Gurobi library version, GUROBI_HOME = ${GUROBI_HOME}.")
elseif (GUROBI_NUMVER EQUAL 1)
    list(GET GUROBI_LIBRARY_VERSIONS 0 GUROBI_LIBRARY_VERSION)
else()
    # More than one — pick the highest version
    list(SORT GUROBI_LIBRARY_VERSIONS COMPARE NATURAL ORDER DESCENDING)
    list(GET GUROBI_LIBRARY_VERSIONS 0 GUROBI_LIBRARY_VERSION)
    message(STATUS "Found multiple Gurobi versions (${GUROBI_LIBRARY_VERSIONS}), using ${GUROBI_LIBRARY_VERSION}")
endif()

# ── Find the library ────────────────────────────────────────────────────────
if (WIN32)
    # On Windows with non-MSVC compilers (Clang, MinGW), find_library may not
    # search bin/ for DLLs. Use the import lib in lib/ as the primary library
    # and record the DLL location separately for IMPORTED_LOCATION.
    find_library(GUROBI_IMPLIB
        NAMES "gurobi${GUROBI_LIBRARY_VERSION}"
        PATHS ${GUROBI_LIB_DIR}
        NO_DEFAULT_PATH
    )
    mark_as_advanced(GUROBI_IMPLIB)

    # Find the DLL in bin/
    find_file(GUROBI_DLL
        NAMES "gurobi${GUROBI_LIBRARY_VERSION}.dll"
        PATHS ${GUROBI_BIN_DIR}
        NO_DEFAULT_PATH
    )
    mark_as_advanced(GUROBI_DLL)

    # Accept either the import lib or a direct DLL find as "found"
    if(GUROBI_IMPLIB)
      set(GUROBI_LIBRARY "${GUROBI_IMPLIB}" CACHE FILEPATH "Gurobi C library" FORCE)
    endif()
else ()
    find_library(GUROBI_LIBRARY
        NAMES "gurobi${GUROBI_LIBRARY_VERSION}"
        PATHS ${GUROBI_LIB_DIR}
        NO_DEFAULT_PATH
    )
endif()
mark_as_advanced(GUROBI_LIBRARY)

# ── Create imported targets ─────────────────────────────────────────────────
if(GUROBI_LIBRARY AND NOT TARGET Gurobi::GurobiC)
    add_library(Gurobi::GurobiC SHARED IMPORTED)
    target_include_directories(Gurobi::GurobiC INTERFACE ${GUROBI_INCLUDE_DIR})
    if(WIN32 AND GUROBI_DLL)
      set_target_properties(Gurobi::GurobiC PROPERTIES
        IMPORTED_LOCATION "${GUROBI_DLL}"
        IMPORTED_IMPLIB "${GUROBI_LIBRARY}")
    else()
      set_target_properties(Gurobi::GurobiC PROPERTIES IMPORTED_LOCATION ${GUROBI_LIBRARY})
      if(GUROBI_IMPLIB)
        set_target_properties(Gurobi::GurobiC PROPERTIES IMPORTED_IMPLIB ${GUROBI_IMPLIB})
      endif()
    endif()
endif()

# Gurobi ships with some compiled versions of its C++ library for specific
# compilers, however it also comes with the source code. We will compile
# the source code outselves -- this is much safer, as it guarantees the same
# compiler version and flags.
# (Note: doing this is motivated by actual sometimes-subtle ABI compatibility bugs)

find_path(GUROBI_SRC_DIR NAMES "Model.h" PATHS "${GUROBI_HOME}/src/cpp/")
mark_as_advanced(GUROBI_SRC_DIR)

file(GLOB GUROBI_CXX_SRC CONFIGURE_DEPENDS ${GUROBI_SRC_DIR}/*.cpp)
if(TARGET Gurobi::GurobiC AND GUROBI_CXX_SRC AND NOT TARGET Gurobi::GurobiCXX)
    add_library(GurobiCXX STATIC EXCLUDE_FROM_ALL ${GUROBI_CXX_SRC})
    add_library(Gurobi::GurobiCXX ALIAS GurobiCXX)

    if(MSVC)
        target_compile_definitions(GurobiCXX PRIVATE "WIN64")
    endif()

    target_include_directories(GurobiCXX PUBLIC ${GUROBI_INCLUDE_DIR})
    target_link_libraries(GurobiCXX PUBLIC Gurobi::GurobiC)
    # We need to be able to link this into a shared library:
    set_target_properties(GurobiCXX PROPERTIES POSITION_INDEPENDENT_CODE ON)

endif()

# legacy support:
set(GUROBI_INCLUDE_DIRS "${GUROBI_INCLUDE_DIR}")
set(GUROBI_LIBRARIES Gurobi::GurobiC Gurobi::GurobiCXX)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gurobi  DEFAULT_MSG GUROBI_LIBRARY GUROBI_INCLUDE_DIR GUROBI_SRC_DIR)
