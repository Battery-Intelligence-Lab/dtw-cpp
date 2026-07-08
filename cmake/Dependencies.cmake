# This cmake file is to add external dependency projects.
# Adapted from https://github.com/cpp-best-practices/cmake_tecomplate/tree/main
include(cmake/CPM.cmake)

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets

function(dtwc_setup_dependencies)
  # For each dependency, see if it's
  # already been provided to us by a parent project
  CPMAddPackage(
    NAME CPMLicenses.cmake
    GITHUB_REPOSITORY cpm-cmake/CPMLicenses.cmake
    VERSION 0.0.7
  )

  if(NOT TARGET Catch2::Catch2WithMain) # Catch2 library:
    CPMAddPackage(
      NAME Catch2
      URL "https://github.com/catchorg/Catch2/archive/refs/tags/v3.13.0.tar.gz"
      # SHA256 pinned (Task 0.12 supply-chain). Computed from the tarball CPM
      # downloaded, cached at build/_deps/catch2-subbuild/.../v3.13.0.tar.gz,
      # on 2026-07-07. Immutable release tag -> GitHub serves identical bytes.
      URL_HASH SHA256=650795f6501af514f806e78c554729847b98db6935e69076f36bb03ed2e985ef
      OPTIONS
      "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF" "CATCH_BUILD_TESTING OFF"
    )
  endif()

  # HiGHS library:
  if(NOT TARGET highs::highs AND DTWC_ENABLE_HIGHS)# HiGHS library:
  CPMAddPackage(
    NAME highs
    URL "https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.15.1.tar.gz"
    # SHA256 pinned (Task 0.12). Computed 2026-07-08 from the GitHub release
    # tarball for the immutable tag v1.15.1 (`curl -sL … | sha256sum`).
    # v1.15.1 ships PDLP (first-order LP: solver="pdlp"/"hipdlp") with an
    # optional GPU/cuPDLP backend (HiGHS CMake option CUPDLP_GPU, default OFF —
    # our build uses CPU PDLP unless that flag is forwarded).
    URL_HASH SHA256=a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16
    SYSTEM
    EXCLUDE_FROM_ALL
    OPTIONS
    "CI OFF" "ZLIB OFF" "BUILD_EXAMPLES OFF" "BUILD_TESTING OFF" "FAST_BUILD ON"
    )
    # Historically HiGHS <=1.14.0 had a debug assertion (ub_consistent) that
    # fired on valid warm-start MIP solves (primal-dual integral bookkeeping
    # tolerance 1e-12 too tight after a presolve reset). Retained defensively:
    # forcing NDEBUG on the HiGHS target keeps its internal asserts off in Debug
    # builds of DTWC++ regardless of whether 1.15.1 tightened the tolerance.
    if(TARGET highs)
      target_compile_definitions(highs PRIVATE NDEBUG)
    endif()
  endif()

  if (NOT TARGET CLI11::CLI11)
  CPMAddPackage(
    NAME CLI11
    URL "https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.2.tar.gz"
    # SHA256 pinned (Task 0.12). Computed 2026-07-07 from cached tarball
    # build/_deps/cli11-subbuild/.../v2.6.2.tar.gz. Immutable release tag.
    URL_HASH SHA256=c6ea6b2e5608b3ea8617999bd5f47420c71b2ebdb8dc4767c1034d1da5785711
    DOWNLOAD_ONLY YES
  )

   add_library(CLI11::CLI11 INTERFACE IMPORTED)
  set_target_properties(CLI11::CLI11 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CLI11_SOURCE_DIR}/include")
  endif()

  # nanobind — Python bindings (BSD-3, by Wenzel Jakob)
  # find_package(Python) and nanobind discovery are handled in python/CMakeLists.txt
  # to ensure scikit-build-core has configured paths first.

  # RapidCSV - header-only CSV parser (BSD 3-Clause license)
  if (NOT TARGET rapidcsv::rapidcsv)
    CPMAddPackage(
      NAME rapidcsv
      GITHUB_REPOSITORY d99kris/rapidcsv
      VERSION 8.92
      DOWNLOAD_ONLY YES
    )

    add_library(rapidcsv::rapidcsv INTERFACE IMPORTED)
    set_target_properties(rapidcsv::rapidcsv PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${rapidcsv_SOURCE_DIR}/src")
  endif()

  # Eigen3 — header-only linear algebra (MPL2 / Apache-2.0 / BSD-3)
  # Used for scratch matrices (aligned SIMD-ready allocation, zero-copy Map),
  # replacing custom ScratchMatrix and DenseDistanceMatrix internals.
  if(NOT TARGET Eigen3::Eigen)
    CPMAddPackage(
      NAME Eigen
      URL "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.bz2"
      # SHA256 pinned (Task 0.12). Computed 2026-07-07 from cached tarball
      # build/_deps/eigen-subbuild/.../eigen-5.0.1.tar.bz2. Immutable release tag.
      URL_HASH SHA256=e4de6b08f33fd8b8985d2f204381408c660bffa6170ac65b68ae1bd3cd575c0a
      DOWNLOAD_ONLY YES
    )
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Eigen_SOURCE_DIR}"
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Eigen_SOURCE_DIR}")
  endif()

  if(DTWC_BUILD_BENCHMARK)
    if(NOT TARGET benchmark::benchmark)
      CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.9.5
        OPTIONS
          "BENCHMARK_ENABLE_TESTING OFF"
          "BENCHMARK_ENABLE_GTEST_TESTS OFF"
          "BENCHMARK_ENABLE_WERROR OFF"
      )
    endif()
  endif()

  # yaml-cpp — YAML configuration file support (MIT license, optional)
  if(DTWC_ENABLE_YAML AND NOT TARGET yaml-cpp)
    CPMAddPackage(
      NAME yaml-cpp
      URL "https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.9.0.tar.gz"
      # OPEN (Task 0.12): URL_HASH SHA256 not yet pinned. This optional dep is
      # OFF by default (DTWC_ENABLE_YAML), so it was not in the local CPM cache
      # and no network was available to fetch the tarball and compute the hash.
      # Maintainer TODO: download once, `sha256sum 0.9.0.tar.gz`, add
      # `URL_HASH SHA256=<hash>` here (immutable release tag -> stable bytes).
      SYSTEM
      EXCLUDE_FROM_ALL
      OPTIONS "YAML_CPP_BUILD_TESTS OFF" "YAML_CPP_BUILD_TOOLS OFF"
    )
    # Ensure namespaced alias exists (CPM subdirectory may not create it)
    if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
      add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
    endif()
    if(NOT TARGET yaml-cpp)
      message(WARNING "yaml-cpp not found -- YAML config support disabled")
      set(DTWC_ENABLE_YAML OFF PARENT_SCOPE)
    endif()
  endif()

  # llfio — memory-mapped I/O for large distance matrices (OPTIONAL).
  # Optional per the project "optional deps only" rule (Task 0.12): the core
  # must configure without llfio. When disabled/absent, DTWC_HAS_MMAP is never
  # defined (see dtwc/CMakeLists.txt, which already guards on TARGET llfio_hl)
  # and mmap-backed stores fall back to the in-memory path.
  #   RESOLVED (Task R3): dtwc/mip/CMakeLists.txt now guards its `llfio_hl` link
  #   (and a mirrored DTWC_HAS_MMAP define) on `if(TARGET llfio_hl)`, so a full
  #   llfio-less configure+generate succeeds — verified by a live configure-only
  #   run with -DDTWC_ENABLE_LLFIO=OFF (exit 0). With the default
  #   (DTWC_ENABLE_LLFIO=ON) behaviour is unchanged.
  option(DTWC_ENABLE_LLFIO "Enable llfio memory-mapped distance matrices" ON)
  if(DTWC_ENABLE_LLFIO AND NOT TARGET llfio_hl)
    CPMAddPackage(
      NAME llfio
      GITHUB_REPOSITORY ned14/llfio
      # PINNED to a specific commit (Task 0.12): previously tracked the moving
      # `develop` branch tip — a supply-chain risk (upstream force-push / hijack
      # changes what we build). SHA below is that branch HEAD read from checkout
      # build/_deps/llfio-src on 2026-07-07 (commit b17613fb, authored
      # 2026-06-01). OPEN: needs maintainer blessing of this exact SHA — offline
      # here, so no tagged release could be selected. `git describe` reported
      # 20260506-5-gb17613fb (5 commits past tag 20260506).
      GIT_TAG b17613fb2149a93b0cc7022c8e649dbf5a015b90
      DOWNLOAD_ONLY YES
    )
    if(llfio_ADDED)
      # ---------------------------------------------------------------------
      # quickcpplib ninja-propagation patch
      #
      # llfio's `find_quickcpplib_library()` (in QuickCppLibUtils.cmake) calls
      # `download_build_install()`, which runs
      #     execute_process(COMMAND "${CMAKE_COMMAND}" .)
      # with no `-G` and no `-DCMAKE_MAKE_PROGRAM`. The grandchild CMake then
      # re-detects a generator from scratch. In sandboxed wheel builds
      # (scikit-build-core's `pip-build-env`, cibuildwheel) the ninja binary
      # lives under an ephemeral path (e.g. `/.../uv/builds-v0/.tmpXXX/bin/`)
      # that isn't on the default PATH and — worse — gets rewritten on every
      # reinstall, making any cached `CMakeCache.txt` in the sub-build point
      # at a dead path. `-DCMAKE_MAKE_PROGRAM` on the sub-CMake command line
      # is the only thing that reliably overrides that stale cache.
      #
      # Fix: pre-clone quickcpplib into the location llfio's bootstrap uses
      # (`${CMAKE_BINARY_DIR}/quickcpplib/repo`) and patch
      # `download_build_install()` to forward `-G` and `-DCMAKE_MAKE_PROGRAM`.
      # llfio's bootstrap will then see the pre-existing repo and skip its
      # own git clone, and subsequently `include(QuickCppLibUtils)` picks up
      # the patched version.
      # ---------------------------------------------------------------------
      set(_dtwc_qcl_root "${CMAKE_BINARY_DIR}/quickcpplib")
      set(_dtwc_qcl_repo "${_dtwc_qcl_root}/repo")
      if(NOT EXISTS "${_dtwc_qcl_repo}/cmakelib/QuickCppLibUtils.cmake")
        find_package(Git REQUIRED)
        file(MAKE_DIRECTORY "${_dtwc_qcl_root}")
        # PINNED (Task 0.12): the previous `git clone --depth 1` checked out
        # whatever the default branch HEAD was at configure time — a moving
        # target and supply-chain risk. Pin to a reviewed commit. A full (non
        # shallow) clone is used because an arbitrary historical SHA is not
        # reachable from a depth-1 tip; we then detach onto the SHA and sync
        # submodules (also full — pinned gitlink commits may predate any shallow
        # tip) to reproduce that exact tree.
        #   SHA read from the local checkout build/quickcpplib/repo on 2026-07-07
        #   (commit 3c1d8cb5, authored 2026-03-10). OPEN: maintainer to bless.
        set(_dtwc_qcl_sha "3c1d8cb5e94722447e4f17e87b5a9e3a0c66fb39")
        message(STATUS
          "Pre-cloning quickcpplib into ${_dtwc_qcl_repo} @ ${_dtwc_qcl_sha} ...")
        execute_process(
          COMMAND "${GIT_EXECUTABLE}" clone --no-checkout --jobs 8
            "https://github.com/ned14/quickcpplib.git" repo
          WORKING_DIRECTORY "${_dtwc_qcl_root}"
          RESULT_VARIABLE _dtwc_clone_rc
        )
        if(_dtwc_clone_rc EQUAL 0)
          execute_process(
            COMMAND "${GIT_EXECUTABLE}" checkout --detach "${_dtwc_qcl_sha}"
            WORKING_DIRECTORY "${_dtwc_qcl_repo}"
            RESULT_VARIABLE _dtwc_clone_rc
          )
        endif()
        if(_dtwc_clone_rc EQUAL 0)
          execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
              --jobs 8
            WORKING_DIRECTORY "${_dtwc_qcl_repo}"
            RESULT_VARIABLE _dtwc_clone_rc
          )
        endif()
        if(NOT _dtwc_clone_rc EQUAL 0
            OR NOT EXISTS "${_dtwc_qcl_repo}/cmakelib/QuickCppLibUtils.cmake")
          message(FATAL_ERROR
            "Failed to pre-clone quickcpplib at pinned SHA ${_dtwc_qcl_sha} "
            "(rc=${_dtwc_clone_rc}). If your build environment is offline, clone "
            "manually: git clone --recursive "
            "https://github.com/ned14/quickcpplib.git ${_dtwc_qcl_repo} && "
            "git -C ${_dtwc_qcl_repo} checkout ${_dtwc_qcl_sha}")
        endif()
      endif()

      # Two patches — each checks its own pattern is still present so we can
      # recover from partially-patched state without touching an
      # already-modified line twice.
      #
      #   A) `download_build_install()` spawns a CHILD CMake with `cmake .`
      #      (no -G, no -DCMAKE_MAKE_PROGRAM).
      #   B) `find_quickcpplib_library()` builds a `cmakeargs` string that
      #      template-substitutes into ExternalProject_Add's CMAKE_ARGS,
      #      driving the GRANDCHILD CMake that configures outcome/etc.
      #      Upstream includes -G here but not -DCMAKE_MAKE_PROGRAM.
      set(_dtwc_qcl_utils "${_dtwc_qcl_repo}/cmakelib/QuickCppLibUtils.cmake")
      file(READ "${_dtwc_qcl_utils}" _dtwc_qcl_orig)
      set(_dtwc_qcl_current "${_dtwc_qcl_orig}")

      set(_dtwc_qcl_from_A "COMMAND \"\${CMAKE_COMMAND}\" .\n    WORKING_DIRECTORY \"\${DBI_DESTINATION}\"")
      set(_dtwc_qcl_to_A   "# DTWC_NINJA_PROPAGATION_PATCH (child)\n    COMMAND \"\${CMAKE_COMMAND}\" . -G \"\${CMAKE_GENERATOR}\" \"-DCMAKE_MAKE_PROGRAM=\${CMAKE_MAKE_PROGRAM}\"\n    WORKING_DIRECTORY \"\${DBI_DESTINATION}\"")
      if(NOT _dtwc_qcl_current MATCHES "DTWC_NINJA_PROPAGATION_PATCH .child.")
        string(REPLACE "${_dtwc_qcl_from_A}" "${_dtwc_qcl_to_A}"
          _dtwc_qcl_current "${_dtwc_qcl_current}")
      endif()

      set(_dtwc_qcl_from_B "set(cmakeargs \"-DCMAKE_BUILD_TYPE=\${config} -G \\\"\${CMAKE_GENERATOR}\\\" -DBUILD_TESTING=OFF \\\"-DQUICKCPPLIB_ROOT_BINARY_DIR=\${QUICKCPPLIB_ROOT_BINARY_DIR}\\\"\")")
      set(_dtwc_qcl_to_B   "# DTWC_NINJA_PROPAGATION_PATCH (grandchild)\n        set(cmakeargs \"-DCMAKE_BUILD_TYPE=\${config} -G \\\"\${CMAKE_GENERATOR}\\\" -DBUILD_TESTING=OFF \\\"-DQUICKCPPLIB_ROOT_BINARY_DIR=\${QUICKCPPLIB_ROOT_BINARY_DIR}\\\" \\\"-DCMAKE_MAKE_PROGRAM=\${CMAKE_MAKE_PROGRAM}\\\"\")")
      if(NOT _dtwc_qcl_current MATCHES "DTWC_NINJA_PROPAGATION_PATCH .grandchild.")
        string(REPLACE "${_dtwc_qcl_from_B}" "${_dtwc_qcl_to_B}"
          _dtwc_qcl_current "${_dtwc_qcl_current}")
      endif()

      if(NOT _dtwc_qcl_current STREQUAL _dtwc_qcl_orig)
        file(WRITE "${_dtwc_qcl_utils}" "${_dtwc_qcl_current}")
        message(STATUS "Patched QuickCppLibUtils.cmake "
          "(forward -G + -DCMAKE_MAKE_PROGRAM to child + grandchild CMake)")
      elseif(NOT _dtwc_qcl_current MATCHES "DTWC_NINJA_PROPAGATION_PATCH")
        message(WARNING
          "Could not apply QuickCppLibUtils ninja-propagation patches — the "
          "upstream pattern may have changed. Python wheel builds in "
          "sandboxed environments may fail at llfio configure. File: "
          "${_dtwc_qcl_utils}")
      endif()
      unset(_dtwc_qcl_orig)
      unset(_dtwc_qcl_current)
      unset(_dtwc_qcl_from_A)
      unset(_dtwc_qcl_to_A)
      unset(_dtwc_qcl_from_B)
      unset(_dtwc_qcl_to_B)
      unset(_dtwc_qcl_root)
      unset(_dtwc_qcl_repo)
      unset(_dtwc_qcl_sha)
      unset(_dtwc_qcl_utils)
      unset(_dtwc_qcl_contents)
      unset(_dtwc_qcl_patched)

      add_subdirectory(${llfio_SOURCE_DIR} ${llfio_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
  endif()

  # OPTIONAL (Task 0.12): llfio must not abort configure when absent — that
  # violated the "optional deps only" rule. Was `message(FATAL_ERROR ...)`.
  if(TARGET llfio_hl)
    message(STATUS "  llfio:    YES (memory-mapped distance matrix enabled)")
    set(DTWC_HAS_MMAP TRUE)
  elseif(DTWC_ENABLE_LLFIO)
    message(WARNING "  llfio:    requested but NOT FOUND — memory-mapped "
      "distance matrices disabled (falls back to in-memory store).")
  else()
    message(STATUS "  llfio:    OFF (DTWC_ENABLE_LLFIO=OFF) — mmap disabled.")
  endif()

  # Apache Arrow + Parquet (optional) — zero-copy IPC and Parquet reading.
  # No Python required. No external package manager required.
  #
  # Detection order:
  #   1. find_package (conda, system install — fastest, no build)
  #   2. CPM: download and build minimal Arrow from source (~5 min first time, cached after)
  #
  if(DTWC_ENABLE_ARROW)
    # 1. Try system-installed Arrow first (conda, apt, brew, module load)
    find_package(Arrow QUIET CONFIG)
    find_package(Parquet QUIET CONFIG)

    if(Arrow_FOUND)
      message(STATUS "  Arrow:    YES (v${Arrow_VERSION}) — system install")
      set(Arrow_FOUND TRUE PARENT_SCOPE)
      if(Parquet_FOUND)
        message(STATUS "  Parquet:  YES (v${Parquet_VERSION}) — system install")
        set(DTWC_HAS_PARQUET_LIB TRUE PARENT_SCOPE)
      endif()
    else()
      # 2. Build minimal Arrow+Parquet from source via CPM
      message(STATUS "  Arrow:    not found — building from source via CPM (first build takes ~5 min)")

      # Workaround: Arrow's ExternalProject passes CMAKE_CXX_FLAGS_* to sub-builds
      # (snappy, zstd, thrift). On Windows+Clang, flags like "-Xclang --dependent-lib=msvcrt"
      # and "-D_DLL -D_MT" contain spaces that break CMake argument parsing in ExternalProject.
      # Strip these from ALL flag variables before Arrow configure, restore after.
      # Flag stripping not needed — Windows+Clang skips CPM build (see guard above)
      # NOTE: On Windows+Clang, Arrow's ExternalProject sub-builds fail because
      # CMake's platform module sets "-Xclang --dependent-lib=msvcrt" in default
      # flags, and the space breaks ExternalProject's semicolon-separated command.
      # This is an Arrow upstream issue. Workarounds:
      #   - Use MSVC compiler instead of Clang on Windows
      #   - Use conda: conda install -c conda-forge arrow-cpp (find_package path)
      #   - Use Linux (SLURM) where this issue doesn't exist
      if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(WARNING "Arrow CPM build is not supported with Windows+Clang due to "
          "ExternalProject flag quoting issues. Use one of:\n"
          "  1. conda install -c conda-forge arrow-cpp  (then find_package works)\n"
          "  2. Build with MSVC instead of Clang\n"
          "  3. Use Linux (SLURM) where CPM build works\n"
          "  4. Use dtwc-convert (Python) as a workaround")
        set(DTWC_ENABLE_ARROW OFF PARENT_SCOPE)
      else()

      CPMAddPackage(
        NAME Arrow
        VERSION 19.0.1
        URL "https://github.com/apache/arrow/archive/refs/tags/apache-arrow-19.0.1.tar.gz"
        # OPEN (Task 0.12): URL_HASH SHA256 not yet pinned. This optional dep is
        # OFF by default (DTWC_ENABLE_ARROW) and was not in the local CPM cache;
        # no network was available to fetch the ~90 MB tarball and hash it.
        # Maintainer TODO: download once, `sha256sum apache-arrow-19.0.1.tar.gz`,
        # add `URL_HASH SHA256=<hash>` here (immutable release tag).
        SOURCE_SUBDIR cpp
        SYSTEM
        EXCLUDE_FROM_ALL
        OPTIONS
          "ARROW_BUILD_STATIC ON"
          "ARROW_BUILD_SHARED OFF"
          "ARROW_PARQUET ON"
          "ARROW_IPC ON"
          "ARROW_FILESYSTEM ON"
          "ARROW_COMPUTE OFF"
          "ARROW_CSV OFF"
          "ARROW_DATASET OFF"
          "ARROW_JSON OFF"
          "ARROW_FLIGHT OFF"
          "ARROW_GANDIVA OFF"
          "ARROW_ORC OFF"
          "ARROW_PLASMA OFF"
          "ARROW_PYTHON OFF"
          "ARROW_S3 OFF"
          "ARROW_HDFS OFF"
          "ARROW_JEMALLOC OFF"
          "ARROW_MIMALLOC OFF"
          "ARROW_WITH_SNAPPY OFF"
          "ARROW_WITH_ZSTD OFF"
          "ARROW_WITH_LZ4 OFF"
          "ARROW_WITH_BROTLI OFF"
          "ARROW_WITH_BZ2 OFF"
          "ARROW_WITH_ZLIB OFF"
          "ARROW_DEPENDENCY_SOURCE BUNDLED"
          "ARROW_SIMD_LEVEL NONE"
          "ARROW_USE_XSIMD OFF"
          "ARROW_RUNTIME_SIMD_LEVEL NONE"
          "ARROW_BUILD_TESTS OFF"
          "ARROW_BUILD_BENCHMARKS OFF"
          "ARROW_BUILD_EXAMPLES OFF"
          "ARROW_BUILD_UTILITIES OFF"
          "PARQUET_BUILD_EXECUTABLES OFF"
          "PARQUET_BUILD_EXAMPLES OFF"
          "PARQUET_REQUIRE_ENCRYPTION OFF"
      )

      if(TARGET arrow_static)
        set(Arrow_FOUND TRUE PARENT_SCOPE)
        set(DTWC_ARROW_FROM_CPM TRUE PARENT_SCOPE)
        set(DTWC_HAS_PARQUET_LIB TRUE PARENT_SCOPE)
        message(STATUS "  Arrow:    built from source (static, IPC + Parquet)")
      else()
        message(WARNING "Arrow CPM build failed.\n"
          "  Install: conda install -c conda-forge arrow-cpp")
        set(DTWC_ENABLE_ARROW OFF PARENT_SCOPE)
      endif()

      endif() # Windows+Clang guard
    endif()
  endif()

  # MPI (optional) — distributed distance matrix computation
  if(DTWC_ENABLE_MPI)
    # On Windows, help CMake find MS-MPI by setting hints from well-known
    # environment variables and registry locations.
    if(WIN32 AND NOT MPI_CXX_FOUND)
      # MS-MPI SDK sets MSMPI_INC and MSMPI_LIB64 / MSMPI_LIB32.
      # If the SDK env-vars are not set, try the default install path.
      if(NOT DEFINED ENV{MSMPI_INC})
        set(_msmpi_sdk_dir "C:/Program Files (x86)/Microsoft SDKs/MPI")
        if(EXISTS "${_msmpi_sdk_dir}/Include/mpi.h")
          set(ENV{MSMPI_INC} "${_msmpi_sdk_dir}/Include")
          if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(ENV{MSMPI_LIB64} "${_msmpi_sdk_dir}/Lib/x64")
          else()
            set(ENV{MSMPI_LIB32} "${_msmpi_sdk_dir}/Lib/x86")
          endif()
          message(STATUS "MS-MPI SDK found at ${_msmpi_sdk_dir}")
        endif()
      endif()
    endif()

    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
      message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    else()
      message(WARNING "MPI requested but not found — disabling.\n"
        "  On Windows, install the MS-MPI SDK from:\n"
        "  https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi\n"
        "  (both the runtime MSMpiSetup.exe AND the SDK msmpisdk.msi are required)")
      set(DTWC_ENABLE_MPI OFF PARENT_SCOPE)
    endif()
  endif()

  # CUDA detection is done in root CMakeLists.txt (enable_language requires
  # directory scope). CUDAToolkit is already found there.

endfunction()
