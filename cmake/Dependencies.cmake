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
    GIT_TAG ca42334d561b83e499b11cf55fe05d97a0767fb9 # v0.0.7
  )

  if(DTWC_BUILD_TESTING AND NOT TARGET Catch2::Catch2WithMain) # Catch2 library:
    CPMAddPackage(
      NAME Catch2
      URL "https://github.com/catchorg/Catch2/archive/refs/tags/v3.13.0.tar.gz"
      # SHA256 pinned. Computed from the tarball CPM
      # downloaded, cached at build/_deps/catch2-subbuild/.../v3.13.0.tar.gz,
      # on 2026-07-07. Immutable release tag -> GitHub serves identical bytes.
      URL_HASH SHA256=650795f6501af514f806e78c554729847b98db6935e69076f36bb03ed2e985ef
      OPTIONS
      "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF" "CATCH_BUILD_TESTING OFF"
    )
  endif()

  # HiGHS library:
  if(NOT TARGET highs::highs AND DTWC_ENABLE_HIGHS)# HiGHS library:
  # HiGHS's cuPDLP GPU backend stays off: nothing here uses PDLP, and ON would
  # force HiGHS shared on Windows and pull cudart/cublas/cusparse.
  set(CUPDLP_GPU OFF CACHE BOOL "Enable HiGHS cuPDLP GPU support" FORCE)
  # HiGHS defaults to a shared libhighs on Linux and macOS. The CLI archive ships
  # it under lib/; a Python extension or a MEX cannot: wheel.exclude drops lib/,
  # the extension records no rpath (delocate and auditwheel stop at the missing
  # library), and the MEX ships as one file. Link it into them instead. A plain
  # variable of this function: HiGHS's option() yields to it (CMP0077) and it goes
  # no further.
  if(DTWC_BUILD_PYTHON OR DTWC_BUILD_MATLAB)
    set(BUILD_SHARED_LIBS OFF)
  endif()
  # A fetch that fails stops the configure inside CPM, which names the package and
  # not the option; this line is the one that does.
  message(STATUS "  HiGHS:    fetching (DTWC_ENABLE_HIGHS=ON; if it cannot be fetched, configure with -DDTWC_ENABLE_HIGHS=OFF)")
  CPMAddPackage(
    NAME highs
    URL "https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.15.1.tar.gz"
    # SHA256 pinned. Computed 2026-07-08 from the GitHub release
    # tarball for the immutable tag v1.15.1 (`curl -sL … | sha256sum`).
    URL_HASH SHA256=a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16
    SYSTEM
    EXCLUDE_FROM_ALL
    OPTIONS
      "CI OFF"
      "ZLIB OFF"
      "BUILD_CXX_EXE OFF"
      "BUILD_EXAMPLES OFF"
      "BUILD_TESTING OFF"
      "FAST_BUILD ON"
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
  # CPM treats an existing source directory as a cache hit, and a download that failed
  # earlier leaves one empty; HiGHS then adds no target and the MIP solver would vanish.
  if(DTWC_ENABLE_HIGHS AND NOT TARGET highs::highs)
    message(FATAL_ERROR "DTWC_ENABLE_HIGHS=ON but HiGHS made no highs::highs target "
      "(source directory ${highs_SOURCE_DIR}; delete it if a download failed earlier).\n"
      "  To build without HiGHS, pass -DDTWC_ENABLE_HIGHS=OFF.")
  endif()

  if (NOT TARGET CLI11::CLI11)
  CPMAddPackage(
    NAME CLI11
    URL "https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.2.tar.gz"
    # SHA256 pinned. Computed 2026-07-07 from cached tarball
    # build/_deps/cli11-subbuild/.../v2.6.2.tar.gz. Immutable release tag.
    URL_HASH SHA256=c6ea6b2e5608b3ea8617999bd5f47420c71b2ebdb8dc4767c1034d1da5785711
    DOWNLOAD_ONLY YES
  )

   add_library(CLI11::CLI11 INTERFACE IMPORTED)
  set_target_properties(CLI11::CLI11 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CLI11_SOURCE_DIR}/include")
  endif()

  # fkYAML — single-header YAML 1.2 parser (MIT). OPTIONAL: it only feeds
  # dtwc/cli/config_file.hpp, which turns a YAML `--config` file into CLI11
  # ConfigItems. With DTWC_ENABLE_YAML=OFF the CLI still builds and rejects YAML
  # config files with a typed error instead of parsing them.
  if(DTWC_ENABLE_YAML AND NOT TARGET fkYAML::fkYAML)
    message(STATUS "  YAML:     fetching fkYAML (DTWC_ENABLE_YAML=ON; if it cannot be fetched, configure with -DDTWC_ENABLE_YAML=OFF)")
    CPMAddPackage(
      NAME fkYAML
      URL "https://github.com/fktn-k/fkYAML/archive/refs/tags/v0.4.4.tar.gz"
      # SHA256 pinned. Computed 2026-09-02 from two independent downloads of the
      # immutable release tag v0.4.4 (`curl -sL … | sha256sum`, identical bytes).
      URL_HASH SHA256=75fa1ce37480ac2ef47b820bfdba04894d4f19ac122ad59d892601553aa45c4e
      DOWNLOAD_ONLY YES
    )
    add_library(fkYAML::fkYAML INTERFACE IMPORTED)
    set_target_properties(fkYAML::fkYAML PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${fkYAML_SOURCE_DIR}/single_include")
  endif()
  if(TARGET fkYAML::fkYAML)
    message(STATUS "  YAML:     YES (fkYAML — dtwc_cl --config accepts TOML or YAML)")
  else()
    message(STATUS "  YAML:     OFF (DTWC_ENABLE_YAML=OFF) — --config accepts TOML only.")
  endif()

  # nanobind — Python bindings (BSD-3, by Wenzel Jakob)
  # find_package(Python) and nanobind discovery are handled in python/CMakeLists.txt
  # to ensure scikit-build-core has configured paths first.

  # Eigen was removed. It was the project's only copyleft dependency
  # (MPL-2.0) and the only MPL obligation in the Python wheel, and it was carried
  # for exactly two uses: ScratchMatrix's base class and a to_full_matrix return
  # type that every caller immediately copied into a std::vector. Both are now
  # plain standard library. The CMake rationale here had also gone stale — it
  # claimed "zero-copy Map" and dense distance-matrix internals, and there was no
  # Eigen::Map anywhere and the matrix was already std::vector<double>.

  # PMU counters. Every route by which this could fail to deliver
  # counters is a FATAL_ERROR, because Google Benchmark's own runtime guard cannot
  # be relied on: the BM_CHECK at v1.9.5 benchmark_runner.cc:323 is inverted (it
  # aborts when the counters *were* set up and stays silent when they were not),
  # and it sits inside an `aggregation_report_mode() != ARM_Unspecified` branch
  # that an ordinary benchmark never enters. A build without libpfm therefore
  # accepts --benchmark_perf_counters, writes a normal-looking JSON with no counter
  # fields, and exits 0. scripts/run_bench.sh catches that at run time; the checks
  # below catch the configurations that cannot work at all.
  if(DTWC_BENCHMARK_PMU)
    if(NOT DTWC_BUILD_BENCHMARK)
      message(FATAL_ERROR
        "DTWC_BENCHMARK_PMU=ON does nothing with DTWC_BUILD_BENCHMARK=OFF: the "
        "counters come from Google Benchmark, and no benchmark target is being "
        "built. Add -DDTWC_BUILD_BENCHMARK=ON, or drop -DDTWC_BENCHMARK_PMU.")
    endif()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
      message(FATAL_ERROR
        "DTWC_BENCHMARK_PMU=ON needs Linux on bare metal; this is "
        "${CMAKE_SYSTEM_NAME}. libpfm4 reads the PMU through perf_event_open, "
        "which macOS does not have (xctrace/Instruments is the nearest equivalent) "
        "and which virtualised hosts — GitHub runners, and usually WSL2 — do not "
        "expose. Configure without -DDTWC_BENCHMARK_PMU and report wall-clock as "
        "advisory, or run on a bare-metal Linux node.")
    endif()
    if(TARGET benchmark::benchmark)
      message(FATAL_ERROR
        "DTWC_BENCHMARK_PMU=ON cannot be honoured: benchmark::benchmark was "
        "already defined by an enclosing project, so BENCHMARK_ENABLE_LIBPFM is "
        "whatever that project chose. Enable libpfm4 there instead.")
    endif()
  endif()

  if(DTWC_BUILD_BENCHMARK)
    if(NOT TARGET benchmark::benchmark)
      # Set outside the CPMAddPackage() call on purpose. scripts/check_pins.py
      # refuses a CPMAddPackage whose arguments expand a ${} that is not a literal,
      # and it is right to: a pin nobody can read statically is not a pin. This is
      # the same mechanism CPM's own OPTIONS use, and google/benchmark's
      # option(BENCHMARK_ENABLE_LIBPFM ...) leaves an existing cache entry alone.
      # ON ⇒ upstream runs find_package(PFM REQUIRED), so missing libpfm4 headers
      # or library stop the configure instead of quietly dropping the counters.
      set(BENCHMARK_ENABLE_LIBPFM ${DTWC_BENCHMARK_PMU} CACHE INTERNAL
          "Google Benchmark libpfm4 support; driven by DTWC_BENCHMARK_PMU")
      CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.9.5
        GIT_TAG 192ef10025eb2c4cdd392bc502f0c852196baa48 # v1.9.5
        OPTIONS
          "BENCHMARK_ENABLE_TESTING OFF"
          "BENCHMARK_ENABLE_GTEST_TESTS OFF"
          "BENCHMARK_ENABLE_WERROR OFF"
      )
    endif()
  endif()

  # llfio — memory-mapped I/O for large distance matrices (OPTIONAL). With
  # DTWC_ENABLE_LLFIO=OFF there is no llfio_hl target, dtwc/CMakeLists.txt then
  # leaves DTWC_HAS_MMAP undefined, and a request for mmap storage raises a typed
  # error. Only dtwc/core/distance_matrix.cpp includes llfio.
  #
  # Header-only, from pinned archives: no llfio CMake, no quickcpplib bootstrap,
  # no nested build. llfio_hl carries what llfio's own header-only target carried
  # through quickcpplib::hl, outcome::hl and, on Windows, ntkernel-error-category::hl;
  # the preprocessed <llfio/v2.0/llfio.hpp> is line for line the one the former
  # superbuild produced (clang 21, Windows, 2026-09-28).
  #
  # Pins. llfio b17613fb is 5 commits past its newest tag, 20260506, and is kept
  # for 284ba8d9 (PR #178, a libc++ char8_t codecvt fix that AppleClang needs);
  # move to a tag once one includes it. quickcpplib publishes no tags, so a commit
  # is the only pin it offers. outcome 32f20369 is the develop == master tip the
  # superbuild used to fetch unpinned. wg14_signals, ntkernel-error-category,
  # span-lite and byte-lite are the submodule commits that llfio b17613fb and
  # quickcpplib 3c1d8cb5 record; GitHub archives leave submodules out.
  option(DTWC_ENABLE_LLFIO "Enable llfio memory-mapped distance matrices" ON)
  if(DTWC_ENABLE_LLFIO AND NOT TARGET llfio_hl)
    message(STATUS "  llfio:    fetching (DTWC_ENABLE_LLFIO=ON; if it cannot be fetched, configure with -DDTWC_ENABLE_LLFIO=OFF)")
    CPMAddPackage(NAME llfio DOWNLOAD_ONLY YES
      URL "https://github.com/ned14/llfio/archive/b17613fb2149a93b0cc7022c8e649dbf5a015b90.tar.gz"
      URL_HASH SHA256=f1dda54633647791101ffb21dc2311750aa5d39a0a017bced4383ffb66362913)
    CPMAddPackage(NAME wg14_signals DOWNLOAD_ONLY YES
      URL "https://github.com/ned14/wg14_signals/archive/36d3cdb66993078c8fecba93e2a5f2c549572d64.tar.gz"
      URL_HASH SHA256=0fc195c3074815486e3d200254fb2940814903895d1d77898eaf498aec409d29)
    CPMAddPackage(NAME quickcpplib DOWNLOAD_ONLY YES
      URL "https://github.com/ned14/quickcpplib/archive/3c1d8cb5e94722447e4f17e87b5a9e3a0c66fb39.tar.gz"
      URL_HASH SHA256=4d2c775c1fcfe984d2b0b8c313d68dfda433e272a97ffd9706c74f86652b4deb)
    CPMAddPackage(NAME span_lite DOWNLOAD_ONLY YES
      URL "https://github.com/martinmoene/span-lite/archive/dbb484f6c2060b41afa55653dec99b228013a813.tar.gz"
      URL_HASH SHA256=ebfde55f9d141ef4ea5ca99a140829ff3c1b17220613fe7177e661e542c77a39)
    CPMAddPackage(NAME byte_lite DOWNLOAD_ONLY YES
      URL "https://github.com/martinmoene/byte-lite/archive/5bf0d80352197a4fb3526ad678a23a4c0c40d094.tar.gz"
      URL_HASH SHA256=b8384d7c184f8e3ec82107651fc56b0e160e2067db59aa03967b3db1d39733de)
    CPMAddPackage(NAME outcome DOWNLOAD_ONLY YES
      URL "https://github.com/ned14/outcome/archive/32f203695dea0073722699890d8f3faca58ff9bb.tar.gz"
      URL_HASH SHA256=c20f52413e1b2a7dbe6fa809f07b923abf43e994d9f01c3179e1cd300c9dd3c7)

    # llfio includes wg14_signals by a path relative to its own headers
    # (detail/impl/signal_guard.hpp: "../../../wg14_signals/include/..."), so the
    # two trees are joined here, in the build tree; the downloads stay untouched.
    set(_dtwc_llfio_include "${CMAKE_BINARY_DIR}/_deps/llfio-hl/include")
    file(COPY "${llfio_SOURCE_DIR}/include/llfio" DESTINATION "${_dtwc_llfio_include}")
    file(COPY "${wg14_signals_SOURCE_DIR}/include"
         DESTINATION "${_dtwc_llfio_include}/llfio/wg14_signals")

    find_package(Threads REQUIRED)
    add_library(llfio_hl INTERFACE IMPORTED)
    set_target_properties(llfio_hl PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES
        "${_dtwc_llfio_include};${quickcpplib_SOURCE_DIR}/include;${outcome_SOURCE_DIR}/include;${span_lite_SOURCE_DIR}/include;${byte_lite_SOURCE_DIR}/include"
      # quickcpplib's own build writes the first define into a generated
      # detail/config.hpp (the archive ships a placeholder). Its C++14 ABI takes
      # span and byte from span-lite and byte-lite, found here on the include path.
      INTERFACE_COMPILE_DEFINITIONS
        "QUICKCPPLIB_REQUIRE_CXX_STANDARD=201402L;QUICKCPPLIB_USE_SYSTEM_SPAN_LITE=1;QUICKCPPLIB_USE_SYSTEM_BYTE_LITE=1;$<$<CONFIG:Debug>:QUICKCPPLIB_ENABLE_VALGRIND=1>"
      # glibc < 2.34 keeps dladdr (ringbuffer_log) in libdl. quickcpplib's own target
      # links dl too, and rt, which only its signal_guard needs; llfio does not use that.
      INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")
    if(WIN32)
      CPMAddPackage(NAME ntkernel_error_category DOWNLOAD_ONLY YES
        URL "https://github.com/ned14/ntkernel-error-category/archive/c20f97bccacc3162d515c0828ff35fe88f4bf9f2.tar.gz"
        URL_HASH SHA256=3c6376496407774b54b59201bf1faedfa0f92a35439cf0d8a87c26cdfd21094c)
      # llfio's Windows API floor, and the NT error category in header-only form:
      # one category object per binary. Upstream warns that comparing error codes
      # across binaries then fails; dtwc never compares them, it turns each llfio
      # error into a message in the binary that raised it.
      set_property(TARGET llfio_hl APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${ntkernel_error_category_SOURCE_DIR}/include")
      set_property(TARGET llfio_hl APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
        _WIN32_WINNT=0x601 NTKERNEL_ERROR_CATEGORY_INLINE NTKERNEL_ERROR_CATEGORY_STATIC)
    endif()
  endif()
  if(TARGET llfio_hl)
    message(STATUS "  llfio:    YES (header-only; memory-mapped distance matrix enabled)")
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
        # find_package() runs inside this function.  Export the package result
        # alongside the existing capability flag so the parent directory can
        # link Parquet::parquet_shared and publish DTWC_HAS_PARQUET.
        set(Parquet_FOUND TRUE PARENT_SCOPE)
        set(DTWC_HAS_PARQUET_LIB TRUE PARENT_SCOPE)
      endif()
    else()
      # 2. Build minimal Arrow+Parquet from source via CPM
      message(STATUS "  Arrow:    not found — building from source via CPM (first build takes ~5 min)")

      # Workaround: Arrow's ExternalProject passes CMAKE_CXX_FLAGS_* to sub-builds
      # (snappy, zstd, thrift). On Windows+Clang, flags like "-Xclang --dependent-lib=msvcrt"
      # and "-D_DLL -D_MT" contain spaces that break CMake argument parsing in ExternalProject.
      # Strip these from ALL flag variables before Arrow configure, restore after.
      # Flag stripping not needed — Windows+Clang never reaches the CPM build (the guard below stops it)
      # NOTE: On Windows+Clang, Arrow's ExternalProject sub-builds fail because
      # CMake's platform module sets "-Xclang --dependent-lib=msvcrt" in default
      # flags, and the space breaks ExternalProject's semicolon-separated command.
      # This is an Arrow upstream issue. Workarounds:
      #   - Use MSVC compiler instead of Clang on Windows
      #   - Use conda: conda install -c conda-forge arrow-cpp (find_package path)
      #   - Use Linux (SLURM) where this issue doesn't exist
      if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "DTWC_ENABLE_ARROW=ON: Arrow was not found, and the CPM build "
          "is not supported with Windows+Clang due to ExternalProject flag quoting issues. Use one of:\n"
          "  1. conda install -c conda-forge arrow-cpp  (then find_package works), "
          "or point -DArrow_DIR and -DParquet_DIR at an Arrow install\n"
          "  2. Build with MSVC instead of Clang\n"
          "  3. Use Linux (SLURM) where CPM build works\n"
          "  4. Pass -DDTWC_ENABLE_ARROW=OFF and convert the data with dtwc-convert (Python)")
      endif()

      CPMAddPackage(
        NAME Arrow
        VERSION 19.0.1
        URL "https://github.com/apache/arrow/archive/refs/tags/apache-arrow-19.0.1.tar.gz"
        URL_HASH SHA256=4c898504958841cc86b6f8710ecb2919f96b5e10fa8989ac10ac4fca8362d86a
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
        message(FATAL_ERROR "DTWC_ENABLE_ARROW=ON: the CPM build of Arrow made no arrow_static target.\n"
          "  Install Arrow (conda install -c conda-forge arrow-cpp), or pass -DDTWC_ENABLE_ARROW=OFF.")
      endif()
    endif()
  endif()

  # CUDA detection is done in root CMakeLists.txt (enable_language requires
  # directory scope). CUDAToolkit is already found there.

endfunction()
