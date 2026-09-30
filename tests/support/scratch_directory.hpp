/**
 * @file scratch_directory.hpp
 * @brief A directory unique to one test case, for tests that write files.
 *
 * @details Test binaries run at the same time (ctest -j, several developers or
 *          agents sharing one %TEMP%). A path fixed in the source is then a race
 *          between two processes running the same test, or between two tests that
 *          chose the same name. The directory name carries the process id and a
 *          per-process serial, so no two live objects share it, in this process
 *          or another.
 */

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace dtwc::test_support {

/// An empty directory under the system temporary directory, removed with the object.
/// `stem` names the case in the directory name and must not contain a separator.
struct ScratchDirectory
{
  const std::filesystem::path path;

  explicit ScratchDirectory(std::string_view stem) : path(unique_path(stem))
  {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored); // left by a crashed run whose pid was recycled
    std::filesystem::create_directories(path);
  }

  ScratchDirectory(const ScratchDirectory &) = delete;
  ScratchDirectory &operator=(const ScratchDirectory &) = delete;

  ~ScratchDirectory()
  {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }

private:
  static std::filesystem::path unique_path(std::string_view stem)
  {
    static unsigned serial = 0; // Catch2 runs the cases of one process one at a time
#ifdef _WIN32
    const auto pid = ::_getpid();
#else
    const auto pid = ::getpid();
#endif
    return std::filesystem::temp_directory_path()
         / ("dtwc_" + std::string(stem) + "_" + std::to_string(pid) + "_" + std::to_string(serial++));
  }
};

} // namespace dtwc::test_support
