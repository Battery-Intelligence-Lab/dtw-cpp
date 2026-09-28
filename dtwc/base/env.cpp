/**
 * @file env.cpp
 * @brief The device grammar and the single-thread warning (see env.hpp).
 *
 * @details The DeviceError messages here are authored in
 * docs/api-contract-2.0.md §6.1 and asserted in tests/unit/test_env_device.cpp.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 07 Jul 2026
 */

#include "env.hpp"
#include "error.hpp"

#include <cctype>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#ifdef DTWC_HAS_OPENMP
#include <omp.h>
#endif

namespace dtwc {

namespace {

std::string to_lower_copy(std::string_view s)
{
  std::string out(s);
  for (auto &c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string trim(std::string_view s)
{
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string_view::npos) return {};
  const auto e = s.find_last_not_of(" \t\r\n");
  return std::string(s.substr(b, e - b + 1));
}

} // namespace

std::string to_string(Device d)
{
  validate_device(d);
  switch (d) {
  case Device::CPU: return "cpu";
  case Device::GPU: return "gpu";
  }
  throw std::logic_error("to_string: unreachable Device");
}

namespace detail {

std::pair<Device, int> parse_device(std::string_view name)
{
  const std::string raw = trim(name);
  const std::string lname = to_lower_copy(raw);

  if (lname == "cpu") return { Device::CPU, 0 };
  if (lname == "gpu" || lname == "cuda") return { Device::GPU, 0 };
  if (lname.rfind("gpu:", 0) == 0 || lname.rfind("cuda:", 0) == 0) {
    const std::string id = lname.substr(lname.find(':') + 1);
    if (!id.empty() && id.find_first_not_of("0123456789") == std::string::npos) {
      try {
        return { Device::GPU, std::stoi(id) };
      } catch (const std::out_of_range &) {
        // An absurdly large ordinal is an unknown name.
      }
    }
  }
  if (lname == "hpc" || lname.rfind("hpc:", 0) == 0)
    throw DeviceError(
      "[dtwc] device '" + raw + "' submits a whole run to a SLURM cluster, which "
      "Python's dtwcpp.device(\"hpc\") and scripts/slurm/slurm_remote.sh do. C++, "
      "MATLAB and dtwc_cl compute where they start: use cpu or gpu.");
  throw DeviceError("[dtwc] unknown device '" + raw
                    + "'. Valid devices: cpu, gpu, gpu:N (aliases cuda, cuda:N).");
}

std::string gpu_not_built_message()
{
  return "[dtwc] device='gpu' requested but this build has no GPU backend compiled in.\n"
         "Rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, on macOS, -DDTWC_ENABLE_METAL=ON.\n"
         "This build will not silently fall back to CPU.";
}

} // namespace detail

void warn_if_single_threaded()
{
  static std::once_flag warned;
  std::call_once(warned, [] {
#ifdef DTWC_SEQUENTIAL_BUILD // defined when configured with -DDTWC_ALLOW_SEQUENTIAL=ON
    std::cerr << "[DTWC++ WARNING] This build was compiled WITHOUT OpenMP (-DDTWC_ALLOW_SEQUENTIAL=ON) — DTWC++ is running SINGLE-THREADED.\n"
                 "  Distance-matrix computation will be extremely slow for large datasets.\n"
                 "  Rebuild without -DDTWC_ALLOW_SEQUENTIAL=ON (with OpenMP available) for parallel execution.\n";
#elif defined(DTWC_HAS_OPENMP)
    // A single-core host (or an unknown core count, 0) is serial by nature: no warning.
    if (omp_get_max_threads() <= 1 && std::thread::hardware_concurrency() > 1)
      std::cerr << "[DTWC++ WARNING] OpenMP is available but only 1 thread is usable — DTWC++ is running SINGLE-THREADED.\n"
                   "  Distance-matrix computation will be extremely slow for large datasets.\n"
                   "  Raise the thread count (unset OMP_NUM_THREADS, or set OMP_NUM_THREADS>1) to use all CPU cores.\n";
#endif
  });
}

} // namespace dtwc
