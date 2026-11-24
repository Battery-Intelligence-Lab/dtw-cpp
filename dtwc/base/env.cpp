/**
 * @file env.cpp
 * @brief The device grammar and the single-thread warning (see env.hpp).
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
  switch (d) {
  case Device::CPU: return "cpu";
  case Device::GPU: return "gpu";
  case Device::CPU_GPU: return "cpu+gpu";
  }
  throw std::logic_error("to_string: unreachable Device");
}

namespace {
/// The process-wide default device: what device(name) set, the CPU until then.
std::mutex g_device_mutex;
std::pair<Device, int> g_device{ Device::CPU, 0 };

std::string format_device(std::pair<Device, int> dev)
{
  std::string out = to_string(dev.first);
  if (uses_gpu(dev.first) && dev.second != 0) out += ":" + std::to_string(dev.second);
  return out;
}
} // namespace

std::string device(std::string_view name)
{
  const auto selected = detail::parse_device(name);
#if !defined(DTWC_HAS_CUDA) && !defined(DTWC_HAS_METAL)
  if (selected.first == Device::GPU) throw DeviceError(detail::gpu_not_built_message());
#endif
  if (selected.first == Device::CPU_GPU && !detail::hybrid_not_built_reason().empty())
    throw DeviceError(detail::hybrid_refused_message("[dtwc] dtwc::device", detail::hybrid_not_built_reason()));
  std::lock_guard<std::mutex> lk(g_device_mutex);
  g_device = selected;
  return format_device(g_device);
}

std::string device()
{
  std::lock_guard<std::mutex> lk(g_device_mutex);
  return format_device(g_device);
}

namespace detail {

std::pair<Device, int> parse_device(std::string_view name)
{
  const std::string raw = trim(name);
  const std::string lname = to_lower_copy(raw);

  if (lname == "cpu") return { Device::CPU, 0 };
  if (lname == "gpu" || lname == "cuda") return { Device::GPU, 0 };
  if (lname == "cpu+gpu") return { Device::CPU_GPU, 0 };
  if (lname.starts_with("gpu:") || lname.starts_with("cuda:") || lname.starts_with("cpu+gpu:")) {
    const std::string id = lname.substr(lname.find(':') + 1);
    if (!id.empty() && id.find_first_not_of("0123456789") == std::string::npos) {
      try {
        return { lname.starts_with("cpu+gpu:") ? Device::CPU_GPU : Device::GPU, std::stoi(id) };
      } catch (const std::out_of_range &) {
        // An absurdly large ordinal is an unknown name.
      }
    }
  }
  if (lname == "hpc" || lname.starts_with("hpc:"))
    throw DeviceError(
      "[dtwc] device '" + raw + "' submits a whole run to a SLURM cluster, which "
      "Python's dtwcpp.device(\"hpc\") and scripts/slurm/slurm_remote.sh do. C++, "
      "MATLAB and dtwc_cl compute where they start: use cpu or gpu.");
  throw DeviceError("[dtwc] unknown device '" + raw
                    + "'. Valid devices: cpu, gpu, gpu:N, cpu+gpu, cpu+gpu:N (aliases cuda, cuda:N).");
}

std::string gpu_not_built_message()
{
  return "[dtwc] device='gpu' requested but this build has no GPU backend compiled in.\n"
         "Rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, on macOS, -DDTWC_ENABLE_METAL=ON.\n"
         "This build will not silently fall back to CPU.";
}

std::string hybrid_refused_message(std::string_view where, std::string_view why, bool gpu_runs)
{
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  const char *const instead = gpu_runs ? "cpu or gpu" : "cpu";
#else
  static_cast<void>(gpu_runs);
  const char *const instead = "cpu"; // a build with no GPU backend refuses gpu too
#endif
  return std::string(where) + ": device 'cpu+gpu' splits FastCLARA's assignment pass and the distance matrix's fill "
         "between the CPU and a CUDA GPU, "
         "but " + std::string(why) + ". Use device " + instead + "; no single-device fallback was attempted.";
}

GpuRefusal gpu_not_built_refusal()
{
  return GpuRefusal(gpu_not_built_message(), "this build has no GPU backend compiled in");
}

GpuRefusal hybrid_refusal(std::string_view where, std::string_view why, bool gpu_runs)
{
  return GpuRefusal(hybrid_refused_message(where, why, gpu_runs), std::string(why));
}

GpuRefusal hybrid_streamed_refusal()
{
  return hybrid_refusal("fast_clara", "the series are streamed, and a streamed batch is assigned on one device");
}

GpuRefusal as_it_goes_refusal(std::string_view where, std::string_view method, Device device)
{
  const std::string name(method);
  const std::string reason =
    "method '" + name + "' computes its distances on the CPU as it goes, so the GPU would sit idle";
  if (device == Device::CPU_GPU)
    return GpuRefusal(
      hybrid_refused_message(where, reason + "; choose pam, kmedoids, hierarchical, mip, lrcore or clara", false),
      reason);
  return GpuRefusal(std::string(where) + ": method '" + name + "' computes its distances on the CPU as it goes, so "
                      "device 'gpu' would sit idle; the GPU fills the distance matrix that pam, kmedoids, mip, lrcore "
                      "and hierarchical use, and clara's sample matrices. Choose one of those, or device 'cpu'. "
                      "No CPU fallback was attempted.",
                    reason);
}

void say_fallback(Device requested, int index, std::string_view reason)
{
  std::cerr << "[dtwc] " + format_device({ requested, index }) + " requested: " + std::string(reason)
                 + "; running on cpu (fallback)\n";
}

std::string share_ignored_message(std::string_view where, std::string_view splits, Device device)
{
  return std::string(where) + ": gpu_share splits " + std::string(splits) + " of device cpu+gpu, but the device is "
         + to_string(device) + ", which would ignore it; leave gpu_share at -1.";
}

std::string hybrid_not_built_reason()
{
#if defined(DTWC_HAS_CUDA)
  return {};
#elif defined(DTWC_HAS_METAL)
  return "Metal has no kernel for that pass";
#else
  return "this build has no CUDA (rebuild with -DDTWC_ENABLE_CUDA=ON)";
#endif
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
