/**
 * @file machine.cpp
 * @brief machine_facts(): the computer's name and system (one block an OS), the CPU (cpuinfo), the memory (llfio, in
 *        distance_matrix.cpp), the run's GPU (its backend) and the build (this file's own compiler macros).
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 5.5 (RR1a)
 * @date 03 Nov 2026
 */

#include "machine.hpp"
#include "distance_matrix.hpp" // available_memory
#include "../Problem.hpp"      // gpu_info

#ifdef DTWC_HAS_CUDA
#include "../cuda/cuda_dtw.cuh" // cuda_device_info
#endif
#ifdef DTWC_HAS_CPUINFO
#include <cpuinfo.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h> // GetComputerNameExA, GetModuleHandleW, GetProcAddress
#else
#include <sys/utsname.h> // uname
#include <unistd.h>      // gethostname
#endif

namespace dtwc::core {

namespace {

const char *const unknown = "unknown";

#ifdef _WIN32
std::string host_name()
{
  char name[256]{};
  DWORD size = static_cast<DWORD>(sizeof name);
  return GetComputerNameExA(ComputerNameDnsHostname, name, &size) && size > 0 ? std::string(name, size) : unknown;
}

/// "Windows <major>.<minor>.<build>" from ntdll's RtlGetVersion, which, unlike GetVersionEx, does not answer as the
/// executable's manifest says.
std::string os_name()
{
  using RtlGetVersion = LONG(WINAPI *)(RTL_OSVERSIONINFOW *);
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  const FARPROC address = ntdll ? GetProcAddress(ntdll, "RtlGetVersion") : nullptr;
  RTL_OSVERSIONINFOW version{};
  version.dwOSVersionInfoSize = static_cast<DWORD>(sizeof version);
  // Through void (*)(), the function pointer type a cast to any other is not warned about.
  if (!address || reinterpret_cast<RtlGetVersion>(reinterpret_cast<void (*)()>(address))(&version) != 0) return unknown;
  char text[64];
  std::snprintf(text, sizeof text, "Windows %lu.%lu.%lu", version.dwMajorVersion, version.dwMinorVersion,
                version.dwBuildNumber);
  return text;
}
#else
std::string host_name()
{
  char name[256]{};
  return gethostname(name, sizeof name - 1) == 0 && name[0] != '\0' ? std::string(name) : unknown;
}

/// /etc/os-release's PRETTY_NAME where the system has one (Linux, FreeBSD), and uname's system, release and machine.
std::string os_name()
{
  std::string pretty;
  std::ifstream release("/etc/os-release");
  for (std::string line; std::getline(release, line);)
    if (line.rfind("PRETTY_NAME=", 0) == 0) {
      pretty = line.substr(12);
      pretty.erase(std::remove(pretty.begin(), pretty.end(), '"'), pretty.end());
      break;
    }
  utsname system{};
  if (uname(&system) < 0) return pretty.empty() ? unknown : pretty;
  const std::string kernel = std::string(system.sysname) + " " + system.release + ", " + system.machine;
  return pretty.empty() ? kernel : pretty + " (" + kernel + ")";
}
#endif

/// The compiler of this file, dtwc_core's, and its version. clang-cl defines _MSC_FULL_VER as well: clang first.
std::string compiler()
{
  char text[64];
#if defined(__clang__) && defined(__apple_build_version__)
  std::snprintf(text, sizeof text, "Apple clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__clang__)
  std::snprintf(text, sizeof text, "clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
  std::snprintf(text, sizeof text, "GCC %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_FULL_VER)
  std::snprintf(text, sizeof text, "MSVC %d.%d.%d", _MSC_VER / 100, _MSC_VER % 100, _MSC_FULL_VER % 100000);
#else
  return unknown;
#endif
  return text;
}

/// The names whose flag is set, in order, else "none of" them all.
std::string named(const bool (&set)[4])
{
  static const char *const names[4] = { "AVX2", "FMA", "AVX-512F", "NEON" };
  std::string list;
  for (std::size_t i = 0; i < 4; ++i)
    if (set[i]) list += std::string(list.empty() ? "" : " ") + names[i];
  return list.empty() ? "none of AVX2, FMA, AVX-512F, NEON" : list;
}

/// The SIMD extensions this file's compiler may use: the arch level dtwc_core is built for (DTWC_ARCH_LEVEL).
std::string build_simd()
{
  bool set[4]{};
#ifdef __AVX2__
  set[0] = true;
#endif
#ifdef __FMA__
  set[1] = true;
#endif
#ifdef __AVX512F__
  set[2] = true;
#endif
#ifdef __ARM_NEON
  set[3] = true;
#endif
  return named(set);
}

/// The optional libraries dtwc_core was built with, as far as its own definitions tell (Arrow is dtwc_io's, YAML
/// dtwc_cli's).
std::string libraries()
{
  std::string list;
#ifdef DTWC_HAS_OPENMP
  list += ", OpenMP";
#endif
#ifdef DTWC_HAS_MMAP
  list += ", llfio";
#endif
#ifdef DTWC_HAS_CPUINFO
  list += ", cpuinfo";
#endif
#ifdef DTWC_ENABLE_HIGHS
  list += ", HiGHS";
#endif
#ifdef DTWC_ENABLE_GUROBI
  list += ", Gurobi";
#endif
#ifdef DTWC_HAS_CUDA
  list += ", CUDA";
#endif
#ifdef DTWC_HAS_METAL
  list += ", Metal";
#endif
  return list.empty() ? "none" : list.substr(2);
}

std::string gib(std::size_t bytes)
{
  char text[32];
  std::snprintf(text, sizeof text, "%.1f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  return text;
}

} // namespace

MachineFacts machine_facts(Device device, int device_index)
{
  MachineFacts facts;
  facts.host = host_name();
  facts.os = os_name();
  facts.cpu = unknown;
  facts.cpu_simd = unknown;
#ifdef DTWC_HAS_CPUINFO
  // It fails where the system does not let it read the CPU (a closed /sys), and then every getter aborts: none is
  // called unless it succeeded.
  if (cpuinfo_initialize()) {
    if (const cpuinfo_package *package = cpuinfo_get_packages_count() > 0 ? cpuinfo_get_package(0) : nullptr) {
      const char *const end = std::find(package->name, package->name + sizeof package->name, '\0');
      if (end != package->name) facts.cpu.assign(package->name, end);
    }
    facts.physical_cores = cpuinfo_get_cores_count();
    facts.logical_cores = cpuinfo_get_processors_count();
    const bool set[4] = { cpuinfo_has_x86_avx2(), cpuinfo_has_x86_fma3(), cpuinfo_has_x86_avx512f(),
                          cpuinfo_has_arm_neon() };
    facts.cpu_simd = named(set);
  }
#endif
  facts.build_simd = build_simd();
  facts.ram_total = total_memory();
  facts.ram_available = available_memory();
  if (device == Device::GPU) {
#ifdef DTWC_HAS_CUDA
    facts.gpu = "CUDA: " + cuda::cuda_device_info(device_index); // gpu_info()'s line for GPU 0
#else
    (void)device_index;
    facts.gpu = gpu_info(); // Metal's one GPU, or why there is none
#endif
  }
  facts.compiler = compiler();
  facts.libraries = libraries();
  return facts;
}

std::string MachineFacts::text() const
{
  std::string out;
  const auto line = [&out](const char *label, const std::string &value) {
    out += "  ";
    out += label;
    out.append(17 - std::strlen(label), ' ');
    out += value;
    out += '\n';
  };
  line("Host:", host);
  line("OS:", os);
  line("CPU:", cpu);
  line("Cores:", physical_cores > 0 && logical_cores > 0
                   ? std::to_string(physical_cores) + " physical, " + std::to_string(logical_cores) + " logical"
                   : unknown);
  line("SIMD:", cpu_simd + " (CPU), " + build_simd + " (build)");
  line("RAM:", ram_total > 0 ? gib(ram_total) + " total, " + (ram_available > 0 ? gib(ram_available) : unknown)
                                 + " available"
                             : unknown);
  if (gpu) line("GPU:", *gpu);
  line("Build:", compiler + "; " + libraries);
  return out;
}

} // namespace dtwc::core
