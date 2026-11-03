/**
 * @file machine.hpp
 * @brief The facts about the computer a run ran on: its name, its system, its CPU, its memory, the run's GPU, and the
 *        build. `dtwc_cl -v` prints them.
 *
 * @details Each fact is what the system or a library says, or "unknown" where this build or the system cannot say:
 *          never a guess. The CPU's model, cores and SIMD extensions come from cpuinfo (`DTWC_HAS_CPUINFO`, the build
 *          option `DTWC_ENABLE_CPUINFO`), the memory from llfio (`DTWC_HAS_MMAP`), a GPU's name and memory from its
 *          backend.
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 5.5 (RR1a)
 * @date 03 Nov 2026
 */

#pragma once

#include "../base/env.hpp" // Device

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace dtwc::core {

/// The computer a run ran on and the build it ran: each text "unknown" where it is not known, each number 0.
struct MachineFacts
{
  std::string host;                 ///< the computer's name
  std::string os;                   ///< the operating system and its version
  std::string cpu;                  ///< the CPU's model, as cpuinfo names it
  std::uint32_t physical_cores = 0; ///< the CPU's cores, every package's (cpuinfo)
  std::uint32_t logical_cores = 0;  ///< its logical processors (cpuinfo)
  std::string cpu_simd;             ///< which of AVX2, FMA, AVX-512F and NEON the CPU has (cpuinfo)
  std::string build_simd;           ///< which of them dtwc_core's code may use (its compiler's __AVX2__ and the like)
  std::size_t ram_total = 0;        ///< total_memory(), bytes
  std::size_t ram_available = 0;    ///< available_memory(), bytes
  std::optional<std::string> gpu;   ///< the run's GPU and its memory: a GPU run's only
  std::string compiler;             ///< the compiler that built dtwc_core, and its version
  std::string libraries;            ///< the optional libraries dtwc_core was built with, or "none"

  /// One line a fact, "  Label:" in a 19-column field and its value, as `dtwc_cl -v`'s diagnostics print them.
  std::string text() const;
};

/// This computer's facts for a run on `device` (`device_index`: the GPU's ordinal, `gpu:N`); the GPU only for
/// Device::GPU: CUDA's GPU `device_index`, else gpu_info()'s line.
MachineFacts machine_facts(Device device = Device::CPU, int device_index = 0);

/// The physical memory of this computer, in bytes: llfio's total, which available_memory() never exceeds; 0 when
/// unknown, as in a build without llfio. Defined in distance_matrix.cpp, beside available_memory().
std::size_t total_memory();

} // namespace dtwc::core
