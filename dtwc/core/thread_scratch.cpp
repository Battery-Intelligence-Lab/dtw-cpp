/**
 * @file thread_scratch.cpp
 * @brief The threads' scratch registries, their release, and KernelTeam (see thread_scratch.hpp).
 * @author Volkan Kumtepeli
 */

#include "thread_scratch.hpp"

#include "distance_matrix.hpp"          // release_free_memory
#include "../base/error.hpp"            // InvalidInput
#include "../base/parallelisation.hpp"  // get_max_threads

#include <algorithm>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwc::core {

namespace {

/// The calling thread's registered buffers. Each is a thread_local of this thread, so it outlives every use here.
thread_local std::vector<detail::ScratchEntry> registry;

std::size_t this_thread_bytes() noexcept
{
  std::size_t bytes = 0;
  for (const auto &entry : registry) bytes += entry.bytes(entry.buffer);
  return bytes;
}

void release_this_thread() noexcept
{
  for (const auto &entry : registry) entry.release(entry.buffer);
  std::vector<detail::ScratchEntry>().swap(registry); // each registers again when it next grows
}

/// What every region uses at most: dtwc's regions take their teams from get_max_threads(), never more. Read past the
/// calling thread's TeamCap, so a capped caller still reaches every thread that holds scratch.
int team_of_every_thread() { return full_team_threads(); }

} // namespace

void detail::register_scratch(const ScratchEntry &entry) { registry.push_back(entry); }

void release_thread_scratch() noexcept
{
#ifdef _OPENMP
#pragma omp parallel num_threads(team_of_every_thread())
  release_this_thread();
#else
  release_this_thread();
#endif
  release_free_memory();
}

std::vector<std::size_t> registered_scratch_bytes()
{
  std::vector<std::size_t> bytes(static_cast<std::size_t>(team_of_every_thread()), 0);
#ifdef _OPENMP
#pragma omp parallel num_threads(static_cast<int>(bytes.size()))
  bytes[static_cast<std::size_t>(omp_get_thread_num())] = this_thread_bytes();
#else
  bytes[0] = this_thread_bytes();
#endif
  return bytes;
}

KernelTeam::KernelTeam(std::size_t limit, bool verbose, std::size_t after)
  : limit_(limit), verbose_(verbose), after_(after), full_(get_max_threads()), team_(full_)
{
}

int KernelTeam::fitting(std::size_t rest, std::size_t per_thread) const noexcept
{
  if (limit_ == 0) return team_;
  per_thread = std::max(per_thread, per_thread_);
  rest = std::max(rest, after_);
  if (rest + priced_ > limit_) return 0;
  const auto team = static_cast<std::size_t>(team_);
  const std::size_t growth = per_thread - per_thread_;
  return static_cast<int>(growth == 0 ? team : std::min(team, (limit_ - rest - priced_) / growth));
}

int KernelTeam::fit(std::string_view phase, std::size_t rest, std::size_t per_thread, std::size_t length,
                    std::string_view held)
{
  per_thread = std::max(per_thread, per_thread_); // a thread keeps what it holds
  if (limit_ == 0) {
    per_thread_ = per_thread;
    return team_;
  }
  if (after_ > rest) held = "the names and labels read after it"; // the threads' scratch is held then too
  rest = std::max(rest, after_);
  // t threads of the team growing to per_thread each, the others keeping what they hold: priced_ + t * growth.
  const auto team = static_cast<std::size_t>(team_);
  const std::size_t growth = per_thread - per_thread_;
  std::size_t fits = 0;
  if (rest + priced_ <= limit_) fits = growth == 0 ? team : std::min(team, (limit_ - rest - priced_) / growth);
  const std::string already = priced_ != 0 ? " and the " + std::to_string(priced_) + " its threads hold already" : "";
  const std::string beside = " beside the " + std::to_string(rest) + " bytes the run holds with it" + already;
  // A thread's scratch: the DTW kernels' for the longest series, and the method's tables (FasterPAM's swap table, the
  // assignment's, the silhouettes' sums, TADPole's counts).
  const std::string what = " (the DTW kernels' for series of up to " + std::to_string(length)
                           + " time steps and the method's tables)";
  // A thread that holds scratch already grows by the difference: what one more needs beside the rest.
  const std::string more = per_thread_ != 0 ? ", " + std::to_string(growth) + " more than it holds" : "";
  if (rest > limit_) // what the run holds exceeds the limit before any thread's scratch
    throw InvalidInput(std::string(phase) + ": --ram-limit = " + std::to_string(limit_) + " cannot hold "
                       + (held.empty() ? std::string("what the run holds") : std::string(held)) + " alone, "
                       + std::to_string(rest) + " bytes, let alone one thread's scratch, " + std::to_string(per_thread)
                       + " bytes" + what + more + "," + beside + "; raise the limit.");
  if (fits == 0)
    throw InvalidInput(std::string(phase) + ": --ram-limit = " + std::to_string(limit_) + " cannot hold one thread's "
                       "scratch, " + std::to_string(per_thread) + " bytes" + what + more + "," + beside
                       + "; raise the limit.");
  if (fits < team && verbose_)
    std::cout << phase << ": " << fits << " of " << full_ << " threads, each holding " << per_thread
              << " bytes of scratch" << what << ", as many as --ram-limit = " << limit_ << " holds" << beside << "\n";
  priced_ += fits * growth;
  team_ = static_cast<int>(fits);
  per_thread_ = per_thread;
  return team_;
}

} // namespace dtwc::core
