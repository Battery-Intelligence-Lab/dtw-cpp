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
int team_of_every_thread()
{
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

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

int KernelTeam::fit(std::string_view phase, std::size_t rest, std::size_t per_thread, std::size_t length)
{
  per_thread = std::max(per_thread, per_thread_); // a thread keeps what it holds
  if (limit_ == 0) {
    per_thread_ = per_thread;
    return team_;
  }
  rest = std::max(rest, after_); // the threads' scratch is held after the call's phases too
  // t threads of the team growing to per_thread each, the others keeping what they hold: priced_ + t * growth.
  const auto team = static_cast<std::size_t>(team_);
  const std::size_t growth = per_thread - per_thread_;
  std::size_t fits = 0;
  if (rest + priced_ <= limit_) fits = growth == 0 ? team : std::min(team, (limit_ - rest - priced_) / growth);
  const std::string beside = " beside the " + std::to_string(rest) + " bytes the run holds with it"
                             + (priced_ != 0 ? " and the " + std::to_string(priced_) + " its threads hold already" : "");
  if (fits == 0)
    throw InvalidInput(std::string(phase) + ": --ram-limit = " + std::to_string(limit_) + " cannot hold one thread's "
                       "DTW scratch, " + std::to_string(per_thread) + " bytes for series of up to "
                       + std::to_string(length) + " time steps," + beside + "; raise the limit.");
  if (fits < team && verbose_)
    std::cout << phase << ": " << fits << " of " << full_ << " threads, each holding " << per_thread
              << " bytes of DTW scratch for series of up to " << length << " time steps, as many as --ram-limit = "
              << limit_ << " holds" << beside << "\n";
  priced_ += fits * growth;
  team_ = static_cast<int>(fits);
  per_thread_ = per_thread;
  return team_;
}

} // namespace dtwc::core
