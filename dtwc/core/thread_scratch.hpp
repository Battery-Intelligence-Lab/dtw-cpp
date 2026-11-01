/**
 * @file thread_scratch.hpp
 * @brief The thread_local scratch of the DTW kernels and the algorithms: grown exactly, registered with its thread so
 *        that a top-level call gives it back when it returns, and priced where a memory limit sizes a phase's team.
 *
 * @details A scratch vector grows and never shrinks within a top-level call, so a kernel run again on its thread
 *          allocates nothing. reserve_scratch() is its growth path: the old block is freed first (its values are
 *          scratch) and exactly n elements are taken, so a thread holds what core::thread_scratch_bytes prices, never a
 *          growth factor more; a buffer that grows from empty joins its thread's registry. release_thread_scratch()
 *          empties every registered buffer of every thread of the OpenMP team (a buffer emptied registers again when it
 *          next grows): dtwc::run calls it as it starts and when it returns or throws (ScratchRelease); Tier-2 calls
 *          keep their scratch, so a k-sweep allocates once. KernelTeam sizes each phase's team within an explicit limit
 *          (TeamCap applies it).
 * @author Volkan Kumtepeli
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace dtwc::core {

namespace detail {

/// One registered buffer: where it is, its bytes, and how it is emptied (a std::vector<T>, its type erased).
struct ScratchEntry
{
  void *buffer;
  std::size_t (*bytes)(const void *) noexcept;
  void (*release)(void *) noexcept;
};

/// Adds `entry` to the calling thread's registry.
void register_scratch(const ScratchEntry &entry);

template <typename T>
std::size_t vector_bytes(const void *buffer) noexcept
{
  return static_cast<const std::vector<T> *>(buffer)->capacity() * sizeof(T);
}

template <typename T>
void vector_release(void *buffer) noexcept
{
  std::vector<T>().swap(*static_cast<std::vector<T> *>(buffer));
}

} // namespace detail

/// Registers `buffer`, one of this thread's thread_local scratch vectors, as it grows from empty.
template <typename T>
void register_scratch(std::vector<T> &buffer)
{
  detail::register_scratch({ &buffer, &detail::vector_bytes<T>, &detail::vector_release<T> });
}

namespace detail {

/// reserve_scratch's growth, out of line: a kernel keeps its code but for one compare and a call it rarely takes.
template <typename T>
#if defined(__GNUC__)
[[gnu::noinline]]
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
void make_room(std::vector<T> &buffer, std::size_t n)
{
  if (buffer.capacity() == 0)
    core::register_scratch(buffer);
  else
    std::vector<T>().swap(buffer); // its values are scratch: freed before the larger block is taken
  buffer.reserve(n);               // exactly n: resize alone grows by a factor
}

} // namespace detail

/// Room for n elements in `buffer`, one of this thread's thread_local scratch vectors, inserted before the resize or
/// assign that sizes it, which stays as it was: when it must grow, its block is freed first and exactly n taken (its
/// values are lost: scratch), and one growing from empty registers with this thread (register_scratch).
template <typename T>
void reserve_scratch(std::vector<T> &buffer, std::size_t n)
{
  if (buffer.capacity() < n) detail::make_room(buffer, n);
}

/// Empties every registered scratch buffer of every thread of the OpenMP team, the calling thread's too, in one
/// parallel region, and returns the freed memory to the system (release_free_memory): what a top-level call does when
/// it returns or throws. A team of the size every region uses at most (omp_get_max_threads()) holds every thread that
/// ran one.
void release_thread_scratch() noexcept;

/// The bytes of registered scratch each thread of the OpenMP team holds, by its thread number in one parallel region
/// (the calling thread's first): a test hook.
std::vector<std::size_t> registered_scratch_bytes();

/// release_thread_scratch() when it is made and when it goes out of scope, on return and on error: dtwc::run's guard,
/// so a run starts with no scratch an earlier call left (a Tier-2 k-sweep's) and leaves none.
struct ScratchRelease
{
  ScratchRelease() { release_thread_scratch(); }
  ScratchRelease(const ScratchRelease &) = delete;
  ScratchRelease &operator=(const ScratchRelease &) = delete;
  ~ScratchRelease() { release_thread_scratch(); }
};

/**
 * @brief The DTW kernels' thread scratch of one call within an explicit memory limit, phase by phase: each phase's
 *        team (the threads its parallel regions use, TeamCap) is the most that fit beside what the phase holds.
 *
 * A team only shrinks, and a thread it leaves keeps what it holds, since scratch grows and never shrinks within a
 * call: the call's threads hold `priced()`, the team's at the most a phase gave them and the others at what they held
 * when they left. That counts each thread once only while a smaller team is the calling thread and workers of the
 * larger one, as libgomp and libomp reuse their workers in order. Without a limit (0: the automatic budget) nothing
 * shrinks, is refused or is priced.
 */
class KernelTeam
{
public:
  /// `after`: what the call holds once its phases are over (a stream's names, read after it), beside the scratch the
  /// threads keep: every phase leaves room for it.
  KernelTeam(std::size_t limit, bool verbose, std::size_t after = 0);

  /// The team of `phase`, whose threads each hold `per_thread` bytes for series of up to `length` time steps
  /// (thread_scratch_bytes and the method's tables) beside the `rest` the run holds with it (`held` says what that is):
  /// every thread of the last phase's team that fits, the threads left out keeping theirs (verbose says how many and
  /// why); InvalidInput, naming the bytes and the limit, before the phase runs when one thread does not fit, or when
  /// `rest` alone does not.
  int fit(std::string_view phase, std::size_t rest, std::size_t per_thread, std::size_t length,
          std::string_view held = {});
  /// The team fit() would give, nothing changed and nothing refused: 0 where one thread does not fit.
  int fitting(std::size_t rest, std::size_t per_thread) const noexcept;

  int threads() const noexcept { return team_; }           ///< the last phase's team
  bool prices() const noexcept { return limit_ != 0; }      ///< an explicit limit: teams are fitted, scratch priced
  std::size_t priced() const noexcept { return priced_; }  ///< what the call's threads hold of their scratch so far

private:
  std::size_t limit_;
  bool verbose_;
  std::size_t after_;
  int full_, team_;             // the OpenMP team; the last phase's
  std::size_t per_thread_ = 0;  // what each thread of the team holds
  std::size_t priced_ = 0;      // what every thread of the call holds
};

} // namespace dtwc::core
