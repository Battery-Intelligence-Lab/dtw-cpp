/**
 * @file unit_test_soft_dtw_hotpath.cpp
 * @brief M46 allocation and unchecked-cell contract for Soft-DTW.
 */

#include <soft_dtw.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

namespace allocation_probe {

constexpr std::size_t large_allocation_floor = 500U * 1024U;
std::atomic<bool> enabled{false};
std::atomic<std::size_t> allocations{0};
std::atomic<std::size_t> large_allocations{0};

void* allocate(std::size_t size)
{
  if (enabled.load(std::memory_order_relaxed)) {
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (size >= large_allocation_floor)
      large_allocations.fetch_add(1, std::memory_order_relaxed);
  }
  if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
  throw std::bad_alloc{};
}

class Scope {
public:
  Scope()
  {
    allocations.store(0, std::memory_order_relaxed);
    large_allocations.store(0, std::memory_order_relaxed);
    enabled.store(true, std::memory_order_relaxed);
  }

  ~Scope() { enabled.store(false, std::memory_order_relaxed); }

  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};

} // namespace allocation_probe

void* operator new(std::size_t size)
{
  return allocation_probe::allocate(size);
}

void* operator new[](std::size_t size)
{
  return allocation_probe::allocate(size);
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

TEST_CASE("unchecked softmin cell is non-throwing and allocation-free",
          "[soft_dtw][softmin][allocation][m46]")
{
  static_assert(noexcept(dtwc::detail::softmin_gamma_unchecked(
    1.0, 2.0, 3.0, 0.5)));
  REQUIRE(dtwc::detail::softmin_gamma_unchecked(1.25, -2.0, 3.5, 0.75)
          == dtwc::softmin_gamma(1.25, -2.0, 3.5, 0.75));

  double checksum = 0.0;
  {
    allocation_probe::Scope probe;
    for (int iteration = 0; iteration < 100'000; ++iteration) {
      const double offset = static_cast<double>(iteration % 17) * 1e-6;
      checksum += dtwc::detail::softmin_gamma_unchecked(
        1.0 + offset, 2.0 - offset, 3.0 + offset, 0.5);
    }
  }

  REQUIRE(std::isfinite(checksum));
  REQUIRE(allocation_probe::allocations.load(std::memory_order_relaxed) == 0);
}

TEST_CASE("warmed Soft-DTW gradient reuses its large DP allocations",
          "[soft_dtw][gradient][allocation][m46]")
{
  std::vector<double> x(257);
  std::vector<double> y(255);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<double>((i * 7) % 29) / 29.0;
  for (std::size_t i = 0; i < y.size(); ++i)
    y[i] = static_cast<double>((i * 11) % 31) / 31.0;

  // Allocate the two thread-local ScratchMatrix buffers before probing.
  const auto warm = dtwc::soft_dtw_gradient<double>(x, y, 0.7);
  REQUIRE(warm.size() == x.size());

  std::vector<double> gradient;
  {
    allocation_probe::Scope probe;
    gradient = dtwc::soft_dtw_gradient<double>(x, y, 0.7);
  }

  REQUIRE(gradient.size() == x.size());
  REQUIRE(allocation_probe::large_allocations.load(std::memory_order_relaxed)
          == 0);
}
