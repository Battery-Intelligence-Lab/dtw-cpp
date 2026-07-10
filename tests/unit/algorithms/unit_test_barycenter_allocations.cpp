/**
 * @file unit_test_barycenter_allocations.cpp
 * @brief Allocation regression gate for the hard-DTW barycenter workspace.
 */

#include <dtwc.hpp>
#include <algorithms/barycenter.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdlib>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace allocation_probe {

constexpr std::size_t large_allocation_floor = 500U * 1024U;
std::atomic<bool> enabled{false};
std::atomic<std::size_t> large_allocations{0};

void* allocate(std::size_t size)
{
  if (enabled.load(std::memory_order_relaxed)
      && size >= large_allocation_floor)
    large_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
  throw std::bad_alloc{};
}

} // namespace allocation_probe

void* operator new(std::size_t size)
{
  return allocation_probe::allocate(size);
}

void* operator new[](std::size_t size)
{
  return allocation_probe::allocate(size);
}

void operator delete(void* memory) noexcept
{
  std::free(memory);
}

void operator delete[](void* memory) noexcept
{
  std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
  std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept
{
  std::free(memory);
}

namespace {

class ProbeScope {
public:
  ProbeScope()
  {
    allocation_probe::large_allocations.store(0, std::memory_order_relaxed);
    allocation_probe::enabled.store(true, std::memory_order_relaxed);
  }

  ~ProbeScope()
  {
    allocation_probe::enabled.store(false, std::memory_order_relaxed);
  }

  ProbeScope(const ProbeScope&) = delete;
  ProbeScope& operator=(const ProbeScope&) = delete;
};

dtwc::Problem make_long_problem()
{
  std::vector<std::vector<dtwc::data_t>> series;
  std::vector<std::string> names;
  for (const std::size_t length : {257U, 255U, 253U}) {
    std::vector<dtwc::data_t> values(length);
    for (std::size_t i = 0; i < length; ++i)
      values[i] = static_cast<double>(i) / static_cast<double>(length)
                  + 0.01 * static_cast<double>(series.size());
    series.push_back(std::move(values));
    names.push_back("long_" + std::to_string(series.size()));
  }
  dtwc::Problem problem("barycenter_allocation_probe");
  problem.set_data(dtwc::Data(std::move(series), std::move(names)));
  problem.set_band(dtwc::settings::DEFAULT_BAND);
  return problem;
}

} // namespace

TEST_CASE("hard-DTW barycenter reuses one caller-owned DP matrix",
          "[barycenter][allocation]")
{
  auto problem = make_long_problem();
  dtwc::algorithms::BarycenterOptions options;
  options.method = dtwc::algorithms::BarycenterMethod::DBA;
  options.max_iter = 1;
  options.tolerance = 0.0;

  {
    ProbeScope probe;
    const auto center = dtwc::algorithms::dtw_barycenter(
      problem, {0, 1, 2}, 257, options);
    REQUIRE(center.size() == 257);
  }

  const auto count = allocation_probe::large_allocations.load(
    std::memory_order_relaxed);
  INFO("allocations >= 500 KiB during one 3-series DBA iteration: " << count);
  REQUIRE(count == 1);
}
