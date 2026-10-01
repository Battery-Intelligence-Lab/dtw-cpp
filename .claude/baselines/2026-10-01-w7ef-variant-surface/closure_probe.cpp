// W7ef: the bound per-pair closures this unit changed, timed and counted on one
// thread: heap allocations per call after a warm-up, and ns per call (median of 9
// blocks of 20,000 calls). Interpolate (float64), WDTW float64 and float32, each on
// pairs whose lengths the series have, and WDTW on a length they lack (a DBA
// centroid's). Built against the base and the head library with the library's flags.
#include <dtwc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <vector>

static std::atomic<bool> counting{ false };
static std::atomic<long long> allocations{ 0 };

void *operator new(std::size_t size)
{
  if (counting.load(std::memory_order_relaxed)) allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc{};
}
void *operator new[](std::size_t size) { return operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

static double checksum = 0;

template <typename Fn, typename T>
void measure(const char *name, const Fn &fn, std::span<const T> x, std::span<const T> y)
{
  checksum += fn(x, y); // warm-up: thread_local buffers grow here
  allocations = 0;
  counting = true;
  for (int i = 0; i < 1000; ++i) checksum += fn(x, y);
  counting = false;
  std::vector<double> block_ns;
  for (int block = 0; block < 9; ++block) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20000; ++i) checksum += fn(x, y);
    block_ns.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 20000);
  }
  std::sort(block_ns.begin(), block_ns.end());
  std::printf("%-26s allocations per call: %.3f  ns per call: %.1f\n", name, allocations.load() / 1000.0, block_ns[4]);
}

template <typename T>
std::vector<T> wave(std::size_t n, double f, double offset)
{
  std::vector<T> s(n);
  for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<T>(std::sin(f * static_cast<double>(i)) + offset);
  return s;
}

int main()
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  {
    auto a = wave<double>(64, 0.1, 0), b = wave<double>(64, 0.07, 0.5), c = wave<double>(64, 0.05, 0.3);
    for (int i : { 0, 1, 2, 20, 21, 22, 60, 61, 62, 63 }) b[i] = nan;
    for (int i : { 5, 6, 40 }) c[i] = nan;
    dtwc::Problem prob("interp_probe");
    prob.set_data(dtwc::Data{ std::vector<std::vector<double>>{ a, b, c }, { "a", "b", "c" } });
    prob.set_missing_strategy(dtwc::core::MissingStrategy::Interpolate);
    const auto &fn = prob.dtw_function();
    const std::span<const double> sa{ a }, sb{ b }, sc{ c };
    measure("interp: no NaN", fn, sa, sa.subspan(0, 63));
    measure("interp: one gappy", fn, sa, sb);
    measure("interp: two gappy", fn, sb, sc);
  }
  const dtwc::core::DTWVariantParams wdtw{ .variant = dtwc::core::DTWVariant::WDTW };
  {
    const auto a = wave<double>(64, 0.1, 0), b = wave<double>(63, 0.07, 0.5), e = wave<double>(80, 0.03, 0.2);
    dtwc::Problem prob("wdtw64_probe");
    prob.set_data(dtwc::Data{ std::vector<std::vector<double>>{ a, b }, { "a", "b" } });
    prob.set_variant(wdtw);
    const auto &fn = prob.dtw_function();
    measure("wdtw f64: bound length", fn, std::span<const double>{ a }, std::span<const double>{ b });
    measure("wdtw f64: other length", fn, std::span<const double>{ a }, std::span<const double>{ e });
  }
  {
    const auto a = wave<float>(64, 0.1, 0), b = wave<float>(63, 0.07, 0.5), e = wave<float>(80, 0.03, 0.2);
    dtwc::Problem prob("wdtw32_probe");
    prob.set_data(dtwc::Data{ std::vector<std::vector<float>>{ a, b }, { "a", "b" } });
    prob.set_variant(wdtw);
    const auto &fn = prob.dtw_function_f32();
    measure("wdtw f32: bound length", fn, std::span<const float>{ a }, std::span<const float>{ b });
    measure("wdtw f32: other length", fn, std::span<const float>{ a }, std::span<const float>{ e });
  }
  std::printf("checksum %.17g\n", checksum);
  return 0;
}
