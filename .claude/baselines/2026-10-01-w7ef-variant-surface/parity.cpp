// W7ef parity harness: prints, in hex, the values a step must keep (dtwFull, the
// Soft-DTW gradient, the Interpolate and WDTW routes) and the one it may move within
// 1e-12 relative (soft_dtw). Built against the base and the head headers with the
// library's flags (no LTO); `diff` of the two outputs is the check.
#include <distance.hpp>
#include <soft_dtw.hpp>
#include <warping.hpp>
#include <warping_wdtw.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

namespace {

std::uint64_t state = 0x9E3779B97F4A7C15ull;
double uniform() // splitmix64 -> [0, 1), the same on every standard library
{
  std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) * 0x1.0p-53;
}

template <typename T>
std::vector<T> walk(std::size_t n)
{
  std::vector<T> s(n);
  double v = 0;
  for (auto &x : s) {
    v += uniform() - 0.5;
    x = static_cast<T>(v);
  }
  return s;
}

template <typename T>
void run(const char *tag)
{
  const std::size_t lengths[][2] = { { 1, 1 }, { 1, 7 }, { 2, 2 }, { 5, 3 }, { 17, 17 }, { 31, 64 }, { 200, 150 }, { 513, 700 } };
  const double gammas[] = { 1e-3, 0.1, 0.7, 1.0, 10.0, static_cast<double>(std::numeric_limits<T>::denorm_min()),
                            static_cast<double>(std::numeric_limits<T>::min()) };
  for (const auto &l : lengths) {
    const auto x = walk<T>(l[0]);
    const auto y = walk<T>(l[1]);
    const std::span<const T> xs{ x }, ys{ y };
    std::printf("%s n=%zu m=%zu dtwFull_l1 %a dtwFull_sq %a dtwFull_ptr %a\n", tag, l[0], l[1],
                static_cast<double>(dtwc::dtwFull<T>(x, y)),
                static_cast<double>(dtwc::dtwFull<T>(x, y, dtwc::core::MetricType::SquaredL2)),
                static_cast<double>(dtwc::dtwFull<T>(x.data(), x.size(), y.data(), y.size())));
    for (const double g : gammas) {
      const T gamma = static_cast<T>(g);
      std::printf("%s n=%zu m=%zu gamma=%a soft %a soft_yx %a\n", tag, l[0], l[1], g,
                  static_cast<double>(dtwc::soft_dtw<T>(xs, ys, gamma)),
                  static_cast<double>(dtwc::soft_dtw<T>(ys, xs, gamma)));
      const auto grad = dtwc::soft_dtw_gradient<T>(xs, ys, gamma);
      std::printf("%s n=%zu m=%zu gamma=%a grad", tag, l[0], l[1], g);
      for (const T v : grad) std::printf(" %a", static_cast<double>(v));
      std::printf("\n");
    }
    // Interpolate (the facade) with gaps at the start, inside and at the end.
    auto xg = x, yg = y;
    const T nan = std::numeric_limits<T>::quiet_NaN();
    if (xg.size() > 3) xg[0] = xg[xg.size() / 2] = xg.back() = nan;
    if (yg.size() > 4) yg[1] = yg[2] = nan;
    for (const int band : { -1, 3 }) {
      const dtwc::core::DTWVariantParams standard{};
      std::printf("%s n=%zu m=%zu band=%d interp %a\n", tag, l[0], l[1], band,
                  static_cast<double>(dtwc::distance::dtw<T>(std::span<const T>{ xg }, std::span<const T>{ yg },
                                                             standard, band, dtwc::core::MetricType::L1,
                                                             dtwc::core::MissingStrategy::Interpolate)));
      std::printf("%s n=%zu m=%zu band=%d wdtw %a\n", tag, l[0], l[1], band,
                  static_cast<double>(dtwc::wdtwBanded<T>(xs, ys, band, static_cast<T>(0.05))));
    }
  }
}

} // namespace

int main()
{
  run<double>("f64");
  state = 0x9E3779B97F4A7C15ull;
  run<float>("f32");
  return 0;
}
