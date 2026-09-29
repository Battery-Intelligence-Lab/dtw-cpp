// P1: single-thread kernel ratio on one core, compiled with the flags of the lane TU
// (dtwc/core/dtw_lanes.cpp: native, -O3 -march=native, the library's FP subset).
// Per round, interleaved: 8 pairs through the per-pair kernels (dtwFull_L, the
// recurrence the lanes mirror; dtwFull_eap, the unbanded fill's per-pair kernel;
// dtwBanded) against one dtw_kernel_lanes call on the same 8 pairs.
//   p1_kernel_ab.exe <L> <band> <rounds>
#include "core/dtw_kernel.hpp"
#include "warping.hpp"
#include "../tests/support/deterministic_series.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using clk = std::chrono::steady_clock;

template <typename F>
double seconds(F &&f, int reps)
{
  const auto t0 = clk::now();
  for (int r = 0; r < reps; ++r) f();
  return std::chrono::duration<double>(clk::now() - t0).count() / reps;
}

double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char **argv)
{
  const std::size_t n = argc > 1 ? std::size_t(std::atoi(argv[1])) : 1000;
  const int band = argc > 2 ? std::atoi(argv[2]) : -1;
  const int rounds = argc > 3 ? std::atoi(argv[3]) : 11;
  constexpr std::size_t W = dtwc::core::dtw_lanes<double>;
  const auto x = dtwc::test_support::benchmark_series(n, 42);
  std::vector<std::vector<double>> y;
  const double *ys[W];
  for (std::size_t w = 0; w < W; ++w) y.push_back(dtwc::test_support::benchmark_series(n, 43 + unsigned(w)));
  for (std::size_t w = 0; w < W; ++w) ys[w] = y[w].data();

  volatile double sink = 0;
  std::array<double, W> lanes{};
  auto run_lanes = [&] {
    lanes = dtwc::core::dtw_kernel_lanes<double>(x.data(), ys, n, band, dtwc::detail::L1Dist{},
                                                 dtwc::core::StandardCell{});
    sink = sink + lanes[0];
  };
  double pair[W];
  auto run_pairs = [&] { // what the fill ran per pair: EAP unbanded, dtwBanded banded
    for (std::size_t w = 0; w < W; ++w)
      pair[w] = band < 0 ? dtwc::dtwFull_eap<double>(x, y[w]) : dtwc::dtwBanded<double>(x, y[w], band);
    sink = sink + pair[0];
  };
  double linear[W];
  auto run_linear = [&] { // the recurrence the lanes mirror (dtwBanded is dtwFull_L for band < 0)
    for (std::size_t w = 0; w < W; ++w) linear[w] = dtwc::dtwBanded<double>(x, y[w], band);
    sink = sink + linear[0];
  };
  const double cells = double(W) * (band < 0 ? double(n) * double(n) : [&] {
    double c = 0;
    for (std::size_t j = 0; j < n; ++j) {
      const auto [lo, hi] = dtwc::core::dtw_band_bounds(band, j, n);
      c += double(hi - lo);
    }
    return c;
  }());
  const int reps = std::max(1, int(2e8 / cells));
  run_lanes(); run_pairs(); run_linear(); // warm-up, scratch growth
  int same = 1;
  for (std::size_t w = 0; w < W; ++w)
    same &= std::memcmp(&lanes[w], &pair[w], sizeof(double)) == 0
            && std::memcmp(&lanes[w], &linear[w], sizeof(double)) == 0;

  std::vector<double> tl, tp, tr, rp, rr;
  for (int r = 0; r < rounds; ++r) {
    double a, b, c;
    if (r % 2 == 0) { a = seconds(run_pairs, reps); c = seconds(run_linear, reps); b = seconds(run_lanes, reps); }
    else { b = seconds(run_lanes, reps); c = seconds(run_linear, reps); a = seconds(run_pairs, reps); }
    tp.push_back(a * 1e9 / cells); tl.push_back(b * 1e9 / cells); tr.push_back(c * 1e9 / cells);
    rp.push_back(a / b); rr.push_back(c / b);
  }
  std::printf("L=%zu band=%d W=%zu reps=%d rounds=%d bitwise_equal=%d\n", n, band, W, reps, rounds, same);
  std::printf("ns/cell median [min-max]: per-pair %.3f [%.3f-%.3f]  linear/banded %.3f [%.3f-%.3f]  lanes %.3f [%.3f-%.3f]\n",
              median(tp), *std::min_element(tp.begin(), tp.end()), *std::max_element(tp.begin(), tp.end()),
              median(tr), *std::min_element(tr.begin(), tr.end()), *std::max_element(tr.begin(), tr.end()),
              median(tl), *std::min_element(tl.begin(), tl.end()), *std::max_element(tl.begin(), tl.end()));
  std::printf("ratio median [min-max]: per-pair/lanes %.2f [%.2f-%.2f]  linear/lanes %.2f [%.2f-%.2f]\n",
              median(rp), *std::min_element(rp.begin(), rp.end()), *std::max_element(rp.begin(), rp.end()),
              median(rr), *std::min_element(rr.begin(), rr.end()), *std::max_element(rr.begin(), rr.end()));
  return same ? 0 : 1;
}
