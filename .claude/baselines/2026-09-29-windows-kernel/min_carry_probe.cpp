// A/B/C: library StandardCell (init-list min) vs nested min vs nested min + left carried in a register.
#include "core/dtw_kernel.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
using namespace dtwc::core;
struct NestedCell {
  template <typename T> T combine(T d, T u, T l, T c, std::size_t, std::size_t) const noexcept { return std::min(std::min(d, u), l) + c; }
  template <typename T> T seed(T c, std::size_t, std::size_t) const noexcept { return c; }
};
// Same recurrence as dtw_kernel_linear (no early abandon), with the row pointer hoisted and dp[i-1, j] carried.
template <typename T, typename Cost, typename Cell>
T linear_carry(std::size_t n_short, std::size_t n_long, Cost cost, Cell cell) {
  constexpr T maxValue = std::numeric_limits<T>::max();
  thread_local static std::vector<T> buf;
  buf.resize(n_short);
  T *s = buf.data();
  s[0] = cell.seed(cost(0, 0), 0, 0);
  for (std::size_t i = 1; i < n_short; ++i) s[i] = cell.combine(maxValue, s[i - 1], maxValue, cost(i, 0), i, 0);
  for (std::size_t j = 1; j < n_long; ++j) {
    T diag = s[0];
    T left = cell.combine(maxValue, maxValue, s[0], cost(0, j), 0, j);
    s[0] = left;
    for (std::size_t i = 1; i < n_short; ++i) {
      const T up = s[i];
      left = cell.combine(diag, up, left, cost(i, j), i, j);
      diag = up;
      s[i] = left;
    }
  }
  return s[n_short - 1];
}
int main() {
  std::mt19937_64 g(42); std::normal_distribution<double> nd;
  for (std::size_t L : {100, 1000, 4000}) {
    std::vector<double> x(L), y(L); for (auto& v : x) v = nd(g); for (auto& v : y) v = nd(g);
    const double* px = x.data(); const double* py = y.data();
    auto cost = [px, py](std::size_t i, std::size_t j) { return std::abs(px[i] - py[j]); };
    const int reps = L <= 100 ? 2000 : (L <= 1000 ? 40 : 4);
    std::vector<double> ta, tb, tc; double ra = 0, rb = 0, rc = 0;
    for (int round = 0; round < 11; ++round) {
      auto t0 = std::chrono::steady_clock::now(); for (int r = 0; r < reps; ++r) ra += dtw_kernel_linear<double>(L, L, cost, StandardCell{});
      auto t1 = std::chrono::steady_clock::now(); for (int r = 0; r < reps; ++r) rb += dtw_kernel_linear<double>(L, L, cost, NestedCell{});
      auto t2 = std::chrono::steady_clock::now(); for (int r = 0; r < reps; ++r) rc += linear_carry<double>(L, L, cost, NestedCell{});
      auto t3 = std::chrono::steady_clock::now();
      const double cells = reps * double(L) * L;
      ta.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / cells);
      tb.push_back(std::chrono::duration<double, std::nano>(t2 - t1).count() / cells);
      tc.push_back(std::chrono::duration<double, std::nano>(t3 - t2).count() / cells);
    }
    std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end()); std::sort(tc.begin(), tc.end());
    std::printf("L=%zu  init_list %.3f  nested %.3f  nested+carry %.3f ns/cell  (x%.2f, x%.2f)  identical=%d%d\n",
                L, ta[5], tb[5], tc[5], ta[5] / tb[5], ta[5] / tc[5], ra == rb, ra == rc);
  }
}
