// Where do the base and candidate kernels disagree for ADTW (f32 and f64)? Same sweep as pprobe check
// (lengths 1..1000, three partners, six bands, five data kinds, seeds 0-2); prints a histogram of the
// mismatches by the shorter length and by band, and the first few cases.
#include "core/dtw_kernel.hpp" // the candidate, dtwc::core
#include "core/dtw_cost.hpp"
#include "base_kernel.hpp"     // the base, base_core
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

static std::uint64_t splitmix(std::uint64_t &s)
{
  std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double unit(std::uint64_t &s) { return double(splitmix(s) >> 11) * 0x1.0p-53; }
template <class T> void gen(std::vector<T> &v, std::size_t n, std::uint64_t seed, int kind)
{
  v.resize(n);
  std::uint64_t s = seed;
  double walk = 0;
  for (std::size_t i = 0; i < n; ++i) {
    switch (kind) {
    case 0: v[i] = T(2 * unit(s) - 1); break;
    case 1: walk += 2 * unit(s) - 1; v[i] = T(walk); break;
    case 2: v[i] = T(splitmix(s) % 3); break;
    case 3: {
      const double m = double(std::numeric_limits<T>::max());
      const double u = unit(s);
      v[i] = T(u < 0.25 ? 0.0 : (u < 0.6 ? m * (0.5 + unit(s) / 2) : -m * (0.5 + unit(s) / 2)));
      break;
    }
    default: walk += 2 * unit(s) - 1; v[i] = unit(s) < 1.0 / 6 ? std::numeric_limits<T>::quiet_NaN() : T(walk);
    }
  }
}

template <class T> void run(const char *tname)
{
  long long total = 0, mism = 0;
  std::map<int, long long> by_short, by_band;
  int shown = 0;
  for (int seed = 0; seed < 3; ++seed)
    for (int kind = 0; kind < 5; ++kind)
      for (int nx = 1; nx <= 1000; ++nx) {
        std::uint64_t s = 0xabcdefULL * (kind + 1) + 104729ULL * nx + 0x9e3779b9ULL * seed;
        const int nys[3] = { nx, 1 + int(splitmix(s) % 1000), std::max(1, nx + int(splitmix(s) % 21) - 10) };
        for (int ny : nys) {
          std::vector<T> x, y;
          gen(x, nx, s + 11, kind);
          gen(y, ny, s + 29, kind);
          if (kind == 2 && ny == nx && nx % 3 == 0) y = x;
          const int L = std::max(nx, ny);
          int bi = 0;
          for (int band : { -1, 0, 1, L / 10, L, std::abs(nx - ny) }) {
            const T a = base_core::run_dtw<dtwc::core::SpanL1Cost>(x.data(), std::size_t(nx), y.data(), std::size_t(ny),
                                                                   band, base_core::ADTWCell<T>{ T(0.5) }, T(-1));
            const T b = dtwc::core::run_dtw<dtwc::core::SpanL1Cost>(x.data(), std::size_t(nx), y.data(), std::size_t(ny),
                                                                    band, dtwc::core::ADTWCell<T>{ T(0.5) }, T(-1));
            ++total;
            if (std::memcmp(&a, &b, sizeof(T)) != 0) {
              ++mism;
              ++by_short[std::min(nx, ny)];
              ++by_band[bi];
              if (shown++ < 6)
                std::printf("  %s mismatch: kind %d seed %d nx %d ny %d band %d: base %.9g candidate %.9g\n", tname, kind,
                            seed, nx, ny, band, double(a), double(b));
            }
            ++bi;
          }
        }
      }
  std::printf("%s ADTW: %lld outputs, %lld differ\n", tname, total, mism);
  for (auto [k, v] : by_short) std::printf("  shorter length %d: %lld\n", k, v);
  for (auto [k, v] : by_band) std::printf("  band slot %d (-1, 0, 1, L/10, L, |nx-ny|): %lld\n", k, v);
}

int main()
{
  run<float>("f32");
  run<double>("f64");
}
