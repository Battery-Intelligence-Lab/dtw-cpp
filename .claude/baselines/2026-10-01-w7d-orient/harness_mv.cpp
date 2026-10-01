// W7d in-process check of the per-pair WDTW path the fill runs: the closure of
// core::resolve_dtw_fn<double> (dtw_dispatch.cpp, compiled from base or head
// sources) over every pair of the variable-length band set, single-threaded.
// Prints, per band, the minimum and median of the sweep times over `reps`
// sweeps, and a checksum of the distances (must agree between builds).
// Usage: harness <varlen_dir> <reps>
#include "Data.hpp"
#include "core/dtw_dispatch.hpp"
#include "core/dtw_options.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
  namespace fs = std::filesystem;
  std::vector<fs::path> files;
  for (const auto &e : fs::directory_iterator(argv[1]))
    if (e.path().extension() == ".csv") files.push_back(e.path());
  std::sort(files.begin(), files.end());
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (const auto &f : files) {
    std::ifstream in(f);
    std::string line;
    std::getline(in, line); // header
    std::vector<double> v;
    while (std::getline(in, line)) v.push_back(std::stod(line.substr(line.find(',') + 1)));
    series.push_back(std::move(v));
    names.push_back(f.stem().string());
  }
  const int reps = std::stoi(argv[2]);
  // Two channels: series 2k and 2k+1 interleaved over their common length.
  std::vector<std::vector<double>> mv;
  std::vector<std::string> mv_names;
  for (std::size_t k = 0; k + 1 < series.size(); k += 2) {
    const std::size_t n = std::min(series[k].size(), series[k + 1].size());
    std::vector<double> v(2 * n);
    for (std::size_t t = 0; t < n; ++t) { v[2 * t] = series[k][t]; v[2 * t + 1] = series[k + 1][t]; }
    mv.push_back(std::move(v));
    mv_names.push_back(names[k]);
  }
  dtwc::Data data(std::move(mv), std::move(mv_names), 2);
  const std::size_t N = data.size();
  struct Case { const char *name; dtwc::core::DTWVariant variant; dtwc::core::MissingStrategy missing; int band; };
  using V = dtwc::core::DTWVariant;
  using M = dtwc::core::MissingStrategy;
  const Case cases[] = {
    { "mv_standard", V::Standard, M::Error, 200 }, { "mv_wdtw", V::WDTW, M::Error, 200 },
    { "mv_adtw", V::ADTW, M::Error, 200 },         { "mv_zero_cost", V::Standard, M::ZeroCost, 200 },
    { "mv_arow", V::Standard, M::AROW, 200 },
  };
  for (const auto &c : cases) {
    const int band = c.band;
    dtwc::core::DistanceConfig config;
    config.variant.variant = c.variant;
    config.missing = c.missing;
    config.band = band;
    config.ndim = 2;
    const auto fn = dtwc::core::resolve_dtw_fn<double>(config, data);
    std::vector<double> times;
    double checksum = 0;
    for (int r = 0; r < reps; ++r) {
      double sum = 0;
      const auto t0 = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j) sum += fn(data.series(i), data.series(j));
      const auto t1 = std::chrono::steady_clock::now();
      times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      checksum = sum;
    }
    std::sort(times.begin(), times.end());
    std::printf("%s band %d: min %.2f ms median %.2f ms checksum %.17g\n", c.name, band, times.front(),
                times[times.size() / 2], checksum);
  }
}
