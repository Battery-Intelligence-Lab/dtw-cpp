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
  dtwc::Data data(std::move(series), std::move(names));
  const std::size_t N = data.size();
  struct Case { const char *name; dtwc::core::DTWVariant variant; dtwc::core::MissingStrategy missing; int band; };
  using V = dtwc::core::DTWVariant;
  using M = dtwc::core::MissingStrategy;
  const Case cases[] = {
    { "wdtw", V::WDTW, M::Error, -1 }, { "standard", V::Standard, M::Error, -1 },
  };
  for (const auto &c : cases) {
    const int band = c.band;
    dtwc::core::DistanceConfig config;
    config.variant.variant = c.variant;
    config.missing = c.missing;
    config.band = band;
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
