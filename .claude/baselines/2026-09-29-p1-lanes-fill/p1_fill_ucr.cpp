// P1: time Problem::fill_distance_matrix (24 OpenMP threads, band as given) on one
// UCR 2018 TSV file (label column first), and hash the packed matrix so the per-pair
// and the lanes binaries can be shown to fill it bit for bit alike. Wall time and the
// process's CPU time (all threads, user + kernel) around the fill alone.
//   p1_fill_ucr.exe <file.tsv> <band> <rounds>
#include "Problem.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double process_cpu_seconds()
{
  FILETIME created, exited, kernel, user;
  GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
  const auto ticks = [](FILETIME f) { return (ULONGLONG(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
  return double(ticks(kernel) + ticks(user)) * 1e-7;
}

int main(int argc, char **argv)
{
  if (argc < 4) {
    std::fprintf(stderr, "usage: p1_fill_ucr <file.tsv> <band> <rounds>\n");
    return 2;
  }
  const int band = std::atoi(argv[2]);
  const int rounds = std::atoi(argv[3]);

  std::vector<std::vector<double>> rows;
  std::ifstream in(argv[1]);
  for (std::string line; std::getline(in, line);) {
    std::istringstream fields(line);
    std::string field;
    std::getline(fields, field, '\t'); // class label
    std::vector<double> row;
    while (std::getline(fields, field, '\t'))
      if (!field.empty() && field != "\r") row.push_back(std::strtod(field.c_str(), nullptr));
    if (!row.empty()) rows.push_back(std::move(row));
  }
  std::printf("file=%s N=%zu L=%zu band=%d\n", argv[1], rows.size(), rows.empty() ? 0 : rows[0].size(), band);

  for (int r = 0; r < rounds; ++r) {
    auto copy = rows;
    std::vector<std::string> names;
    for (std::size_t i = 0; i < copy.size(); ++i) names.push_back("s" + std::to_string(i));
    dtwc::Problem prob("ucr");
    prob.set_data(dtwc::Data(std::move(copy), std::move(names)));
    prob.band = band;
    const double c0 = process_cpu_seconds();
    const auto t0 = std::chrono::steady_clock::now();
    prob.fill_distance_matrix();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double cpu = process_cpu_seconds() - c0;

    std::uint64_t h = 1469598103934665603ull; // FNV-1a over the packed upper triangle
    const auto &m = prob.dense_distance_matrix();
    for (std::size_t i = 0; i < rows.size(); ++i)
      for (std::size_t j = i + 1; j < rows.size(); ++j) {
        const double d = m.get(i, j);
        unsigned char b[sizeof d];
        std::memcpy(b, &d, sizeof d);
        for (unsigned char c : b) h = (h ^ c) * 1099511628211ull;
      }
    std::printf("round %d fill_s=%.4f cpu_s=%.3f hash=%016llx\n", r, s, cpu,
                static_cast<unsigned long long>(h));
    std::fflush(stdout);
  }
  return 0;
}
