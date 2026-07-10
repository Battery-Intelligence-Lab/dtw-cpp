/**
 * @file bench_openmp_schedule.cpp
 * @brief Measure-first OpenMP schedule sweep for the triangular DTW hot loop.
 *
 * This benchmark deliberately uses schedule(runtime) so dynamic,1;
 * dynamic,16; and guided can be compared in one identical binary. It does not
 * alter production scheduling until the measurements justify a change.
 */

#include <benchmark/benchmark.h>

#include <warping.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using Dataset = std::vector<std::vector<double>>;

Dataset make_dataset(int n, int length)
{
  Dataset data(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    auto& series = data[static_cast<std::size_t>(i)];
    series.resize(static_cast<std::size_t>(length + (i % 17)));
    for (std::size_t t = 0; t < series.size(); ++t)
      series[t] = std::sin(0.031 * static_cast<double>(t) + i * 0.07)
                  + 0.001 * static_cast<double>(i * static_cast<int>(t));
  }
  return data;
}

enum class Schedule { Dynamic1, Dynamic16, Guided };

void run_schedule(benchmark::State& state, Schedule schedule)
{
  const int n = static_cast<int>(state.range(0));
  const int length = static_cast<int>(state.range(1));
  const auto data = make_dataset(n, length);
#ifdef _OPENMP
  switch (schedule) {
    case Schedule::Dynamic1: omp_set_schedule(omp_sched_dynamic, 1); break;
    case Schedule::Dynamic16: omp_set_schedule(omp_sched_dynamic, 16); break;
    case Schedule::Guided: omp_set_schedule(omp_sched_guided, 1); break;
  }
#endif
  for (auto _ : state) {
    double checksum = 0.0;
    #pragma omp parallel for schedule(runtime) reduction(+:checksum)
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j)
        checksum += dtwc::dtwFull_L<double>(data[static_cast<std::size_t>(i)],
                                            data[static_cast<std::size_t>(j)]);
    }
    benchmark::DoNotOptimize(checksum);
  }
  state.SetItemsProcessed(state.iterations() * n * (n - 1) / 2);
}

void dynamic_1(benchmark::State& state) { run_schedule(state, Schedule::Dynamic1); }
void dynamic_16(benchmark::State& state) { run_schedule(state, Schedule::Dynamic16); }
void guided(benchmark::State& state) { run_schedule(state, Schedule::Guided); }

BENCHMARK(dynamic_1)->Args({64, 128})->UseRealTime();
BENCHMARK(dynamic_16)->Args({64, 128})->UseRealTime();
BENCHMARK(guided)->Args({64, 128})->UseRealTime();

} // namespace

BENCHMARK_MAIN();

