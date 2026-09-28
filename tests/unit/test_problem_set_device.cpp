/**
 * @file test_problem_set_device.cpp
 * @brief IF-1: Problem::set_device and the one device-name grammar
 *        (detail::parse_device) that dtwc::device() and the bindings share.
 *
 * @details Every case runs in every build: GPU builds check the selected
 * backend, a build without one checks the verbatim §6.1 DeviceError. Oracle for
 * the messages: the frozen strings of docs/api-contract-2.0.md §6.1.
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using dtwc::Device;
using dtwc::DistanceMatrixStrategy;

namespace {

// docs/api-contract-2.0.md §6.1, verbatim.
const std::string kMsgUnknownTpu =
  "[dtwc] unknown device 'tpu'. Valid devices: cpu, gpu, gpu:N (aliases cuda, cuda:N).";
[[maybe_unused]] const std::string kMsgGpuNotBuilt =
  "[dtwc] device='gpu' requested but this build has no GPU backend compiled in.\n"
  "Rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, on macOS, -DDTWC_ENABLE_METAL=ON.\n"
  "This build will not silently fall back to CPU.";

template <typename Error, typename F>
std::string message_of(F &&f)
{
  try {
    f();
  } catch (const Error &e) {
    return e.what();
  }
  FAIL("the expected exception was not thrown");
  return {};
}

dtwc::Problem problem_with_data()
{
  dtwc::Problem prob("set_device");
  prob.set_data(dtwc::Data(
    std::vector<std::vector<double>>{ { 0.0, 1.0, 2.0, 1.0 }, { 0.0, 2.0, 1.0, 0.0 },
                                      { 5.0, 5.0, 4.0, 5.0 } },
    { "a", "b", "c" }));
  return prob;
}

} // namespace

TEST_CASE("IF-1: parse_device is the one device grammar", "[if1][device]")
{
  using dtwc::detail::parse_device;
  CHECK(parse_device("cpu") == std::pair{ Device::CPU, 0 });
  CHECK(parse_device(" GpU:3 ") == std::pair{ Device::GPU, 3 });
  CHECK(parse_device("cuda") == std::pair{ Device::GPU, 0 });
  CHECK(parse_device("CUDA:0") == std::pair{ Device::GPU, 0 });
  CHECK_THROWS_AS(parse_device("hpc"), dtwc::DeviceError);
  CHECK(message_of<dtwc::DeviceError>([] { (void)parse_device("tpu"); })
        == kMsgUnknownTpu);
  for (const char *bad : { "gpu:", "gpu:-1", "gpu:x", "cuda:99999999999", "" })
    CHECK_THROWS_AS(parse_device(bad), dtwc::DeviceError);
}

TEST_CASE("IF-1: set_device(cpu) leaves a CPU strategy alone and moves a GPU one to Auto",
          "[if1][device]")
{
  auto prob = problem_with_data();
  prob.set_device(Device::CPU);
  CHECK(prob.distance_strategy == DistanceMatrixStrategy::Auto);

  for (const auto chosen : { DistanceMatrixStrategy::BruteForce,
                             DistanceMatrixStrategy::Pruned }) {
    prob.set_distance_strategy(chosen);
    prob.set_device(Device::CPU);
    CHECK(prob.distance_strategy == chosen);
  }
  for (const auto gpu : { DistanceMatrixStrategy::CUDA,
                          DistanceMatrixStrategy::Metal }) {
    prob.set_distance_strategy(gpu);
    prob.set_device(Device::CPU);
    CHECK(prob.distance_strategy == DistanceMatrixStrategy::Auto);
  }
  prob.fill_distance_matrix();
  CHECK(prob.dist_by_ind(0, 1) == dtwc::dtwFull_L<double>(prob.series(0), prob.series(1)));
}

TEST_CASE("IF-1: set_device(gpu) selects this build's backend and records the index",
          "[if1][device]")
{
  auto prob = problem_with_data();
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  prob.set_device(Device::GPU, 2);
#  if defined(DTWC_HAS_CUDA)
  CHECK(prob.distance_strategy == DistanceMatrixStrategy::CUDA);
#  else
  CHECK(prob.distance_strategy == DistanceMatrixStrategy::Metal);
#  endif
  CHECK(prob.cuda_settings.device_id == 2);
  prob.set_device(Device::GPU);
  CHECK(prob.cuda_settings.device_id == 0);
#else
  CHECK(message_of<dtwc::DeviceError>([&] { prob.set_device(Device::GPU); })
        == kMsgGpuNotBuilt);
  CHECK(prob.distance_strategy == DistanceMatrixStrategy::Auto);
#endif
}

TEST_CASE("IF-1: set_device rejects a negative index", "[if1][device]")
{
  auto prob = problem_with_data();
  CHECK_THAT(message_of<dtwc::InvalidInput>([&] { prob.set_device(Device::GPU, -1); }),
             ContainsSubstring("got -1"));
  CHECK(prob.distance_strategy == DistanceMatrixStrategy::Auto);
  CHECK(prob.cuda_settings.device_id == 0);
}

TEST_CASE("IF-1: a Problem does not read the process-wide device", "[if1][device]")
{
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  (void)dtwc::device("gpu");
#endif
  const dtwc::Problem prob("fresh");
  CHECK(prob.distance_strategy == DistanceMatrixStrategy::Auto);
  CHECK(dtwc::device("cpu") == "cpu");
}

TEST_CASE("IF-1: Tier-1 cluster(device=gpu) runs through Problem::set_device",
          "[if1][device][tier1]")
{
  // Two clusters of three constant series; each cluster's middle series is its
  // medoid by a margin no FP32 rounding can flip.
  dtwc::Dataset::series_type series;
  for (const double level : { 0.1, 0.3, 0.5, 10.1, 10.3, 10.5 })
    series.emplace_back(5, level);
  const auto dataset = dtwc::load(series);
  const auto sorted = [](std::vector<int> v) {
    std::sort(v.begin(), v.end());
    return v;
  };
  const auto cpu = dtwc::cluster(dataset, 2, "pam", -1, "cpu");
  CHECK(sorted(cpu.medoids()) == std::vector<int>{ 1, 4 });
#if defined(DTWC_HAS_METAL)
  if (!dtwc::metal::metal_available()) {
    CHECK_THROWS_AS(dtwc::cluster(dataset, 2, "pam", -1, "gpu"), dtwc::DeviceError);
    return;
  }
#endif
#if defined(DTWC_HAS_CUDA)
  if (!dtwc::cuda::cuda_available()) {
    CHECK_THROWS_AS(dtwc::cluster(dataset, 2, "pam", -1, "gpu"), dtwc::DeviceError);
    return;
  }
#endif
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  const auto gpu = dtwc::cluster(dataset, 2, "pam", -1, "gpu");
  CHECK(gpu.device() == "gpu");
  CHECK(sorted(gpu.medoids()) == sorted(cpu.medoids()));
  for (std::size_t i = 0; i < series.size(); ++i)
    for (std::size_t j = 0; j < series.size(); ++j)
      CHECK((gpu.labels()[i] == gpu.labels()[j])
            == (cpu.labels()[i] == cpu.labels()[j]));
#  if defined(DTWC_HAS_METAL)
  // Metal computes in FP32, so every distance it returns is a float; the CPU's
  // FP64 distances for these levels are not. Proof that the Metal route ran.
  const auto float_exact = [](const std::vector<double> &m) {
    return std::all_of(m.begin(), m.end(), [](double v) {
      return static_cast<double>(static_cast<float>(v)) == v;
    });
  };
  CHECK(float_exact(gpu.distance_matrix()));
  CHECK_FALSE(float_exact(cpu.distance_matrix()));
#  endif
#else
  CHECK(message_of<dtwc::DeviceError>(
          [&] { (void)dtwc::cluster(dataset, 2, "pam", -1, "gpu"); })
        == kMsgGpuNotBuilt);
#endif
}
