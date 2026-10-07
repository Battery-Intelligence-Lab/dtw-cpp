/**
 * @file test_problem_set_device.cpp
 * @brief Problem::set_device and the one device-name grammar
 *        (detail::parse_device) that dtwc::device() and the bindings share.
 *
 * @details Every case runs in every build: GPU builds check the selected
 * backend, a build without one checks the verbatim no-GPU DeviceError. Oracle for
 * the messages: the literal strings below.
 */

#include <dtwc.hpp>

#include "../support/dtw_route_bound.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using dtwc::Device;

namespace {

// set_device and set_gpu_precision are the whole device surface: the strategy
// enum's accessors and the CUDA settings struct's are gone.
template <class P>
constexpr bool has_strategy_surface = requires { &P::distance_strategy; } || requires { &P::set_distance_strategy; }
                                      || requires { &P::cuda_settings; } || requires { &P::set_cuda_settings; };
template <class P>
constexpr bool has_device_surface = requires { &P::set_device; &P::set_gpu_precision; &P::gpu_precision; };

// The DeviceError strings, verbatim.
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

TEST_CASE("IF-1: the strategy names are gone; set_device and set_gpu_precision remain",
          "[if1][device]")
{
  STATIC_REQUIRE_FALSE(has_strategy_surface<dtwc::Problem>);
  STATIC_REQUIRE(has_device_surface<dtwc::Problem>);
}

TEST_CASE("IF-1: set_device(cpu) computes on the CPU", "[if1][device]")
{
  auto prob = problem_with_data();
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  prob.set_device(Device::GPU);
#endif
  prob.set_device(Device::CPU, 3); // the CPU has no index
  CHECK(prob.device() == std::pair{ Device::CPU, 0 });
  prob.fill_distance_matrix();
  // The fill runs the SIMD lanes; dtwFull_L is the per-pair kernel.
  const double per_pair = dtwc::dtwFull_L<double>(prob.series(0), prob.series(1));
  CHECK(dtwc::test_support::dtw_routes_agree<double>(
    prob.dist_by_ind(0, 1), per_pair, prob.series(0).size(), prob.series(1).size()));
}

TEST_CASE("IF-1: set_device(gpu) records this build's GPU index, or refuses it",
          "[if1][device]")
{
  auto prob = problem_with_data();
#if defined(DTWC_HAS_CUDA)
  prob.set_device(Device::GPU, 2);
  CHECK(prob.device() == std::pair{ Device::GPU, 2 });
#elif defined(DTWC_HAS_METAL)
  // Metal runs on the system default GPU: another index is refused, never run on GPU 0.
  CHECK_THAT(message_of<dtwc::DeviceError>([&] { prob.set_device(Device::GPU, 2); }),
             ContainsSubstring("Metal runs on the system default GPU, but GPU index = 2"));
  CHECK(prob.device() == std::pair{ Device::CPU, 0 });
#endif
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  prob.set_device(Device::GPU);
  CHECK(prob.device() == std::pair{ Device::GPU, 0 });
#else
  CHECK(message_of<dtwc::DeviceError>([&] { prob.set_device(Device::GPU); })
        == kMsgGpuNotBuilt);
  CHECK(prob.device() == std::pair{ Device::CPU, 0 });
#endif
}

TEST_CASE("IF-1: set_device rejects a negative index", "[if1][device]")
{
  auto prob = problem_with_data();
  CHECK_THAT(message_of<dtwc::InvalidInput>([&] { prob.set_device(Device::GPU, -1); }),
             ContainsSubstring("got -1"));
  CHECK(prob.device() == std::pair{ Device::CPU, 0 });
}

TEST_CASE("IF-1: a Problem does not read the process-wide device", "[if1][device]")
{
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  (void)dtwc::device("gpu");
#endif
  const dtwc::Problem prob("fresh");
  CHECK(prob.device() == std::pair{ Device::CPU, 0 });
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
  const auto sorted = [](std::vector<dtwc::index_t> v) {
    std::sort(v.begin(), v.end());
    return v;
  };
  const auto cpu = dtwc::cluster(dataset, 2, "pam", -1, "cpu");
  CHECK(sorted(cpu.medoids()) == std::vector<dtwc::index_t>{ 1, 4 });
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  if (!dtwc::gpu_available()) {
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
