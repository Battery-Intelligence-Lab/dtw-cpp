/**
 * @file test_gpu_fallback.cpp
 * @brief `fallback`: a GPU request that cannot be honoured is a DeviceError, or, where the caller says so, a run on
 *        the CPU that says so on stderr and in the run report.
 *
 * @details The request is one no GPU honours on any build (variant WDTW: a build with no GPU backend has no GPU at all,
 * one with a backend has no kernel for it), so every case runs here and in the CUDA and Metal trees, and each
 * fallback is checked against the CPU's own result. What only a GPU build can refuse (no device, a GPU index past the
 * last, a method with no GPU path) is the same branch of the same code; the CUDA tree's commands run it.
 *
 * @author Volkan Kumtepeli
 */

#include <dtwc.hpp>
#include <algorithms/one_batch_pam.hpp>
#include <algorithms/tadpole.hpp>
#include <cdist.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::EndsWith;
using Catch::Matchers::StartsWith;
using dtwc::Device;
using dtwc::Method;

namespace {

/// What the code under it writes on std::cerr, kept apart from the test's own output.
class CerrCapture
{
public:
  CerrCapture() : before_(std::cerr.rdbuf(text_.rdbuf())) {}
  ~CerrCapture() { std::cerr.rdbuf(before_); }
  CerrCapture(const CerrCapture &) = delete;
  CerrCapture &operator=(const CerrCapture &) = delete;
  std::string text() const { return text_.str(); }

private:
  std::ostringstream text_;
  std::streambuf *before_;
};

/// Two groups of three constant series: each group's middle series is its medoid.
dtwc::Data levels()
{
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (const double level : { 0.1, 0.3, 0.5, 10.1, 10.3, 10.5 }) {
    series.emplace_back(5, level);
    names.push_back(std::to_string(names.size()));
  }
  return dtwc::Data(std::move(series), std::move(names));
}

/// A GPU request no GPU honours: WDTW, which no kernel runs (a build with no GPU backend refuses the GPU itself).
dtwc::Config gpu_request(Method method, bool fallback)
{
  dtwc::Config config;
  config.k = 2;
  config.method = method;
  config.device = Device::GPU;
  config.variant.variant = dtwc::core::DTWVariant::WDTW;
  config.fallback = fallback;
  config.output.clear(); // write nothing
  config.tadpole_dc = 3.0;
  return config;
}

dtwc::Config on_cpu(dtwc::Config config)
{
  config.device = Device::CPU;
  config.fallback = false;
  return config;
}

/// `text` with every run of white space one space, as the report wraps its rows at any of them.
std::string flattened(const std::string &text)
{
  std::string out;
  for (const char c : text) {
    const bool space = c == ' ' || c == '\n' || c == '\r' || c == '\t';
    if (!space) out += c;
    else if (!out.empty() && out.back() != ' ') out += ' ';
  }
  return out;
}

/// The one line a fallback writes on stderr, naming the device the request asked for.
void check_says_fallback(const std::string &written, const std::string &requested = "gpu")
{
  CHECK_THAT(written, StartsWith("[dtwc] " + requested + " requested: "));
  CHECK_THAT(written, EndsWith("; running on cpu (fallback)\n"));
  CHECK(std::count(written.begin(), written.end(), '\n') == 1);
}

} // namespace

TEST_CASE("fallback: a GPU request that cannot be honoured is an error, or runs on the CPU and says so",
          "[device][fallback]")
{
  SECTION("dtwc::run: the DeviceError without fallback, the CPU's own run with it")
  {
    CHECK_THROWS_AS(dtwc::run(gpu_request(Method::PAM, false), levels()), dtwc::DeviceError);

    const CerrCapture cerr;
    const auto result = dtwc::run(gpu_request(Method::PAM, true), levels());
    const auto expected = dtwc::run(on_cpu(gpu_request(Method::PAM, true)), levels());
    check_says_fallback(cerr.text());
    CHECK(result.device() == "cpu");
    CHECK(result.labels() == expected.labels());
    CHECK(result.medoids() == expected.medoids());
    CHECK(result.cost() == expected.cost());
    CHECK_THAT(flattened(result.report()), ContainsSubstring(", device cpu (gpu requested; fallback: "));
    CHECK_THAT(flattened(result.report()), ContainsSubstring("--device \"gpu\" --fallback")); // the command that asked
    CHECK_THAT(expected.report(), !ContainsSubstring("gpu requested"));

    // The line and the report's row name the GPU asked for, ordinal included.
    auto second = gpu_request(Method::PAM, true);
    second.device_index = 1;
    const CerrCapture second_cerr;
    const auto on_second = dtwc::run(second, levels());
    check_says_fallback(second_cerr.text(), "gpu:1");
    CHECK_THAT(flattened(on_second.report()), ContainsSubstring(", device cpu (gpu:1 requested; fallback: "));
  }

  SECTION("series in memory that a GPU cannot take (float32, two features a time step) run on the CPU from the check")
  {
    // The kernels take Standard DTW, so only the series refuse; the run knows them before it plans or reads anything.
    std::vector<std::vector<double>> pairs;
    std::vector<std::string> names;
    for (const double level : { 0.1, 0.3, 0.5, 10.1, 10.3, 10.5 }) {
      pairs.emplace_back(6, level); // three time steps of two features
      names.push_back(std::to_string(names.size()));
    }
    const dtwc::Data two_features(std::move(pairs), std::move(names), 2);
    const dtwc::Data single = dtwc::detail::owned_copy(levels(), dtwc::core::Precision::Float32);
    for (const dtwc::Data *series : { &single, &two_features }) {
      auto config = gpu_request(Method::PAM, false);
      config.variant.variant = dtwc::core::DTWVariant::Standard;
      CHECK_THROWS_AS(dtwc::run(config, *series), dtwc::DeviceError);
      config.fallback = true;
      const CerrCapture cerr;
      CHECK(dtwc::run(config, *series).device() == "cpu");
      check_says_fallback(cerr.text());
    }
  }

  SECTION("a forced share: refused on a device that would ignore it, dropped where cpu+gpu's split has no CPU twin")
  {
    auto forced = gpu_request(Method::PAM, true);
    forced.gpu_share = 0.5;
    {
      const CerrCapture cerr;
      CHECK_THROWS_AS(dtwc::run(forced, levels()), dtwc::InvalidInput); // on every build, fallback or not
      CHECK(cerr.text().empty());                                       // and no fallback is announced
    }
    forced.device = Device::CPU_GPU;
    const CerrCapture cerr;
    const auto result = dtwc::run(forced, levels());
    CHECK(result.device() == "cpu");
    // The line and the report's row name the device asked for, which here is the GPU together with the CPU.
    check_says_fallback(cerr.text(), "cpu+gpu");
    CHECK_THAT(flattened(result.report()), ContainsSubstring(", device cpu (cpu+gpu requested; fallback: "));
  }

  SECTION("a method with no GPU path runs on the CPU too, and the same run without fallback is refused")
  {
    for (const auto method : { Method::OneBatch, Method::TADPole }) {
      CAPTURE(dtwc::name_of(dtwc::method_names, method));
      CHECK_THROWS_AS(dtwc::run(gpu_request(method, false), levels()), dtwc::DeviceError);
      const CerrCapture cerr;
      const auto result = dtwc::run(gpu_request(method, true), levels());
      check_says_fallback(cerr.text());
      CHECK(result.device() == "cpu");
      CHECK(result.labels() == dtwc::run(on_cpu(gpu_request(method, true)), levels()).labels());
    }
  }

  SECTION("Tier 2: a Problem keeps its GPU request and computes the call on the CPU")
  {
    dtwc::Problem cpu("cpu");
    cpu.set_data(levels());
    cpu.set_n_clusters(2);

    dtwc::Problem prob("fallback");
    prob.set_data(levels());
    prob.set_n_clusters(2);
    prob.set_device(Device::GPU, 0, true); // stored on any build: it is resolved where a method or a fill starts
    CHECK(prob.fallback());
    CHECK(prob.device() == std::pair{ Device::GPU, 0 });

    {
      // A forced share is for a fill or a pass these methods have no part in: refused whatever the fallback, before the
      // fallback says anything.
      prob.set_gpu_share(0.5);
      const CerrCapture cerr;
      CHECK_THROWS_AS(dtwc::algorithms::one_batch_pam(prob, { .n_clusters = 2 }), dtwc::InvalidInput);
      CHECK_THROWS_AS(dtwc::algorithms::tadpole(prob, 2, 3.0), dtwc::InvalidInput);
      CHECK(cerr.text().empty());
      prob.set_gpu_share(-1.0);
    }
    {
      const CerrCapture cerr;
      const auto onebatch = dtwc::algorithms::one_batch_pam(prob, { .n_clusters = 2 });
      check_says_fallback(cerr.text());
      CHECK(onebatch.labels == dtwc::algorithms::one_batch_pam(cpu, { .n_clusters = 2 }).labels);
    }
    {
      const CerrCapture cerr;
      const auto tadpole = dtwc::algorithms::tadpole(prob, 2, 3.0);
      check_says_fallback(cerr.text());
      CHECK(tadpole.labels == dtwc::algorithms::tadpole(cpu, 2, 3.0).labels);
    }
    {
      prob.set_variant(dtwc::core::DTWVariant::WDTW);
      cpu.set_variant(dtwc::core::DTWVariant::WDTW);
      const CerrCapture cerr;
      prob.fill_distance_matrix();
      check_says_fallback(cerr.text());
      cpu.fill_distance_matrix();
      CHECK(prob.is_distance_matrix_filled());
      for (std::size_t i = 0; i < prob.size(); ++i)
        for (std::size_t j = 0; j < i; ++j) CHECK(prob.dist_by_ind(i, j) == cpu.dist_by_ind(i, j));
      prob.fill_distance_matrix(); // filled: no device is looked at, and nothing more is said
      const std::string written = cerr.text();
      CHECK(std::count(written.begin(), written.end(), '\n') == 1);
    }
    {
      // Problem::cluster settles the request before its method's own work: TADPole's automatic cutoff samples
      // distances first, and the nested fill and function do not say it again.
      prob.set_method(Method::TADPole);
      cpu.set_method(Method::TADPole);
      const CerrCapture cerr;
      const auto clustered = prob.cluster();
      check_says_fallback(cerr.text());
      CHECK(clustered.labels == cpu.cluster().labels);
    }
    CHECK(prob.device() == std::pair{ Device::GPU, 0 }); // the request outlives the calls
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
    // Without fallback a GPU Problem has no part to give onebatch and tadpole: they are refused, as devices.md says.
    dtwc::Problem strict("strict");
    strict.set_data(levels());
    strict.set_device(Device::GPU);
    CHECK_THROWS_AS(dtwc::algorithms::one_batch_pam(strict, { .n_clusters = 2 }), dtwc::DeviceError);
    CHECK_THROWS_AS(dtwc::algorithms::tadpole(strict, 2, 3.0), dtwc::DeviceError);
    strict.set_n_clusters(2);
    strict.set_method(Method::TADPole);
    CHECK_THROWS_AS(strict.cluster(), dtwc::DeviceError);
#endif
  }

#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  SECTION("a GPU build: a request the GPU can honour computes there and says nothing, and with no device it is a fallback")
  {
    const bool device_here = dtwc::gpu_available();
    auto config = gpu_request(Method::PAM, true);
    config.variant.variant = dtwc::core::DTWVariant::Standard;
    {
      const CerrCapture cerr;
      CHECK(dtwc::run(config, levels()).device() == (device_here ? "gpu" : "cpu"));
      if (device_here)
        CHECK(cerr.text().empty());
      else
        check_says_fallback(cerr.text());
    }
    {
      const CerrCapture cerr;
      dtwc::Problem prob("gpu");
      prob.set_data(levels());
      prob.set_device(Device::GPU, 0, true);
      prob.fill_distance_matrix();
      CHECK(prob.is_distance_matrix_filled());
      if (device_here)
        CHECK(cerr.text().empty());
      else
        check_says_fallback(cerr.text());
    }
  }
#endif

  SECTION("cdist: the DeviceError without fallback, the CPU's matrix with it")
  {
    const auto data = levels();
    const dtwc::core::DistanceConfig distance{ .variant = { .variant = dtwc::core::DTWVariant::WDTW } };
    CHECK_THROWS_AS(dtwc::cdist(data, data, distance, Device::GPU), dtwc::DeviceError);
    {
      const CerrCapture cerr;
      const auto matrix = dtwc::cdist(data, data, distance, Device::GPU, 0, dtwc::GpuPrecision::Auto, -1.0, true);
      check_says_fallback(cerr.text());
      CHECK(matrix == dtwc::cdist(data, data, distance));
    }

    // A GPU index below 0 and a band below -1 are malformed whatever the device can do: InvalidInput before the
    // device is asked, with or without fallback, so a fallback never announces itself and then fails.
    const CerrCapture cerr;
    auto narrow = distance;
    narrow.band = -2;
    for (const bool fallback : { false, true }) {
      CAPTURE(fallback);
      CHECK_THROWS_AS(dtwc::cdist(data, data, distance, Device::GPU, -1, dtwc::GpuPrecision::Auto, -1.0, fallback),
                      dtwc::InvalidInput);
      CHECK_THROWS_AS(dtwc::cdist(data, data, narrow, Device::GPU, 0, dtwc::GpuPrecision::Auto, -1.0, fallback),
                      dtwc::InvalidInput);
    }
    CHECK(cerr.text().empty());
  }
}

TEST_CASE("fallback: a checkpoint records the precision of the device that computes",
          "[device][fallback][checkpoint]")
{
  // The request is a GPU run of WDTW, which no GPU takes, so every pair here is the CPU's, in the series' Float64. A
  // fill's own saves are made inside the fallback, and a save, a load or a bind outside it: all must name that precision,
  // whichever the GPU request names (an explicit FP32; auto, which CUDA leaves open and Metal makes FP32).
  for (const auto precision : { dtwc::GpuPrecision::FP32, dtwc::GpuPrecision::Auto }) {
    CAPTURE(static_cast<int>(precision));
    const dtwc::test_support::ScratchDirectory scratch("fallback_checkpoint");
    const auto problem = [precision](Device device, bool fallback) {
      dtwc::Problem prob("fallback"); // one name: one checkpoint file
      prob.set_data(levels());
      prob.set_variant(dtwc::core::DTWVariant::WDTW);
      prob.set_gpu_precision(precision);
      prob.set_device(device, 0, fallback);
      return prob;
    };

    auto filled = problem(Device::GPU, true);
    filled.checkpoint.enabled = true;
    filled.checkpoint.directory = (scratch.path / "autosave").string();
    filled.checkpoint.save_interval = 2;
    {
      const CerrCapture cerr;
      filled.fill_distance_matrix();
      check_says_fallback(cerr.text());
    }
    REQUIRE(filled.is_distance_matrix_filled());
    dtwc::save_checkpoint(filled, (scratch.path / "saved").string());

    // Each file is read by the same request, which falls back as it did, and by the CPU Problem whose pairs these are.
    for (const char *file : { "autosave", "saved" }) {
      for (const bool gpu_request : { true, false }) {
        CAPTURE(file, gpu_request);
        auto reader = problem(gpu_request ? Device::GPU : Device::CPU, gpu_request);
        REQUIRE(dtwc::load_checkpoint(reader, (scratch.path / file).string()));
        REQUIRE(reader.is_distance_matrix_filled());
        CHECK(reader.dist_by_ind(0, 5) == filled.dist_by_ind(0, 5));
        CHECK(reader.distance_checkpoint_identity() == filled.distance_checkpoint_identity());
      }
    }
  }
}

#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
TEST_CASE("fallback: a request a GPU honours is identified as the GPU, and one it cannot as the CPU",
          "[device][fallback][checkpoint]")
{
  // Standard DTW on Float64 series, in FP32: the kernels take it, where a device is here, and no pair is computed.
  const bool device_here = dtwc::gpu_available();
  const auto identity = [](Device device, bool fallback) {
    dtwc::Problem prob("fallback");
    prob.set_data(levels());
    prob.set_gpu_precision(dtwc::GpuPrecision::FP32);
    prob.set_device(device, 0, fallback);
    return prob.distance_checkpoint_identity();
  };
  const auto gpu = identity(Device::GPU, false); // the GPU's FP32
  const auto cpu = identity(Device::CPU, false); // the series' Float64
  REQUIRE(gpu != cpu);
  CHECK(identity(Device::GPU, true) == (device_here ? gpu : cpu));
}
#endif
