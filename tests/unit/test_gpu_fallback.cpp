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

/// The one line a fallback writes on stderr.
void check_says_fallback(const std::string &written)
{
  CHECK_THAT(written, StartsWith("[dtwc] gpu requested: "));
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
  }

  SECTION("a forced share: refused on a device that would ignore it, dropped where cpu+gpu's split has no CPU twin")
  {
    auto forced = gpu_request(Method::PAM, true);
    forced.gpu_share = 0.5;
    const CerrCapture cerr;
    CHECK_THROWS_AS(dtwc::run(forced, levels()), dtwc::InvalidInput); // on every build, fallback or not
    forced.device = Device::CPU_GPU;
    CHECK(dtwc::run(forced, levels()).device() == "cpu");
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
  SECTION("a request the GPU can honour is no fallback: it computes there, and says nothing")
  {
    if (!dtwc::gpu_available()) SKIP("no GPU device");
    auto config = gpu_request(Method::PAM, true);
    config.variant.variant = dtwc::core::DTWVariant::Standard;
    const CerrCapture cerr;
    CHECK(dtwc::run(config, levels()).device() == "gpu");
    dtwc::Problem prob("gpu");
    prob.set_data(levels());
    prob.set_device(Device::GPU, 0, true);
    prob.fill_distance_matrix();
    CHECK(prob.is_distance_matrix_filled());
    CHECK(cerr.text().empty());
  }
#endif

  SECTION("cdist: the DeviceError without fallback, the CPU's matrix with it")
  {
    const auto data = levels();
    const dtwc::core::DistanceConfig distance{ .variant = { .variant = dtwc::core::DTWVariant::WDTW } };
    CHECK_THROWS_AS(dtwc::cdist(data, data, distance, Device::GPU), dtwc::DeviceError);
    const CerrCapture cerr;
    const auto matrix = dtwc::cdist(data, data, distance, Device::GPU, 0, dtwc::GpuPrecision::Auto, -1.0, true);
    check_says_fallback(cerr.text());
    CHECK(matrix == dtwc::cdist(data, data, distance));
  }
}
