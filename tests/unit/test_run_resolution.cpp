/**
 * @file test_run_resolution.cpp
 * @brief Every cell of dtwc::run's method x device table (IF-2 S3), in process.
 *
 * @details In-memory series, so nothing is read or written. On `cpu` every
 * method runs and `auto` is pam up to N = 5000, clara above. On `gpu` the matrix
 * methods run with the GPU filling the matrix (on Metal every distance is then
 * FP32-exact, which the CPU's are not: proof the GPU ran), `auto` is pam at any
 * N, and the methods that compute on the CPU as they go raise DeviceError; a
 * build without a GPU raises §6.1's message in every gpu cell, a GPU build
 * without a device the backend's. `hpc` raises DeviceError for every method.
 * The GPU rules that need no series are raised before one is read: the input
 * named does not exist. A CUDA build runs the CUDA branch blind here (V-row).
 * A build without HiGHS raises SolverError for `mip`, the one method that needs
 * a solver; `lrcore` needs none and runs there like every other method.
 *
 * @date 24 Sep 2026
 */

#include <dtwc.hpp>
#include <cli/run.hpp>
#include <metal/metal_dtw.hpp>
#include <mip/mip.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::StartsWith;
using dtwc::ClusterMethod;
using dtwc::Device;

namespace {

constexpr ClusterMethod kAll[]{ ClusterMethod::Auto,     ClusterMethod::PAM,   ClusterMethod::OneBatch,
                                ClusterMethod::CLARA,    ClusterMethod::Kmedoids, ClusterMethod::MIP,
                                ClusterMethod::LRCore,   ClusterMethod::TADPole,  ClusterMethod::Hierarchical };
/// The methods whose distance matrix a GPU fills (clara: its sample covers every series here).
constexpr ClusterMethod kMatrix[]{ ClusterMethod::Auto,     ClusterMethod::PAM, ClusterMethod::CLARA,
                                   ClusterMethod::Kmedoids, ClusterMethod::MIP, ClusterMethod::LRCore,
                                   ClusterMethod::Hierarchical };

const std::string kCpuAsItGoes = "computes its distances on the CPU as it goes, so device 'gpu' would sit idle";

/// Two groups of three constant series; each group's middle series is its
/// medoid by a margin no FP32 rounding can flip, and 5 * 0.2 is not FP32-exact.
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

dtwc::Config config_for(ClusterMethod method, Device device)
{
  dtwc::Config config;
  config.k = 2;
  config.method = method;
  config.device = device;
  config.output.clear(); // write nothing
  config.tadpole_dc = 3.0; // TADPole's auto cutoff splits these six series 1 + 5
  return config;
}

std::string name(ClusterMethod method) { return std::string(dtwc::name_of(dtwc::cluster_method_names, method)); }

ClusterMethod resolved(ClusterMethod method) { return method == ClusterMethod::Auto ? ClusterMethod::PAM : method; }

bool two_groups(const dtwc::Result &result)
{
  const auto &l = result.labels();
  return l.size() == 6 && l[0] == l[1] && l[1] == l[2] && l[3] == l[4] && l[4] == l[5] && l[0] != l[3];
}

bool float_exact(const std::vector<double> &matrix)
{
  return std::all_of(matrix.begin(), matrix.end(),
                     [](double v) { return static_cast<double>(static_cast<float>(v)) == v; });
}

/// Without HiGHS, `mip` (whose default backend it is) must raise SolverError, not
/// solve some other way. True once that refusal has been asserted, so the caller
/// skips the checks that need a result; false on a build with HiGHS.
bool mip_refused_without_highs(ClusterMethod method)
{
  if (method != ClusterMethod::MIP || dtwc::highs_solver_available()) return false;
  CHECK_THROWS_MATCHES(dtwc::run(config_for(method, Device::CPU), levels()), dtwc::SolverError,
                       MessageMatches(ContainsSubstring("HiGHS solver is unavailable")));
  return true;
}

template <class F>
std::string device_error(F &&f)
{
  try {
    f();
  } catch (const dtwc::DeviceError &e) {
    return e.what();
  }
  return "(no DeviceError)";
}

[[maybe_unused]] bool gpu_present()
{
#if defined(DTWC_HAS_CUDA)
  return dtwc::cuda::cuda_available();
#elif defined(DTWC_HAS_METAL)
  return dtwc::metal::metal_available();
#else
  return false;
#endif
}

} // namespace

TEST_CASE("run on cpu: every method runs; auto is pam up to 5000 series, clara above", "[run][device][cpu]")
{
  for (const auto method : kAll) {
    CAPTURE(name(method));
    if (mip_refused_without_highs(method)) continue;
    const auto result = dtwc::run(config_for(method, Device::CPU), levels());
    CHECK(result.method() == resolved(method));
    CHECK(result.device() == "cpu");
    CHECK(two_groups(result));
    CHECK_FALSE(float_exact(result.distance_matrix())); // FP64 distances
  }
  // The boundary itself is plan_parquet_load's (unit_test_cli_args), the same
  // resolution; here run() applies it to the series it loaded.
  std::vector<std::vector<double>> many;
  for (int i = 0; i < 5001; ++i) many.push_back({ double(i % 97), double(i % 89) });
  CHECK(dtwc::run(config_for(ClusterMethod::Auto, Device::CPU), dtwc::Data(std::move(many), std::vector<std::string>(5001)))
          .method()
        == ClusterMethod::CLARA);
  // A sample smaller than N is CLARA proper, still on the CPU.
  auto partial = config_for(ClusterMethod::CLARA, Device::CPU);
  partial.sample_size = 3;
  CHECK(two_groups(dtwc::run(partial, levels())));
}

TEST_CASE("run on cpu: squared Euclidean distances are computed, not refused", "[run][device][cpu][metric]")
{
  // dtwc_cl refused any metric but l1 on cpu (validate_metric_for_device); the
  // Problem's fill now takes it (IF-2 S2). Oracle: the checked free function.
  auto config = config_for(ClusterMethod::PAM, Device::CPU);
  config.metric = dtwc::core::MetricType::SquaredL2;
  const auto data = levels();
  const auto matrix = dtwc::run(config, levels()).distance_matrix();
  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = 0; j < 6; ++j)
      // A different call path to the same recurrence: under MSVC /fp:contract
      // the last bit may differ (observed: 1 ulp), so 1e-14 relative.
      CHECK_THAT(matrix[i * 6 + j],
                 Catch::Matchers::WithinRel(
                   dtwc::distance::dtw<double>(data.p_vec[i], data.p_vec[j], -1,
                                               dtwc::core::MetricType::SquaredL2),
                   1e-14));
}

TEST_CASE("run on gpu: the matrix methods fill on the GPU; the as-it-goes methods raise", "[run][device][gpu]")
{
  auto partial = config_for(ClusterMethod::CLARA, Device::GPU);
  partial.sample_size = 3; // < N = 6: the samples are views the GPU cannot upload
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  for (const auto method : { ClusterMethod::OneBatch, ClusterMethod::TADPole }) {
    CAPTURE(name(method));
    CHECK_THAT(device_error([&] { (void)dtwc::run(config_for(method, Device::GPU), levels()); }),
               StartsWith("run: method '" + name(method) + "' ") && ContainsSubstring(kCpuAsItGoes));
  }
  CHECK_THAT(device_error([&] { (void)dtwc::run(partial, levels()); }),
             StartsWith("run: method 'clara' ") && ContainsSubstring(kCpuAsItGoes));

  const auto cpu = dtwc::run(config_for(ClusterMethod::PAM, Device::CPU), levels());
  for (const auto method : kMatrix) {
    CAPTURE(name(method));
    if (!gpu_present()) { // the backend's own refusal; nothing ran on the CPU instead
      CHECK_THAT(device_error([&] { (void)dtwc::run(config_for(method, Device::GPU), levels()); }),
                 ContainsSubstring("GPU was detected") && ContainsSubstring("No CPU fallback was attempted"));
      continue;
    }
    const auto result = dtwc::run(config_for(method, Device::GPU), levels());
    CHECK(result.method() == resolved(method));
    CHECK(result.device() == "gpu");
    CHECK(two_groups(result));
    std::vector<int> medoids = result.medoids();
    std::vector<int> cpu_medoids = cpu.medoids();
    std::sort(medoids.begin(), medoids.end());
    std::sort(cpu_medoids.begin(), cpu_medoids.end());
    CHECK(medoids == cpu_medoids);
#  if defined(DTWC_HAS_METAL)
    CHECK(float_exact(result.distance_matrix())); // Metal computes in FP32: it ran
#  endif
  }
  if (gpu_present()) { // auto stays pam above N = 5000 (it failed there before)
    std::vector<std::vector<double>> many;
    for (int i = 0; i < 5001; ++i) many.push_back({ double(i % 97), double(i % 89) });
    auto config = config_for(ClusterMethod::Auto, Device::GPU);
    config.max_iter = 1;
    CHECK(dtwc::run(config, dtwc::Data(std::move(many), std::vector<std::string>(5001))).method()
          == ClusterMethod::PAM);
  }
#else
  // No GPU backend: every gpu cell is §6.1's refusal, before any method rule.
  for (const auto method : kAll) {
    CAPTURE(name(method));
    CHECK(device_error([&] { (void)dtwc::run(config_for(method, Device::GPU), levels()); })
          == dtwc::detail::gpu_not_built_message());
  }
  CHECK(device_error([&] { (void)dtwc::run(partial, levels()); }) == dtwc::detail::gpu_not_built_message());
#endif
}

TEST_CASE("--device hpc is a DeviceError naming the Python and SLURM routes", "[run][device][hpc]")
{
  for (const char *name : { "hpc", "hpc:gpu" }) {
    CAPTURE(name);
    CHECK_THAT(device_error([&] { (void)dtwc::parse_config({ { "device", name } }); }),
               ContainsSubstring("dtwcpp.device(\"hpc\")") && ContainsSubstring("slurm_remote.sh"));
  }
}

TEST_CASE("run on gpu: a request the GPU kernels cannot honour raises before any series is read",
          "[run][device][gpu][fx1]")
{
  const auto missing = (std::filesystem::temp_directory_path() / "dtwc_run_never_created.csv").string();
  REQUIRE_FALSE(std::filesystem::exists(missing));
  const auto from_file = [&](Device device, auto &&set) {
    auto config = config_for(ClusterMethod::PAM, device);
    config.input = missing;
    set(config);
    return config;
  };
  const auto wdtw = [](dtwc::Config &c) { c.variant.variant = dtwc::core::DTWVariant::WDTW; };
  const auto zero_cost = [](dtwc::Config &c) { c.missing = dtwc::core::MissingStrategy::ZeroCost; };
  const auto float32 = [](dtwc::Config &c) { c.dtype = dtwc::core::Precision::Float32; };
  const auto index_1 = [](dtwc::Config &c) { c.gpu.device_id = 1; };
  const auto fp64 = [](dtwc::Config &c) { c.gpu.precision = dtwc::GpuPrecision::FP64; };

  // The CPU takes each of these, so it goes on to read the file, which fails.
  for (const auto &set : { +wdtw, +zero_cost, +float32 })
    CHECK_THROWS_MATCHES(dtwc::run(from_file(Device::CPU, set)), dtwc::IOError,
                         MessageMatches(StartsWith("load: failed to read '" + missing + "'")));
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
#  if defined(DTWC_HAS_CUDA)
  const std::string backend = "run: CUDA ";
#  else
  const std::string backend = "run: Metal ";
#  endif
  const auto rejects = [&](auto &&set, const std::string &axis) {
    CHECK_THAT(device_error([&] { (void)dtwc::run(from_file(Device::GPU, set)); }),
               StartsWith(backend) && ContainsSubstring(axis) && ContainsSubstring("no backend call or CPU fallback"));
  };
  rejects(wdtw, "variant = WDTW");
  rejects(zero_cost, "missing_strategy = ZeroCost");
  rejects(float32, "precision = Float32");
#  if defined(DTWC_HAS_METAL) && !defined(DTWC_HAS_CUDA)
  // Metal runs on the system default GPU in FP32; CUDA takes an index and FP64.
  rejects(index_1, "GPU index = 1");
  CHECK_THAT(device_error([&] { (void)dtwc::run(from_file(Device::GPU, fp64)); }),
             ContainsSubstring("precision FP64 is not implemented"));
#  else
  for (const auto &set : { +index_1, +fp64 }) // blind: V-row
    CHECK_THROWS_AS(dtwc::run(from_file(Device::GPU, set)), dtwc::IOError);
#  endif
#else
  for (const auto &set : { +wdtw, +zero_cost, +float32, +index_1, +fp64 })
    CHECK(device_error([&] { (void)dtwc::run(from_file(Device::GPU, set)); })
          == dtwc::detail::gpu_not_built_message());
#endif
}

TEST_CASE("run with in-memory series refuses the options only a file reader applies", "[run][input]")
{
  const auto refuses = [](auto &&set, const std::string &what) {
    auto config = config_for(ClusterMethod::PAM, Device::CPU);
    set(config);
    CHECK_THROWS_MATCHES(dtwc::run(config, levels()), dtwc::InvalidInput, MessageMatches(ContainsSubstring(what)));
  };
  refuses([](dtwc::Config &c) { c.input = "series.csv"; }, "the series are passed in memory");
  refuses([](dtwc::Config &c) { c.column = "values"; }, "--column selects a Parquet column");
  refuses([](dtwc::Config &c) { c.skip_rows = 1; }, "--skip-rows, --skip-cols and --delimiter");
  refuses([](dtwc::Config &c) { c.skip_cols = 1; }, "--skip-rows, --skip-cols and --delimiter");
  refuses([](dtwc::Config &c) { c.delimiter = ';'; }, "--skip-rows, --skip-cols and --delimiter");
  refuses([](dtwc::Config &c) { c.ram_limit = 1024; }, "--ram-limit caps Parquet series materialisation");
  refuses([](dtwc::Config &c) { c.k = 7; }, "cluster: k must not exceed the number of series.");
}

TEST_CASE("Result reports the method, iterations and convergence the run had", "[run][result]")
{
  // Oracle: FastPAM called directly on the same series and seed.
  dtwc::Problem problem("oracle");
  problem.set_data(levels());
  const auto direct = dtwc::fast_pam_seeded(problem, 2, 42, 100);
  const auto pam = dtwc::run(config_for(ClusterMethod::PAM, Device::CPU), levels());
  CHECK(pam.iterations() == direct.iterations);
  CHECK(pam.converged() == direct.converged);
  CHECK(pam.cost() == direct.total_cost);

  // Lloyd reports its own count; the exact methods and TADPole finish.
  const auto lloyd = dtwc::run(config_for(ClusterMethod::Kmedoids, Device::CPU), levels());
  CHECK(lloyd.iterations() >= 1);
  CHECK(lloyd.converged() == (lloyd.iterations() < 100));
  for (const auto method : { ClusterMethod::MIP, ClusterMethod::LRCore, ClusterMethod::TADPole }) {
    CAPTURE(name(method));
    if (mip_refused_without_highs(method)) continue;
    const auto result = dtwc::run(config_for(method, Device::CPU), levels());
    CHECK(result.converged());
    CHECK(result.iterations() == 0);
  }
}
