/**
 * @file test_run_resolution.cpp
 * @brief Every cell of dtwc::run's method x device table, in process.
 *
 * @details In-memory series, so nothing is read or written. On `cpu` every
 * method runs and `auto` is pam up to N = 5000, clara above. On `gpu` the matrix
 * methods run with the GPU filling the matrix (on Metal every distance is then
 * FP32-exact, which the CPU's are not: proof the GPU ran), as does clara with a
 * sample smaller than N, `auto` is pam at any N, and the methods that compute
 * on the CPU as they go raise DeviceError; a
 * build without a GPU raises the no-GPU DeviceError message in every gpu cell, a GPU build
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
#include <mip/mip.hpp>

#include "../support/scratch_directory.hpp"

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
using dtwc::Method;
using dtwc::Device;

namespace {

constexpr Method kAll[]{ Method::Auto,   Method::PAM,     Method::OneBatch, Method::CLARA,       Method::Kmedoids,
                         Method::MIP,    Method::LRCore,  Method::TADPole,  Method::Hierarchical };
/// The methods whose distance matrix a GPU fills (clara: its sample covers every series here).
constexpr Method kMatrix[]{ Method::Auto, Method::PAM,    Method::CLARA,       Method::Kmedoids,
                            Method::MIP,  Method::LRCore, Method::Hierarchical };

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

dtwc::Config config_for(Method method, Device device)
{
  dtwc::Config config;
  config.k = 2;
  config.method = method;
  config.device = device;
  config.output.clear(); // write nothing
  config.tadpole_dc = 3.0; // TADPole's auto cutoff splits these six series 1 + 5
  return config;
}

std::string name(Method method) { return std::string(dtwc::name_of(dtwc::method_names, method)); }

Method resolved(Method method) { return method == Method::Auto ? Method::PAM : method; }

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
/// solve some other way, on either device and before a GPU is looked for. True
/// once that refusal has been asserted, so the caller skips the checks that need
/// a result; false on a build with HiGHS.
bool mip_refused_without_highs(Method method, Device device = Device::CPU)
{
  if (method != Method::MIP || dtwc::highs_solver_available()) return false;
  CHECK_THROWS_MATCHES(dtwc::run(config_for(method, device), levels()), dtwc::SolverError,
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
  CHECK(dtwc::run(config_for(Method::Auto, Device::CPU), dtwc::Data(std::move(many), std::vector<std::string>(5001)))
          .method()
        == Method::CLARA);
  // A sample smaller than N is CLARA proper, still on the CPU.
  auto partial = config_for(Method::CLARA, Device::CPU);
  partial.sample_size = 3;
  CHECK(two_groups(dtwc::run(partial, levels())));
}

TEST_CASE("run on cpu: squared Euclidean distances are computed, not refused", "[run][device][cpu][metric]")
{
  // dtwc_cl refused any metric but l1 on cpu (validate_metric_for_device); the
  // Problem's fill now takes it. Oracle: the checked free function.
  auto config = config_for(Method::PAM, Device::CPU);
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

TEST_CASE("run on gpu: the matrix methods and clara run on the GPU; the as-it-goes methods raise",
          "[run][device][gpu]")
{
  auto partial = config_for(Method::CLARA, Device::GPU);
  partial.sample_size = 3; // < N = 6: clara proper
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  for (const auto method : { Method::OneBatch, Method::TADPole }) {
    CAPTURE(name(method));
    CHECK_THAT(device_error([&] { (void)dtwc::run(config_for(method, Device::GPU), levels()); }),
               StartsWith("run: method '" + name(method) + "' ") && ContainsSubstring(kCpuAsItGoes));
  }
  // clara on a sample runs: its sample matrices fill on the GPU, and on CUDA its
  // assignment runs there too. Its medoids need not be the CPU's, since {0.3, 10.1}
  // and {0.5, 10.3} tie at cost 5, which FP32 and FP64 break either way; its cost is the CPU's.
  if (dtwc::gpu_available()) {
    auto on_cpu = partial;
    on_cpu.device = Device::CPU;
    const auto result = dtwc::run(partial, levels());
    CHECK(result.method() == Method::CLARA);
    CHECK(two_groups(result));
    CHECK_THAT(result.cost(), Catch::Matchers::WithinRel(dtwc::run(on_cpu, levels()).cost(), 1e-6));
  } else {
    CHECK_THAT(device_error([&] { (void)dtwc::run(partial, levels()); }),
               ContainsSubstring("GPU was detected") && ContainsSubstring("No CPU fallback was attempted"));
  }

  const auto cpu = dtwc::run(config_for(Method::PAM, Device::CPU), levels());
  for (const auto method : kMatrix) {
    CAPTURE(name(method));
    if (mip_refused_without_highs(method, Device::GPU)) continue;
    if (!dtwc::gpu_available()) { // the backend's own refusal; nothing ran on the CPU instead
      CHECK_THAT(device_error([&] { (void)dtwc::run(config_for(method, Device::GPU), levels()); }),
                 ContainsSubstring("GPU was detected") && ContainsSubstring("No CPU fallback was attempted"));
      continue;
    }
    const auto result = dtwc::run(config_for(method, Device::GPU), levels());
    CHECK(result.method() == resolved(method));
    CHECK(result.device() == "gpu");
    CHECK(two_groups(result));
    std::vector<dtwc::index_t> medoids = result.medoids();
    std::vector<dtwc::index_t> cpu_medoids = cpu.medoids();
    std::sort(medoids.begin(), medoids.end());
    std::sort(cpu_medoids.begin(), cpu_medoids.end());
    CHECK(medoids == cpu_medoids);
#  if defined(DTWC_HAS_METAL)
    CHECK(float_exact(result.distance_matrix())); // Metal computes in FP32: it ran
#  endif
  }
  if (dtwc::gpu_available()) { // auto stays pam above N = 5000 (it failed there before)
    std::vector<std::vector<double>> many;
    for (int i = 0; i < 5001; ++i) many.push_back({ double(i % 97), double(i % 89) });
    auto config = config_for(Method::Auto, Device::GPU);
    config.max_iter = 1;
    CHECK(dtwc::run(config, dtwc::Data(std::move(many), std::vector<std::string>(5001))).method()
          == Method::PAM);
  }
#else
  // No GPU backend: every gpu cell is the no-GPU refusal, before any method rule.
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
  const dtwc::test_support::ScratchDirectory scratch{ "run_never_created" };
  const auto missing = (scratch.path / "never_created.csv").string();
  REQUIRE_FALSE(std::filesystem::exists(missing));
  const auto from_file = [&](Device device, auto &&set) {
    auto config = config_for(Method::PAM, device);
    config.input = missing;
    set(config);
    return config;
  };
  const auto wdtw = [](dtwc::Config &c) { c.variant.variant = dtwc::core::DTWVariant::WDTW; };
  const auto zero_cost = [](dtwc::Config &c) { c.missing = dtwc::core::MissingStrategy::ZeroCost; };
  const auto float32 = [](dtwc::Config &c) { c.dtype = dtwc::core::Precision::Float32; };
  const auto index_1 = [](dtwc::Config &c) { c.device_index = 1; };
  const auto fp64 = [](dtwc::Config &c) { c.gpu_precision = dtwc::GpuPrecision::FP64; };

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
  // The index is refused by set_device, which run calls first.
  CHECK_THAT(device_error([&] { (void)dtwc::run(from_file(Device::GPU, index_1)); }),
             StartsWith("Problem::set_device: Metal ") && ContainsSubstring("GPU index = 1"));
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
    auto config = config_for(Method::PAM, Device::CPU);
    set(config);
    CHECK_THROWS_MATCHES(dtwc::run(config, levels()), dtwc::InvalidInput, MessageMatches(ContainsSubstring(what)));
  };
  refuses([](dtwc::Config &c) { c.input = "series.csv"; }, "the series are passed in memory");
  refuses([](dtwc::Config &c) { c.column = "values"; }, "--column selects a Parquet column");
  refuses([](dtwc::Config &c) { c.skip_rows = 1; }, "--skip-rows and --skip-cols (skip_rows, skip_cols) drop");
  refuses([](dtwc::Config &c) { c.skip_cols = 1; }, "--skip-rows and --skip-cols (skip_rows, skip_cols) drop");
  refuses([](dtwc::Config &c) { c.delimiter = ';'; }, "--delimiter (delimiter) splits CSV/TSV text");
  refuses([](dtwc::Config &c) { c.ram_limit = 1024; }, "--ram-limit caps Parquet series materialisation");
  refuses([](dtwc::Config &c) { c.k = 7; }, "cluster: k must not exceed the number of series.");
  refuses([](dtwc::Config &c) { c.k = dtwc::Config{}.k; }, "-k/--n-clusters, the number of clusters, is required.");
}

TEST_CASE("Problem::cluster() runs each method with its settings, as run() does", "[run][problem]")
{
  // Thirty series with no clear grouping, so each setting below changes the result.
  const auto scattered = [] {
    std::vector<std::vector<double>> series;
    for (int i = 0; i < 30; ++i) {
      series.emplace_back();
      for (int t = 0; t < 6; ++t) series.back().push_back(double((i * 7 + t * 13 + i * t) % 17));
    }
    return dtwc::Data(std::move(series), std::vector<std::string>(30));
  };
  using dtwc::core::ClusteringResult;
  const auto same = [](const ClusteringResult &a, const ClusteringResult &b) {
    return a.labels == b.labels && a.medoid_indices == b.medoid_indices && a.total_cost == b.total_cost
           && a.iterations == b.iterations && a.converged == b.converged;
  };
  // Each row: a setting the method reads, at a value that is not its default, and
  // the method's own function as the oracle.
  struct Row
  {
    Method method;
    void (*set)(dtwc::Config &);
    ClusteringResult (*oracle)(dtwc::Problem &, const dtwc::Config &);
  };
  const Row rows[]{
    { Method::Auto, [](dtwc::Config &) {}, // pam at N = 30
      [](dtwc::Problem &p, const dtwc::Config &c) { return dtwc::fast_pam(p, c.k, c.max_iter, c.seed); } },
    { Method::PAM, [](dtwc::Config &c) { c.seed = 7; },
      [](dtwc::Problem &p, const dtwc::Config &c) { return dtwc::fast_pam(p, c.k, c.max_iter, c.seed); } },
    { Method::OneBatch, [](dtwc::Config &c) { c.batch_size = 3; },
      [](dtwc::Problem &p, const dtwc::Config &c) {
        return dtwc::algorithms::one_batch_pam(p, { .n_clusters = c.k, .batch_size = c.batch_size,
                                                    .max_iter = c.max_iter, .random_seed = c.seed });
      } },
    { Method::CLARA, [](dtwc::Config &c) { c.sample_size = 8; c.n_samples = 2; },
      [](dtwc::Problem &p, const dtwc::Config &c) {
        return dtwc::algorithms::fast_clara(p, { .n_clusters = c.k, .sample_size = c.sample_size,
                                                 .n_samples = c.n_samples, .max_iter = c.max_iter,
                                                 .random_seed = c.seed });
      } },
    { Method::Hierarchical, [](dtwc::Config &c) { c.linkage = dtwc::algorithms::Linkage::Single; },
      [](dtwc::Problem &p, const dtwc::Config &c) {
        return dtwc::algorithms::cut_dendrogram(dtwc::algorithms::build_dendrogram(p, { .linkage = c.linkage }), p,
                                                c.k);
      } },
  };
  for (const Row &row : rows) {
    CAPTURE(name(row.method));
    auto config = config_for(row.method, Device::CPU);
    config.k = 3;
    const auto oracle = [&](const dtwc::Config &c) {
      dtwc::Problem p;
      p.set_data(scattered());
      return row.oracle(p, c);
    };
    const auto by_default = oracle(config);
    row.set(config);
    const auto expected = oracle(config);
    if (row.method != Method::Auto) CHECK_FALSE(same(expected, by_default)); // the setting bites here

    dtwc::Problem prob;
    prob.set_data(scattered());
    prob.set_n_clusters(config.k);
    prob.set_method(row.method);
    prob.set_random_seed(config.seed);
    prob.set_sample_size(config.sample_size);
    prob.set_n_samples(config.n_samples);
    prob.set_batch_size(config.batch_size);
    prob.set_linkage(config.linkage);
    CHECK(same(prob.cluster(), expected));
    CHECK(prob.labels() == expected.labels); // published, as set_result does

    const auto run = dtwc::run(config, scattered());
    CHECK(run.method() == resolved(row.method));
    CHECK(same({ run.labels(), run.medoids(), run.cost(), run.iterations(), run.converged() }, expected));
  }
}

TEST_CASE("Result reports the method, iterations and convergence the run had", "[run][result]")
{
  // Oracle: FastPAM called directly on the same series and seed.
  dtwc::Problem problem("oracle");
  problem.set_data(levels());
  const auto direct = dtwc::fast_pam(problem, 2, 100, 42);
  const auto pam = dtwc::run(config_for(Method::PAM, Device::CPU), levels());
  CHECK(pam.iterations() == direct.iterations);
  CHECK(pam.converged() == direct.converged);
  CHECK(pam.cost() == direct.total_cost);

  // Lloyd reports its own count; the exact methods and TADPole finish.
  const auto lloyd = dtwc::run(config_for(Method::Kmedoids, Device::CPU), levels());
  CHECK(lloyd.iterations() >= 1);
  CHECK(lloyd.converged() == (lloyd.iterations() < 100));
  for (const auto method : { Method::MIP, Method::LRCore, Method::TADPole }) {
    CAPTURE(name(method));
    if (mip_refused_without_highs(method)) continue;
    const auto result = dtwc::run(config_for(method, Device::CPU), levels());
    CHECK(result.converged());
    CHECK(result.iterations() == 0);
  }
}
