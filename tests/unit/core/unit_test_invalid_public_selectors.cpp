/**
 * @file unit_test_invalid_public_selectors.cpp
 * @brief Task 8.2-F1 preregistration: clustering selectors reject before effects.
 *
 * Public caller-controlled enum inventory under dtwc headers:
 * - F1: Method, Solver, algorithms::Linkage, algorithms::BarycenterMethod, and
 *   Device.
 * - Already exhaustively pinned by M47: core::ConstraintType, MetricType,
 *   DTWVariant, MVMode, MissingStrategy, DistanceMatrixStrategy,
 *   core::Precision, KernelOverride,
 *   CUDASettings::precision, cuda::CUDAPrecision, and metal::MetalPrecision.
 *
 * Derived/internal enum inventory (not caller selectors):
 * cuda::detail::KernelPath, and cuda::FP64Rate.
 * Their values are produced by validated policy/hardware paths rather than
 * accepted at a public operation boundary.
 *
 * Production validation is deliberately not part of this commit. Each invalid
 * domain is probed at -1, first-above, INT_MIN, and INT_MAX. Orthogonal valid
 * controls execute every declared F1 value.
 */

#include <dtwc.hpp>

#include <algorithms/barycenter.hpp>
#include <algorithms/hierarchical.hpp>
#include <base/env.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace dtwc;

constexpr std::string_view method_error = "Invalid Method value.";
constexpr std::string_view solver_error = "Invalid Solver value.";
constexpr std::string_view linkage_error = "Invalid Linkage value.";
constexpr std::string_view barycenter_method_error =
  "Invalid BarycenterMethod value.";
constexpr std::string_view device_error = "Invalid Device value.";

template <typename Enum, Enum Last, typename Function>
void for_each_invalid_enum(Function &&function)
{
  static_assert(std::is_same_v<std::underlying_type_t<Enum>, int>);
  constexpr std::array<int, 4> invalid_values{
    -1, static_cast<int>(Last) + 1, INT_MIN, INT_MAX
  };
  for (const int raw : invalid_values) {
    CAPTURE(raw);
    std::forward<Function>(function)(static_cast<Enum>(raw));
  }
}

template <typename Function>
void expect_invalid_input(std::string_view expected, Function &&function)
{
  bool caught = false;
  try {
    std::forward<Function>(function)();
  } catch (const InvalidInput &error) {
    caught = true;
    CHECK(std::string_view(error.what()) == expected);
  } catch (const std::exception &error) {
    FAIL_CHECK("wrong exception type: " << error.what());
  } catch (...) {
    FAIL_CHECK("wrong non-standard exception type");
  }
  CHECK(caught);
}

Data scalar_data(std::size_t n = 4, bool nonfinite = false)
{
  std::vector<std::vector<data_t>> series;
  std::vector<std::string> names;
  series.reserve(n);
  names.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double value = nonfinite
      ? std::numeric_limits<double>::quiet_NaN()
      : static_cast<double>(i * i + i);
    series.push_back({value});
    names.push_back("s" + std::to_string(i));
  }
  return Data(std::move(series), std::move(names));
}

Problem make_problem(std::size_t n = 4, bool nonfinite = false)
{
  Problem problem("f1_selector_fixture");
  problem.set_verbose(false);
  problem.set_data(scalar_data(n, nonfinite));
  const int k = n == 1 ? 1 : 2;
  problem.set_n_clusters(k);
  problem.centroids_ind = n == 1 ? std::vector<int>{0}
                                 : std::vector<int>{1, 3};
  problem.clusters_ind.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    problem.clusters_ind[i] = static_cast<int>(i % static_cast<std::size_t>(k));
  return problem;
}

struct ProblemSnapshot
{
  Method method;
  int n_clusters;
  std::vector<int> medoids;
  std::vector<int> labels;
  std::vector<std::vector<std::uint64_t>> series_bits;
  std::vector<std::string> names;
  bool matrix_mapped;
  std::size_t matrix_size;
  std::vector<std::uint64_t> packed_bits;
};

std::vector<std::vector<std::uint64_t>> exact_series_bits(const Problem &problem)
{
  std::vector<std::vector<std::uint64_t>> result;
  result.reserve(problem.data().p_vec.size());
  for (const auto &series : problem.data().p_vec) {
    std::vector<std::uint64_t> bits;
    bits.reserve(series.size());
    for (const double value : series)
      bits.push_back(std::bit_cast<std::uint64_t>(value));
    result.push_back(std::move(bits));
  }
  return result;
}

ProblemSnapshot snapshot(const Problem &problem)
{
  const auto &matrix = problem.distance_matrix();
  std::vector<std::uint64_t> bits;
  bits.reserve(matrix.packed_count());
  for (std::size_t i = 0; i < matrix.packed_count(); ++i)
    bits.push_back(std::bit_cast<std::uint64_t>(matrix.raw()[i]));
  return {
    problem.method(),
    problem.n_clusters(),
    problem.centroids_ind,
    problem.clusters_ind,
    exact_series_bits(problem),
    problem.data().p_names,
    problem.distance_matrix().is_mapped(),
    matrix.size(),
    std::move(bits)
  };
}

void check_snapshot(const Problem &problem, const ProblemSnapshot &before)
{
  CHECK(problem.method() == before.method);
  CHECK(problem.n_clusters() == before.n_clusters);
  CHECK(problem.centroids_ind == before.medoids);
  CHECK(problem.clusters_ind == before.labels);
  CHECK(exact_series_bits(problem) == before.series_bits);
  CHECK(problem.data().p_names == before.names);
  CHECK(problem.distance_matrix().is_mapped() == before.matrix_mapped);
  const auto &matrix = problem.distance_matrix();
  CHECK(matrix.size() == before.matrix_size);
  std::vector<std::uint64_t> after;
  after.reserve(matrix.packed_count());
  for (std::size_t i = 0; i < matrix.packed_count(); ++i)
    after.push_back(std::bit_cast<std::uint64_t>(matrix.raw()[i]));
  CHECK(after == before.packed_bits);
}

} // namespace

TEST_CASE("F1 invalid Method rejects before clustering or publication",
          "[f1][enum][invalid][method]")
{
  for_each_invalid_enum<Method, Method::TADPole>([](Method invalid) {
    {
      auto problem = make_problem();
      const auto before = snapshot(problem);
      expect_invalid_input(method_error, [&] {
        problem.set_method(invalid);
        problem.cluster();
      });
      check_snapshot(problem, before);
    }

  });
}

TEST_CASE("F1 invalid Solver rejects before any backend call",
          "[f1][enum][invalid][solver]")
{
  for_each_invalid_enum<Solver, Solver::HiGHS>([](Solver invalid) {
    auto problem = make_problem();
    problem.mip_settings.warm_start = false;
    REQUIRE_NOTHROW(problem.set_solver(Solver::HiGHS));
    const auto before = snapshot(problem);
    expect_invalid_input(solver_error, [&] {
      (void)problem.set_solver(invalid);
      problem.cluster_by_mip();
    });
    check_snapshot(problem, before);
  });
}

TEST_CASE("F1 invalid Linkage dominates matrix checks and degenerate loops",
          "[f1][enum][invalid][linkage]")
{
  using algorithms::Linkage;
  for_each_invalid_enum<Linkage, Linkage::Average>([](Linkage invalid) {
    {
      auto problem = make_problem();
      const auto before = snapshot(problem);
      algorithms::HierarchicalOptions options;
      options.linkage = invalid;
      expect_invalid_input(linkage_error, [&] {
        (void)algorithms::build_dendrogram(problem, options);
      });
      check_snapshot(problem, before);
    }

    {
      auto problem = make_problem(1);
      problem.fill_distance_matrix();
      const auto before = snapshot(problem);
      algorithms::HierarchicalOptions options;
      options.linkage = invalid;
      expect_invalid_input(linkage_error, [&] {
        (void)algorithms::build_dendrogram(problem, options);
      });
      check_snapshot(problem, before);
    }
  });
}

TEST_CASE("F1 invalid BarycenterMethod rejects before reading series",
          "[f1][enum][invalid][barycenter]")
{
  using algorithms::BarycenterMethod;
  for_each_invalid_enum<BarycenterMethod, BarycenterMethod::SoftDTW>(
    [](BarycenterMethod invalid) {
      auto problem = make_problem(4, true);
      const auto before = snapshot(problem);

      algorithms::BarycenterOptions barycenter;
      barycenter.method = invalid;
      barycenter.max_iter = 1;
      expect_invalid_input(barycenter_method_error, [&] {
        (void)algorithms::dtw_barycenter(problem, {0, 1}, 1, barycenter);
      });
      check_snapshot(problem, before);

      algorithms::BarycenterClusteringOptions clustering;
      clustering.n_clusters = 2;
      clustering.max_iter = 1;
      clustering.barycenter_max_iter = 1;
      clustering.method = invalid;
      expect_invalid_input(barycenter_method_error, [&] {
        (void)algorithms::barycenter_kmeans(problem, clustering);
      });
      check_snapshot(problem, before);
    });
}

TEST_CASE("F1 invalid Device never aliases CPU in public reporting",
          "[f1][enum][invalid][device]")
{
  for_each_invalid_enum<Device, Device::GPU>([](Device invalid) {
    expect_invalid_input(device_error, [&] { (void)to_string(invalid); });
  });
}

TEST_CASE("F1 all declared Method values remain accepted",
          "[f1][enum][valid][method]")
{
  constexpr std::array values{
    Method::Kmedoids, Method::MIP, Method::LRCore, Method::TADPole
  };
  for (const Method value : values) {
    auto problem = make_problem();
    CHECK_NOTHROW(problem.set_method(value));
    CHECK(problem.method() == value);
  }
}

TEST_CASE("F1 all declared Solver values remain accepted",
          "[f1][enum][valid][solver]")
{
  constexpr std::array values{Solver::Gurobi, Solver::HiGHS};
  for (const Solver value : values) {
    auto problem = make_problem();
    CHECK_NOTHROW((void)problem.set_solver(value));
  }
}

TEST_CASE("F1 all declared Linkage values dispatch",
          "[f1][enum][valid][linkage]")
{
  using algorithms::Linkage;
  constexpr std::array values{Linkage::Single, Linkage::Complete,
                              Linkage::Average};
  for (const Linkage value : values) {
    auto problem = make_problem();
    problem.fill_distance_matrix();
    algorithms::HierarchicalOptions options;
    options.linkage = value;
    const auto dendrogram = algorithms::build_dendrogram(problem, options);
    CHECK(dendrogram.n_points == 4);
    CHECK(dendrogram.merges.size() == 3);
  }
}

TEST_CASE("F1 all declared BarycenterMethod values dispatch at both entry points",
          "[f1][enum][valid][barycenter]")
{
  using algorithms::BarycenterMethod;
  constexpr std::array values{
    BarycenterMethod::SSG,
    BarycenterMethod::DBA,
    BarycenterMethod::SoftDTW
  };
  for (const BarycenterMethod value : values) {
    auto problem = make_problem();
    algorithms::BarycenterOptions barycenter;
    barycenter.method = value;
    barycenter.max_iter = 1;
    const auto center = algorithms::dtw_barycenter(
      problem, {0, 1, 2, 3}, 1, barycenter);
    REQUIRE(center.size() == 1);
    CHECK(std::isfinite(center[0]));

    algorithms::BarycenterClusteringOptions clustering;
    clustering.n_clusters = 2;
    clustering.max_iter = 1;
    clustering.barycenter_max_iter = 1;
    clustering.method = value;
    const auto result = algorithms::barycenter_kmeans(problem, clustering);
    CHECK(result.labels.size() == 4);
    CHECK(result.barycenters.size() == 2);
    CHECK(std::isfinite(result.total_cost));
  }
}

TEST_CASE("F1 all declared Device values report their exact names",
          "[f1][enum][valid][device]")
{
  CHECK(to_string(Device::CPU) == "cpu");
  CHECK(to_string(Device::GPU) == "gpu");
}
