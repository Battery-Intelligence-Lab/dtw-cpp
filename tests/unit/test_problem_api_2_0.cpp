/**
 * @file test_problem_api_2_0.cpp
 * @brief Task 1.6 tests: 2.0 Problem/scores rename shims, result write-back,
 *        and the variant_params -> dtw_fn_ rebind invariant.
 *
 * @details Pins the LIVE public entry points introduced/changed by Task 1.6 of
 *          the DTWC++ 2.0 refactor (docs/api-contract-2.0.md, FROZEN):
 *            1. Deprecated 1.x camelCase shims forward to the canonical
 *               snake_case names (compile with a deprecation warning, which we
 *               suppress locally; assert identical behaviour).
 *            2. scores::silhouette(prob) works in pure C++ immediately after
 *               fast_pam(prob, k) with NO manual wiring — the result write-back
 *               moved into core (was binding-only in 1.x).
 *            3. set_variant(...) rebinds the bound DTW function dtw_fn_ — a
 *               direct behavioural check: a known distance changes from the
 *               registered Standard value to the registered ADTW value.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <dtwc.hpp>
#include <mip/mip.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef DTWC_F22_TEST_ROOT
#  error "DTWC_F22_TEST_ROOT must name the absolute build-local fixture root"
#endif

// Local, cross-compiler suppression of -Wdeprecated-declarations so this test
// can call the deprecated 1.x shims on purpose without failing a -Werror build.
#if defined(__clang__)
#  define DTWC_PUSH_NO_DEPRECATED _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#  define DTWC_POP_NO_DEPRECATED  _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#  define DTWC_PUSH_NO_DEPRECATED _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#  define DTWC_POP_NO_DEPRECATED  _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
#  define DTWC_PUSH_NO_DEPRECATED __pragma(warning(push)) __pragma(warning(disable : 4996))
#  define DTWC_POP_NO_DEPRECATED  __pragma(warning(pop))
#else
#  define DTWC_PUSH_NO_DEPRECATED
#  define DTWC_POP_NO_DEPRECATED
#endif

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: two clearly separated groups of 3 flat-ish series each (length 3).
// Group L ~ 0, Group H ~ 100 -> fast_pam with k=2 recovers the two groups and
// intra-cluster DTW distances are ~0 while inter-cluster ones are ~200-300.
// ---------------------------------------------------------------------------
static Problem make_two_group_problem()
{
  std::vector<std::vector<data_t>> vecs = {
    { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0 },       // group L (near 0)
    { 100.0, 100.0, 100.0 }, { 100.0, 100.0, 101.0 }, { 100.0, 101.0, 100.0 }, // group H (near 100)
  };
  std::vector<std::string> names = { "L0", "L1", "L2", "H0", "H1", "H2" };
  Data data(std::move(vecs), std::move(names));
  Problem prob("api_2_0_two_group");
  prob.set_data(std::move(data));
  return prob;
}

namespace {

constexpr std::size_t f22_point_count = 6;
using f22_matrix_t =
  std::array<std::array<double, f22_point_count>, f22_point_count>;

constexpr f22_matrix_t f22_distance_oracle{ {
  { { 0.0, 2.0, 9.0, 40.0, 55.0, 100.0 } },
  { { 2.0, 0.0, 7.0, 38.0, 53.0, 98.0 } },
  { { 9.0, 7.0, 0.0, 31.0, 46.0, 91.0 } },
  { { 40.0, 38.0, 31.0, 0.0, 15.0, 60.0 } },
  { { 55.0, 53.0, 46.0, 15.0, 0.0, 45.0 } },
  { { 100.0, 98.0, 91.0, 60.0, 45.0, 0.0 } },
} };

constexpr f22_matrix_t f22_external_matrix{ {
  { { 0.0, 3.0, 8.0, 15.0, 24.0, 35.0 } },
  { { 3.0, 0.0, 6.0, 13.0, 22.0, 33.0 } },
  { { 8.0, 6.0, 0.0, 9.0, 18.0, 29.0 } },
  { { 15.0, 13.0, 9.0, 0.0, 11.0, 22.0 } },
  { { 24.0, 22.0, 18.0, 11.0, 0.0, 13.0 } },
  { { 35.0, 33.0, 29.0, 22.0, 13.0, 0.0 } },
} };

constexpr std::string_view f22_external_matrix_csv =
  "0,3,8,15,24,35\n"
  "3,0,6,13,22,33\n"
  "8,6,0,9,18,29\n"
  "15,13,9,0,11,22\n"
  "24,22,18,11,0,13\n"
  "35,33,29,22,13,0\n";

constexpr std::string_view f22_distance_stdout =
  "0,2,9,40,55,100\n"
  "2,0,7,38,53,98\n"
  "9,7,0,31,46,91\n"
  "40,38,31,0,15,60\n"
  "55,53,46,15,0,45\n"
  "100,98,91,60,45,0\n";

constexpr std::string_view f22_clusters_stdout =
  "Clusters centroids: s1 s4 \n"
  "The cluster with centroid s1 has following members: s0 s1 s2 \n"
  "The cluster with centroid s4 has following members: s3 s4 s5 \n";

constexpr std::string_view f22_failed_read_message =
  "Cannot open file for reading";

bool f22_same_bits(double lhs, double rhs)
{
  return std::bit_cast<std::uint64_t>(lhs)
      == std::bit_cast<std::uint64_t>(rhs);
}

Problem make_f22_problem(
  const std::filesystem::path &output = {},
  bool clustered = true)
{
  std::vector<std::vector<data_t>> series{
    { 0.0 }, { 2.0 }, { 9.0 }, { 40.0 }, { 55.0 }, { 100.0 }
  };
  std::vector<std::string> names{
    "s0", "s1", "s2", "s3", "s4", "s5"
  };
  Problem problem("f22_asymmetric");
  problem.set_data(Data(std::move(series), std::move(names)));
  problem.set_verbose(false);
  if (!output.empty())
    problem.set_output_folder(output);
  if (clustered) {
    problem.set_n_clusters(2);
    problem.centroids_ind = { 1, 4 };
    problem.clusters_ind = { 0, 0, 0, 1, 1, 1 };
  }
  return problem;
}

bool f22_matrix_matches(Problem &problem, const f22_matrix_t &expected)
{
  for (std::size_t i = 0; i < f22_point_count; ++i)
    for (std::size_t j = 0; j < f22_point_count; ++j)
      if (!f22_same_bits(
            problem.dist_by_ind(static_cast<int>(i), static_cast<int>(j)),
            expected[i][j]))
        return false;
  return true;
}

bool f22_matrices_match(Problem &lhs, Problem &rhs)
{
  for (std::size_t i = 0; i < f22_point_count; ++i)
    for (std::size_t j = 0; j < f22_point_count; ++j)
      if (!f22_same_bits(
            lhs.dist_by_ind(static_cast<int>(i), static_cast<int>(j)),
            rhs.dist_by_ind(static_cast<int>(i), static_cast<int>(j))))
        return false;
  return true;
}

class f22_cout_capture
{
public:
  f22_cout_capture() : previous_(std::cout.rdbuf(output_.rdbuf())) {}
  f22_cout_capture(const f22_cout_capture &) = delete;
  f22_cout_capture &operator=(const f22_cout_capture &) = delete;
  ~f22_cout_capture() { restore(); }

  void restore()
  {
    if (previous_ != nullptr) {
      std::cout.rdbuf(previous_);
      previous_ = nullptr;
    }
  }

  std::string str() const { return output_.str(); }

private:
  std::ostringstream output_;
  std::streambuf *previous_;
};

template <typename Function>
std::string f22_capture_stdout(Function &&function)
{
  f22_cout_capture capture;
  std::forward<Function>(function)();
  capture.restore();
  return capture.str();
}

std::filesystem::path f22_validated_test_root()
{
  const auto root =
    std::filesystem::path{ DTWC_F22_TEST_ROOT }.lexically_normal();
  if (!root.is_absolute()
      || !root.has_parent_path()
      || root.filename() != "f22-cpp-compat")
    throw std::runtime_error(
      "F22 test root must be absolute and end in f22-cpp-compat: "
      + root.string());
  return root;
}

class f22_scoped_test_root
{
public:
  f22_scoped_test_root() : path_(f22_validated_test_root())
  {
    // f22_validated_test_root() establishes the absolute, terminal-leaf
    // contract before this recursive removal is reachable.
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    if (error)
      throw std::runtime_error(
        "F22 failed to clear fixture root: " + path_.string());
    std::filesystem::create_directories(path_, error);
    if (error)
      throw std::runtime_error(
        "F22 failed to create fixture root: " + path_.string());
  }

  f22_scoped_test_root(const f22_scoped_test_root &) = delete;
  f22_scoped_test_root &operator=(const f22_scoped_test_root &) = delete;
  ~f22_scoped_test_root()
  {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path &path() const { return path_; }

  std::filesystem::path make_leaf(std::string_view leaf) const
  {
    const std::filesystem::path leaf_path{ std::string{ leaf } };
    if (leaf_path.empty()
        || leaf_path.has_parent_path()
        || leaf_path == "."
        || leaf_path == "..")
      throw std::runtime_error("F22 invalid fixture leaf: " + leaf_path.string());
    const auto result = (path_ / leaf_path).lexically_normal();
    if (result.parent_path() != path_)
      throw std::runtime_error(
        "F22 fixture leaf escaped configured root: " + result.string());
    std::error_code error;
    std::filesystem::create_directory(result, error);
    if (error)
      throw std::runtime_error(
        "F22 failed to create fixture leaf: " + result.string());
    return result;
  }

private:
  std::filesystem::path path_;
};

bool f22_write_text(
  const std::filesystem::path &path,
  std::string_view contents)
{
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) return false;
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  output.close();
  return output.good();
}

std::optional<std::string> f22_read_bytes(
  const std::filesystem::path &path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) return std::nullopt;
  return std::string{
    std::istreambuf_iterator<char>{ input },
    std::istreambuf_iterator<char>{}
  };
}

std::string f22_normalize_crlf(std::string_view text)
{
  std::string normalized;
  normalized.reserve(text.size());
  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] == '\r'
        && index + 1 < text.size()
        && text[index + 1] == '\n')
      continue;
    normalized.push_back(text[index]);
  }
  return normalized;
}

bool f22_equal_nonempty_files(
  const std::filesystem::path &legacy,
  const std::filesystem::path &canonical)
{
  const auto legacy_bytes = f22_read_bytes(legacy);
  const auto canonical_bytes = f22_read_bytes(canonical);
  return legacy_bytes.has_value()
      && canonical_bytes.has_value()
      && !legacy_bytes->empty()
      && legacy_bytes == canonical_bytes;
}

enum class f22_exception_kind {
  none,
  invalid_input,
  solver_error,
  standard,
  unknown
};

struct f22_exception_outcome {
  f22_exception_kind kind{ f22_exception_kind::none };
  std::string message;
};

template <typename Function>
f22_exception_outcome f22_capture_exception(Function &&function)
{
  try {
    std::forward<Function>(function)();
  } catch (const InvalidInput &error) {
    return { f22_exception_kind::invalid_input, error.what() };
  } catch (const SolverError &error) {
    return { f22_exception_kind::solver_error, error.what() };
  } catch (const std::exception &error) {
    return { f22_exception_kind::standard, error.what() };
  } catch (...) {
    return { f22_exception_kind::unknown, {} };
  }
  return {};
}

enum class f22_entity : std::size_t {
  set_number_of_clusters,
  refresh_distance_matrix,
  read_distance_matrix,
  max_distance,
  dist_by_ind,
  is_distance_matrix_filled,
  fill_distance_matrix,
  print_distance_matrix,
  write_distance_matrix_named,
  write_distance_matrix_default,
  print_clusters,
  write_clusters,
  write_medoid_members,
  write_silhouettes,
  find_total_cost,
  assign_clusters,
  calculate_medoids,
  cluster_by_mip,
  cluster_by_kmedoids_lloyd,
  cluster_size,
  davies_bouldin,
  dunn,
  calinski_harabasz,
  adjusted_rand,
  normalized_mutual_info,
  loader_start_column,
  loader_start_row,
  set_data_path_path,
  set_data_path_cstring,
  set_results_path_path,
  set_results_path_cstring,
  max_iter_field,
  n_repetition_field,
  count
};

constexpr std::size_t f22_entity_count =
  static_cast<std::size_t>(f22_entity::count);
static_assert(f22_entity_count == 33);

enum class f22_field_route : std::size_t {
  max_iter_canonical_write,
  max_iter_legacy_write,
  n_repetition_canonical_write,
  n_repetition_legacy_write,
  count
};

enum class f22_io_route : std::size_t {
  read_distance_matrix,
  write_distance_matrix_named,
  write_distance_matrix_default,
  print_clusters,
  write_clusters,
  write_medoid_members,
  write_silhouettes,
  count
};

enum class f22_file_route : std::size_t {
  read_source,
  write_distance_matrix_named,
  write_distance_matrix_default,
  write_clusters,
  write_medoid_members,
  write_silhouettes,
  count
};

enum class f22_stdout_route : std::size_t {
  print_distance_matrix,
  print_clusters,
  count
};

template <typename Key, std::size_t Count>
class f22_keyed_ledger
{
public:
  void record(Key key, bool passed)
  {
    const auto index = static_cast<std::size_t>(key);
    REQUIRE(index < Count);
    const bool duplicate = seen_.test(index);
    CHECK_FALSE(duplicate);
    if (duplicate) return;
    seen_.set(index);
    passed_.set(index, passed);
    CHECK(passed);
  }

  int seen_count() const { return static_cast<int>(seen_.count()); }
  int passed_count() const { return static_cast<int>(passed_.count()); }
  bool all() const { return seen_.all() && passed_.all(); }

private:
  std::bitset<Count> seen_;
  std::bitset<Count> passed_;
};

class f22_compat_ledger
{
public:
  void behavior(f22_entity entity, bool passed)
  {
    behavior_.record(entity, passed);
  }

  void field_route(f22_field_route route, bool passed)
  {
    fields_.record(route, passed);
  }

  void io_route(f22_io_route route, bool passed)
  {
    io_.record(route, passed);
  }

  void file_identity(f22_file_route route, bool passed)
  {
    files_.record(route, passed);
  }

  void stdout_identity(f22_stdout_route route, bool passed)
  {
    stdout_.record(route, passed);
  }

  int inventory() const { return behavior_.seen_count(); }
  int behavior_count() const { return behavior_.passed_count(); }
  int field_routes() const { return fields_.passed_count(); }
  int io_routes() const { return io_.passed_count(); }
  int file_identity_count() const { return files_.passed_count(); }
  int stdout_identity_count() const { return stdout_.passed_count(); }

  bool all() const
  {
    return behavior_.all()
        && fields_.all()
        && io_.all()
        && files_.all()
        && stdout_.all();
  }

private:
  f22_keyed_ledger<f22_entity, f22_entity_count> behavior_;
  f22_keyed_ledger<
    f22_field_route,
    static_cast<std::size_t>(f22_field_route::count)> fields_;
  f22_keyed_ledger<
    f22_io_route,
    static_cast<std::size_t>(f22_io_route::count)> io_;
  f22_keyed_ledger<
    f22_file_route,
    static_cast<std::size_t>(f22_file_route::count)> files_;
  f22_keyed_ledger<
    f22_stdout_route,
    static_cast<std::size_t>(f22_stdout_route::count)> stdout_;
};

class f22_path_settings_guard
{
public:
  f22_path_settings_guard()
    : data_(settings::paths::data), results_(settings::paths::results)
  {}
  f22_path_settings_guard(const f22_path_settings_guard &) = delete;
  f22_path_settings_guard &operator=(const f22_path_settings_guard &) = delete;
  ~f22_path_settings_guard()
  {
    settings::paths::data = std::move(data_);
    settings::paths::results = std::move(results_);
  }

private:
  settings::fs::path data_;
  settings::fs::path results_;
};

} // namespace

// ===========================================================================
// Test 1: Deprecated 1.x shims forward to the canonical 2.0 names.
// exercises: Problem::{set_numberOfClusters,cluster_size,distByInd,maxDistance}
//            and scores::{daviesBouldinIndex,dunnIndex,adjustedRandIndex}
//            deprecated shims -> their snake_case canonical implementations.
// ===========================================================================
TEST_CASE("Task 1.6: deprecated shims forward to canonical names", "[api_2_0][deprecated]")
{
  SECTION("set_numberOfClusters == set_n_clusters, cluster_size() == n_clusters()")
  {
    Problem p_old = make_two_group_problem();
    Problem p_new = make_two_group_problem();

    DTWC_PUSH_NO_DEPRECATED
    p_old.set_numberOfClusters(2); // deprecated -> forwards to set_n_clusters
    DTWC_POP_NO_DEPRECATED
    p_new.set_n_clusters(2);       // canonical

    DTWC_PUSH_NO_DEPRECATED
    const auto k_old = p_old.cluster_size(); // deprecated -> forwards to n_clusters()
    DTWC_POP_NO_DEPRECATED
    REQUIRE(k_old == p_new.n_clusters());
    REQUIRE(p_new.n_clusters() == 2);
  }

  SECTION("distByInd == dist_by_ind, maxDistance == max_distance")
  {
    Problem prob = make_two_group_problem();
    prob.fill_distance_matrix();

    DTWC_PUSH_NO_DEPRECATED
    const double d_old = prob.distByInd(0, 3); // deprecated -> dist_by_ind
    const double md_old = prob.maxDistance();  // deprecated -> max_distance
    DTWC_POP_NO_DEPRECATED

    REQUIRE_THAT(d_old, WithinAbs(prob.dist_by_ind(0, 3), 1e-12));
    REQUIRE_THAT(md_old, WithinAbs(prob.max_distance(), 1e-12));
  }

  SECTION("scores deprecated aliases forward to canonical")
  {
    Problem prob = make_two_group_problem();
    (void)fast_pam(prob, 2); // write-back populates centroids_ind/clusters_ind

    DTWC_PUSH_NO_DEPRECATED
    const double dbi_old = scores::daviesBouldinIndex(prob); // -> davies_bouldin
    const double dunn_old = scores::dunnIndex(prob);         // -> dunn
    DTWC_POP_NO_DEPRECATED
    REQUIRE_THAT(dbi_old, WithinAbs(scores::davies_bouldin(prob), 1e-12));
    REQUIRE_THAT(dunn_old, WithinAbs(scores::dunn(prob), 1e-12));

    const std::vector<int> a = { 0, 0, 1, 1 };
    const std::vector<int> b = { 1, 1, 0, 0 };
    DTWC_PUSH_NO_DEPRECATED
    const double ari_old = scores::adjustedRandIndex(a, b); // -> adjusted_rand
    DTWC_POP_NO_DEPRECATED
    REQUIRE_THAT(ari_old, WithinAbs(scores::adjusted_rand(a, b), 1e-12));
  }
}

// ===========================================================================
// Test 2: result write-back — silhouette works right after fast_pam, no wiring.
// exercises: dtwc::fast_pam() write-back -> dtwc::scores::silhouette() LIVE path
//            (cluster then score in pure C++; no manual centroids_ind assignment).
// Registered band: two well-separated groups => mean silhouette > 0.9.
// ===========================================================================
TEST_CASE("Task 1.6: silhouette works after fast_pam with no manual wiring", "[api_2_0][write_back][silhouette]")
{
  Problem prob = make_two_group_problem();

  const auto result = fast_pam(prob, 2);

  // Write-back landed the result into prob (was binding-only in 1.x).
  REQUIRE(prob.n_clusters() == 2);
  REQUIRE(prob.centroids_ind == result.medoid_indices);
  REQUIRE(prob.clusters_ind == result.labels);
  REQUIRE(prob.centroids_ind.size() == 2);
  REQUIRE(prob.clusters_ind.size() == 6);

  // LIVE score path: silhouette reads prob state that fast_pam wrote. Before the
  // 2.0 write-back this returned the degenerate all-(-1) fill (centroids empty).
  const auto sil = scores::silhouette(prob);
  REQUIRE(sil.size() == 6);
  const double mean = std::accumulate(sil.begin(), sil.end(), 0.0) / static_cast<double>(sil.size());
  REQUIRE(mean > 0.9); // registered pass band for two clearly separated groups
}

// ===========================================================================
// Test 3: set_variant() rebinds the bound DTW function dtw_fn_ (behavioural).
// exercises: dtwc::Problem::set_variant() -> refresh_distance_matrix() ->
//            rebind_dtw_fn(); prove by computing a known distance before/after.
//
// Registered oracle (hand-derived, L1, full DTW band=-1, x={0,0} vs y={0,1,2}):
//   Standard DTW  = C(1,2) = 3.0
//       C(i,j) = |x_i-y_j| + min(C(i-1,j-1), C(i-1,j), C(i,j-1))
//   ADTW (penalty=1.0) = C(1,2) = 4.0
//       C(i,j) = |x_i-y_j| + min(C(i-1,j-1), C(i-1,j)+p, C(i,j-1)+p)
//   (both values registered here BEFORE the run; see warping_adtw.hpp recurrence.)
// ===========================================================================
TEST_CASE("Task 1.6: set_variant rebinds dtw_fn_ (Standard=3.0 -> ADTW=4.0)", "[api_2_0][rebind][variant]")
{
  std::vector<std::vector<data_t>> vecs = { { 0.0, 0.0 }, { 0.0, 1.0, 2.0 } };
  std::vector<std::string> names = { "x", "y" };
  Data data(std::move(vecs), std::move(names));
  Problem prob("api_2_0_rebind");
  prob.set_data(std::move(data));
  prob.set_band(-1); // full DTW (also the default); matches the hand-derived oracle

  // Bound function is Standard DTW at construction (default variant).
  const double d_std = prob.dtw_function()(prob.series(0), prob.series(1));
  REQUIRE_THAT(d_std, WithinAbs(3.0, 1e-9)); // registered Standard value

  // Writing the variant through the setter MUST rebind dtw_fn_ to ADTW.
  core::DTWVariantParams vp;
  vp.variant = core::DTWVariant::ADTW;
  vp.adtw_penalty = 1.0;
  prob.set_variant(vp);

  const double d_adtw = prob.dtw_function()(prob.series(0), prob.series(1));
  REQUIRE_THAT(d_adtw, WithinAbs(4.0, 1e-9)); // registered ADTW value
  REQUIRE(d_adtw != d_std);                   // the rebind actually changed the function
}

TEST_CASE("F22 asymmetric compatibility oracle is non-degenerate",
          "[api_2_0][f22][oracle]")
{
  Problem problem = make_f22_problem();
  problem.fill_distance_matrix();

  for (std::size_t i = 0; i < f22_point_count; ++i)
    for (std::size_t j = 0; j < f22_point_count; ++j)
      CHECK(f22_same_bits(
        problem.dist_by_ind(static_cast<int>(i), static_cast<int>(j)),
        f22_distance_oracle[i][j]));

  const std::vector<int> expected_labels{ 0, 0, 0, 1, 1, 1 };
  const std::vector<int> expected_medoids{ 1, 4 };
  problem.clusters_ind.assign(f22_point_count, -1);
  problem.assign_clusters();
  CHECK(problem.labels() == expected_labels);

  problem.centroids_ind = { 0, 3 };
  problem.calculate_medoids();
  CHECK(problem.medoids() == expected_medoids);
  CHECK(f22_same_bits(problem.find_total_cost(), 69.0));

  double best_cost = std::numeric_limits<double>::max();
  std::array<int, 2> best_pair{ -1, -1 };
  int best_pair_count = 0;
  std::vector<double> candidate_costs;
  for (std::size_t first = 0; first < f22_point_count; ++first) {
    for (std::size_t second = first + 1; second < f22_point_count; ++second) {
      double cost = 0.0;
      for (std::size_t point = 0; point < f22_point_count; ++point)
        cost += std::min(
          f22_distance_oracle[point][first],
          f22_distance_oracle[point][second]);
      candidate_costs.push_back(cost);
      if (cost < best_cost) {
        best_cost = cost;
        best_pair = {
          static_cast<int>(first), static_cast<int>(second)
        };
        best_pair_count = 1;
      } else if (f22_same_bits(cost, best_cost)) {
        ++best_pair_count;
      }
    }
  }
  std::sort(candidate_costs.begin(), candidate_costs.end());
  CHECK(f22_same_bits(best_cost, 69.0));
  REQUIRE(candidate_costs.size() >= 2);
  CHECK(f22_same_bits(candidate_costs[0], 69.0));
  CHECK(f22_same_bits(candidate_costs[1], 71.0));
  CHECK(best_pair_count == 1);
  CHECK((best_pair == std::array<int, 2>{ 1, 4 }));
}

TEST_CASE("F22 all retained C++ aliases preserve canonical behavior",
          "[api_2_0][deprecated][f22]")
{
  f22_compat_ledger ledger;
  const auto configured_root = f22_validated_test_root();
  bool source_identity_before_cleanup = false;
  {
  f22_scoped_test_root fixture_root;
  const auto legacy_root = fixture_root.make_leaf("legacy");
  const auto canonical_root = fixture_root.make_leaf("canonical");
  const auto source_root = fixture_root.make_leaf("source");
  CHECK(legacy_root != canonical_root);
  CHECK(legacy_root != source_root);
  CHECK(canonical_root != source_root);

  // 01. Problem::set_numberOfClusters(int)
  {
    Problem legacy = make_f22_problem({}, false);
    Problem canonical = make_f22_problem({}, false);
    DTWC_PUSH_NO_DEPRECATED
    legacy.set_numberOfClusters(2);
    DTWC_POP_NO_DEPRECATED
    canonical.set_n_clusters(2);
    ledger.behavior(
      f22_entity::set_number_of_clusters,
      legacy.n_clusters() == 2
        && canonical.n_clusters() == 2
        && legacy.labels().size() == f22_point_count
        && legacy.labels().size() == canonical.labels().size()
        && legacy.medoids().size() == 2
        && legacy.medoids().size() == canonical.medoids().size());
  }

  // 02. Problem::refreshDistanceMatrix()
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    legacy.fill_distance_matrix();
    canonical.fill_distance_matrix();
    const bool filled_before =
      legacy.is_distance_matrix_filled()
      && canonical.is_distance_matrix_filled();
    DTWC_PUSH_NO_DEPRECATED
    legacy.refreshDistanceMatrix();
    DTWC_POP_NO_DEPRECATED
    canonical.refresh_distance_matrix();
    const bool empty_after =
      !legacy.is_distance_matrix_filled()
      && !canonical.is_distance_matrix_filled();
    const double legacy_value = legacy.dist_by_ind(2, 5);
    const double canonical_value = canonical.dist_by_ind(2, 5);
    ledger.behavior(
      f22_entity::refresh_distance_matrix,
      filled_before
        && empty_after
        && f22_same_bits(legacy_value, 91.0)
        && f22_same_bits(legacy_value, canonical_value));
  }

  // 03. Problem::readDistanceMatrix(const fs::path&)
  {
    const auto source = source_root / "external.csv";
    const bool source_written =
      f22_write_text(source, f22_external_matrix_csv);
    const auto source_before = f22_read_bytes(source);

    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    DTWC_PUSH_NO_DEPRECATED
    legacy.readDistanceMatrix(source);
    DTWC_POP_NO_DEPRECATED
    canonical.read_distance_matrix(source);
    const bool loaded =
      f22_matrix_matches(legacy, f22_external_matrix)
      && f22_matrix_matches(canonical, f22_external_matrix)
      && f22_matrices_match(legacy, canonical);

    Problem failed_legacy = make_f22_problem();
    Problem failed_canonical = make_f22_problem();
    failed_legacy.fill_distance_matrix();
    failed_canonical.fill_distance_matrix();
    // A failed read propagates (audit 2026-09-02 A1); the deprecated alias must
    // fail exactly like the canonical spelling and must leave the already
    // filled matrix untouched.
    const auto capture_failure = [](auto &&call) {
      std::string message;
      try {
        std::forward<decltype(call)>(call)();
      } catch (const std::exception &error) {
        message = error.what();
      }
      return message;
    };
    std::string legacy_failure;
    DTWC_PUSH_NO_DEPRECATED
    legacy_failure = capture_failure([&] {
      failed_legacy.readDistanceMatrix(
        source_root / "missing-legacy.csv");
    });
    DTWC_POP_NO_DEPRECATED
    const auto canonical_failure = capture_failure([&] {
      failed_canonical.read_distance_matrix(
        source_root / "missing-canonical.csv");
    });
    const bool failure_identity =
      legacy_failure.find(f22_failed_read_message) != std::string::npos
      && canonical_failure.find(f22_failed_read_message) != std::string::npos
      && legacy_failure.find("missing-legacy.csv") != std::string::npos
      && canonical_failure.find("missing-canonical.csv") != std::string::npos
      && f22_matrix_matches(failed_legacy, f22_distance_oracle)
      && f22_matrix_matches(failed_canonical, f22_distance_oracle);

    const auto source_after = f22_read_bytes(source);
    source_identity_before_cleanup =
      source_written
      && source_before.has_value()
      && source_before == source_after
      && *source_before == f22_external_matrix_csv;
    const bool route_ok =
      loaded && failure_identity && source_identity_before_cleanup;
    ledger.behavior(f22_entity::read_distance_matrix, route_ok);
    ledger.io_route(f22_io_route::read_distance_matrix, route_ok);
    ledger.file_identity(
      f22_file_route::read_source,
      source_identity_before_cleanup);
  }

  // 04. Problem::maxDistance() const
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    legacy.fill_distance_matrix();
    canonical.fill_distance_matrix();
    double legacy_value = 0.0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value = legacy.maxDistance();
    DTWC_POP_NO_DEPRECATED
    const double canonical_value = canonical.max_distance();
    ledger.behavior(
      f22_entity::max_distance,
      f22_same_bits(legacy_value, 100.0)
        && f22_same_bits(legacy_value, canonical_value));
  }

  // 05. Problem::distByInd(int,int)
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    double legacy_value = 0.0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value = legacy.distByInd(2, 5);
    DTWC_POP_NO_DEPRECATED
    const double canonical_value = canonical.dist_by_ind(2, 5);
    ledger.behavior(
      f22_entity::dist_by_ind,
      f22_same_bits(legacy_value, 91.0)
        && f22_same_bits(legacy_value, canonical_value));
  }

  // 06. Problem::isDistanceMatrixFilled() const
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    bool legacy_before = true;
    DTWC_PUSH_NO_DEPRECATED
    legacy_before = legacy.isDistanceMatrixFilled();
    DTWC_POP_NO_DEPRECATED
    const bool canonical_before = canonical.is_distance_matrix_filled();
    legacy.fill_distance_matrix();
    canonical.fill_distance_matrix();
    bool legacy_after = false;
    DTWC_PUSH_NO_DEPRECATED
    legacy_after = legacy.isDistanceMatrixFilled();
    DTWC_POP_NO_DEPRECATED
    const bool canonical_after = canonical.is_distance_matrix_filled();
    ledger.behavior(
      f22_entity::is_distance_matrix_filled,
      !legacy_before && !canonical_before
        && legacy_after && canonical_after);
  }

  // 07. Problem::fillDistanceMatrix()
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    DTWC_PUSH_NO_DEPRECATED
    legacy.fillDistanceMatrix();
    DTWC_POP_NO_DEPRECATED
    canonical.fill_distance_matrix();
    ledger.behavior(
      f22_entity::fill_distance_matrix,
      legacy.is_distance_matrix_filled()
        && canonical.is_distance_matrix_filled()
        && f22_matrix_matches(legacy, f22_distance_oracle)
        && f22_matrix_matches(canonical, f22_distance_oracle)
        && f22_matrices_match(legacy, canonical));
  }

  // 08. Problem::printDistanceMatrix() const
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    legacy.fill_distance_matrix();
    canonical.fill_distance_matrix();
    std::string legacy_stdout;
    DTWC_PUSH_NO_DEPRECATED
    legacy_stdout = f22_capture_stdout([&] {
      legacy.printDistanceMatrix();
    });
    DTWC_POP_NO_DEPRECATED
    const auto canonical_stdout = f22_capture_stdout([&] {
      canonical.print_distance_matrix();
    });
    const bool stdout_ok =
      legacy_stdout == canonical_stdout
      && legacy_stdout == f22_distance_stdout;
    ledger.behavior(f22_entity::print_distance_matrix, stdout_ok);
    ledger.stdout_identity(
      f22_stdout_route::print_distance_matrix,
      stdout_ok);
  }

  // 09. Problem::writeDistanceMatrix(const std::string&) const
  {
    Problem legacy = make_f22_problem(legacy_root);
    Problem canonical = make_f22_problem(canonical_root);
    legacy.fill_distance_matrix();
    canonical.fill_distance_matrix();
    DTWC_PUSH_NO_DEPRECATED
    legacy.writeDistanceMatrix("explicit.csv");
    DTWC_POP_NO_DEPRECATED
    canonical.write_distance_matrix("explicit.csv");
    const auto legacy_file = legacy_root / "explicit.csv";
    const auto canonical_file = canonical_root / "explicit.csv";
    const auto bytes = f22_read_bytes(legacy_file);
    const bool file_ok =
      f22_equal_nonempty_files(legacy_file, canonical_file)
      && bytes.has_value()
      && *bytes == f22_distance_stdout;
    ledger.behavior(f22_entity::write_distance_matrix_named, file_ok);
    ledger.io_route(
      f22_io_route::write_distance_matrix_named,
      file_ok);
    ledger.file_identity(
      f22_file_route::write_distance_matrix_named,
      file_ok);
  }

  // 10. Problem::writeDistanceMatrix() const
  {
    Problem legacy = make_f22_problem(legacy_root);
    Problem canonical = make_f22_problem(canonical_root);
    legacy.fill_distance_matrix();
    canonical.fill_distance_matrix();
    DTWC_PUSH_NO_DEPRECATED
    legacy.writeDistanceMatrix();
    DTWC_POP_NO_DEPRECATED
    canonical.write_distance_matrix();
    const auto legacy_file =
      legacy_root / "f22_asymmetric_distanceMatrix.csv";
    const auto canonical_file =
      canonical_root / "f22_asymmetric_distanceMatrix.csv";
    const auto bytes = f22_read_bytes(legacy_file);
    const bool file_ok =
      f22_equal_nonempty_files(legacy_file, canonical_file)
      && bytes.has_value()
      && *bytes == f22_distance_stdout;
    ledger.behavior(f22_entity::write_distance_matrix_default, file_ok);
    ledger.io_route(
      f22_io_route::write_distance_matrix_default,
      file_ok);
    ledger.file_identity(
      f22_file_route::write_distance_matrix_default,
      file_ok);
  }

  // 11. Problem::printClusters() const
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    std::string legacy_stdout;
    DTWC_PUSH_NO_DEPRECATED
    legacy_stdout = f22_capture_stdout([&] {
      legacy.printClusters();
    });
    DTWC_POP_NO_DEPRECATED
    const auto canonical_stdout = f22_capture_stdout([&] {
      canonical.print_clusters();
    });
    const bool stdout_ok =
      legacy_stdout == canonical_stdout
      && legacy_stdout == f22_clusters_stdout;
    ledger.behavior(f22_entity::print_clusters, stdout_ok);
    ledger.io_route(f22_io_route::print_clusters, stdout_ok);
    ledger.stdout_identity(f22_stdout_route::print_clusters, stdout_ok);
  }

  // 12. Problem::writeClusters()
  {
    constexpr std::string_view expected =
      "Cluster centroids:\n"
      "s1,s4\n"
      "\n"
      "Data,its cluster\n"
      "s0,s1\n"
      "s1,s1\n"
      "s2,s1\n"
      "s3,s4\n"
      "s4,s4\n"
      "s5,s4\n"
      "Procedure is completed with cost: 69\n";
    Problem legacy = make_f22_problem(legacy_root);
    Problem canonical = make_f22_problem(canonical_root);
    DTWC_PUSH_NO_DEPRECATED
    legacy.writeClusters();
    DTWC_POP_NO_DEPRECATED
    canonical.write_clusters();
    const auto legacy_file =
      legacy_root / "f22_asymmetric_Nc_2.csv";
    const auto canonical_file =
      canonical_root / "f22_asymmetric_Nc_2.csv";
    const auto bytes = f22_read_bytes(legacy_file);
    const bool file_ok =
      f22_equal_nonempty_files(legacy_file, canonical_file)
      && bytes.has_value()
      && f22_normalize_crlf(*bytes) == expected;
    ledger.behavior(f22_entity::write_clusters, file_ok);
    ledger.io_route(f22_io_route::write_clusters, file_ok);
    ledger.file_identity(f22_file_route::write_clusters, file_ok);
  }

  // 13. Problem::writeMedoidMembers(int,int) const
  {
    constexpr std::string_view expected =
      "s0,s1,s2,\n"
      "s3,s4,s5,\n";
    Problem legacy = make_f22_problem(legacy_root);
    Problem canonical = make_f22_problem(canonical_root);
    DTWC_PUSH_NO_DEPRECATED
    legacy.writeMedoidMembers(7, 3);
    DTWC_POP_NO_DEPRECATED
    canonical.write_medoid_members(7, 3);
    constexpr std::string_view filename =
      "medoidMembers_Nc_2_rep_3_iter_7.csv";
    const auto legacy_file = legacy_root / std::string{ filename };
    const auto canonical_file = canonical_root / std::string{ filename };
    const auto bytes = f22_read_bytes(legacy_file);
    const bool file_ok =
      f22_equal_nonempty_files(legacy_file, canonical_file)
      && bytes.has_value()
      && f22_normalize_crlf(*bytes) == expected;
    ledger.behavior(f22_entity::write_medoid_members, file_ok);
    ledger.io_route(f22_io_route::write_medoid_members, file_ok);
    ledger.file_identity(
      f22_file_route::write_medoid_members,
      file_ok);
  }

  // 14. Problem::writeSilhouettes()
  {
    constexpr std::string_view expected =
      "Silhouettes:\n"
      "s0,0.915385\n"
      "s1,0.928571\n"
      "s2,0.857143\n"
      "s3,-0.0311111\n"
      "s4,0.415584\n"
      "s5,0.455017\n";
    Problem legacy = make_f22_problem(legacy_root);
    Problem canonical = make_f22_problem(canonical_root);
    DTWC_PUSH_NO_DEPRECATED
    legacy.writeSilhouettes();
    DTWC_POP_NO_DEPRECATED
    canonical.write_silhouettes();
    const auto legacy_file =
      legacy_root / "f22_asymmetric_silhouettes_Nc_2.csv";
    const auto canonical_file =
      canonical_root / "f22_asymmetric_silhouettes_Nc_2.csv";
    const auto bytes = f22_read_bytes(legacy_file);
    const bool expected_contents =
      bytes.has_value()
      && f22_normalize_crlf(*bytes) == expected;
    const bool file_ok =
      f22_equal_nonempty_files(legacy_file, canonical_file)
      && expected_contents;
    ledger.behavior(f22_entity::write_silhouettes, file_ok);
    ledger.io_route(f22_io_route::write_silhouettes, file_ok);
    ledger.file_identity(f22_file_route::write_silhouettes, file_ok);
  }

  // 15. Problem::findTotalCost()
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    double legacy_value = 0.0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value = legacy.findTotalCost();
    DTWC_POP_NO_DEPRECATED
    const double canonical_value = canonical.find_total_cost();
    ledger.behavior(
      f22_entity::find_total_cost,
      f22_same_bits(legacy_value, 69.0)
        && f22_same_bits(legacy_value, canonical_value));
  }

  // 16. Problem::assignClusters()
  {
    const std::vector<int> expected{ 0, 0, 0, 1, 1, 1 };
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    legacy.clusters_ind.assign(f22_point_count, -1);
    canonical.clusters_ind.assign(f22_point_count, -1);
    DTWC_PUSH_NO_DEPRECATED
    legacy.assignClusters();
    DTWC_POP_NO_DEPRECATED
    canonical.assign_clusters();
    ledger.behavior(
      f22_entity::assign_clusters,
      legacy.labels() == expected
        && canonical.labels() == expected
        && legacy.labels() == canonical.labels());
  }

  // 17. Problem::calculateMedoids()
  {
    const std::vector<int> expected{ 1, 4 };
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    legacy.fill_distance_matrix();
    canonical.fill_distance_matrix();
    legacy.centroids_ind = { 0, 3 };
    canonical.centroids_ind = { 0, 3 };
    DTWC_PUSH_NO_DEPRECATED
    legacy.calculateMedoids();
    DTWC_POP_NO_DEPRECATED
    canonical.calculate_medoids();
    ledger.behavior(
      f22_entity::calculate_medoids,
      legacy.medoids() == expected
        && canonical.medoids() == expected
        && legacy.medoids() == canonical.medoids());
  }

  // 18. Problem::cluster_by_MIP()
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    legacy.centroids_ind = { 0, 5 };
    legacy.clusters_ind = { 1, 1, 1, 0, 0, 0 };
    canonical.centroids_ind = { 2, 3 };
    canonical.clusters_ind = { 1, 0, 1, 0, 1, 0 };
    const auto legacy_labels_before = legacy.labels();
    const auto legacy_medoids_before = legacy.medoids();
    const auto canonical_labels_before = canonical.labels();
    const auto canonical_medoids_before = canonical.medoids();
    legacy.mip_settings.warm_start = false;
    canonical.mip_settings.warm_start = false;
    legacy.mip_settings.mip_gap = 0.0;
    canonical.mip_settings.mip_gap = 0.0;
    legacy.mip_settings.verbose_solver = false;
    canonical.mip_settings.verbose_solver = false;
    legacy.set_solver(Solver::HiGHS);
    canonical.set_solver(Solver::HiGHS);
    f22_exception_outcome legacy_outcome;
    DTWC_PUSH_NO_DEPRECATED
    legacy_outcome = f22_capture_exception([&] {
      legacy.cluster_by_MIP();
    });
    DTWC_POP_NO_DEPRECATED
    const auto canonical_outcome = f22_capture_exception([&] {
      canonical.cluster_by_mip();
    });

    bool route_ok = false;
    if (highs_solver_available()) {
      const std::vector<int> expected_labels{ 0, 0, 0, 1, 1, 1 };
      const std::vector<int> expected_medoids{ 1, 4 };
      route_ok =
        legacy_outcome.kind == f22_exception_kind::none
        && canonical_outcome.kind == f22_exception_kind::none
        && legacy.labels() == expected_labels
        && canonical.labels() == expected_labels
        && legacy.medoids() == expected_medoids
        && canonical.medoids() == expected_medoids
        && legacy.labels() == canonical.labels()
        && legacy.medoids() == canonical.medoids()
        && f22_same_bits(legacy.find_total_cost(), 69.0)
        && f22_same_bits(
          legacy.find_total_cost(), canonical.find_total_cost());
    } else {
      constexpr std::string_view expected =
        "HiGHS solver is unavailable; rebuild with "
        "-DDTWC_ENABLE_HIGHS=ON";
      route_ok =
        legacy_outcome.kind == f22_exception_kind::solver_error
        && canonical_outcome.kind == f22_exception_kind::solver_error
        && legacy_outcome.message == expected
        && canonical_outcome.message == expected
        && legacy.labels() == legacy_labels_before
        && legacy.medoids() == legacy_medoids_before
        && canonical.labels() == canonical_labels_before
        && canonical.medoids() == canonical_medoids_before;
    }
    ledger.behavior(f22_entity::cluster_by_mip, route_ok);
  }

  // 19. Problem::cluster_by_kMedoidsLloyd()
  {
    constexpr std::string_view expected =
      "Lloyd k-medoids requires n_repetitions >= 1.";
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    legacy.set_n_repetitions(0);
    canonical.set_n_repetitions(0);
    const auto legacy_labels_before = legacy.labels();
    const auto legacy_medoids_before = legacy.medoids();
    const auto canonical_labels_before = canonical.labels();
    const auto canonical_medoids_before = canonical.medoids();
    f22_exception_outcome legacy_outcome;
    DTWC_PUSH_NO_DEPRECATED
    legacy_outcome = f22_capture_exception([&] {
      legacy.cluster_by_kMedoidsLloyd();
    });
    DTWC_POP_NO_DEPRECATED
    const auto canonical_outcome = f22_capture_exception([&] {
      canonical.cluster_by_kmedoids_lloyd();
    });
    ledger.behavior(
      f22_entity::cluster_by_kmedoids_lloyd,
      legacy_outcome.kind == f22_exception_kind::invalid_input
        && canonical_outcome.kind == f22_exception_kind::invalid_input
        && legacy_outcome.message == expected
        && canonical_outcome.message == expected
        && legacy.labels() == legacy_labels_before
        && legacy.medoids() == legacy_medoids_before
        && canonical.labels() == canonical_labels_before
        && canonical.medoids() == canonical_medoids_before);
  }

  // 20. Problem::cluster_size() const
  {
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    int legacy_value = 0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value = legacy.cluster_size();
    DTWC_POP_NO_DEPRECATED
    const int canonical_value = canonical.n_clusters();
    ledger.behavior(
      f22_entity::cluster_size,
      legacy_value == 2 && legacy_value == canonical_value);
  }

  // 21. scores::daviesBouldinIndex(Problem&)
  {
    const double expected = 23.0 / 53.0;
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    double legacy_value = 0.0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value = scores::daviesBouldinIndex(legacy);
    DTWC_POP_NO_DEPRECATED
    const double canonical_value = scores::davies_bouldin(canonical);
    ledger.behavior(
      f22_entity::davies_bouldin,
      std::isfinite(legacy_value)
        && legacy_value > 0.0
        && f22_same_bits(legacy_value, expected)
        && f22_same_bits(legacy_value, canonical_value));
  }

  // 22. scores::dunnIndex(Problem&)
  {
    const double expected = 31.0 / 60.0;
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    double legacy_value = 0.0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value = scores::dunnIndex(legacy);
    DTWC_POP_NO_DEPRECATED
    const double canonical_value = scores::dunn(canonical);
    ledger.behavior(
      f22_entity::dunn,
      std::isfinite(legacy_value)
        && legacy_value > 0.0
        && f22_same_bits(legacy_value, expected)
        && f22_same_bits(legacy_value, canonical_value));
  }

  // 23. scores::calinskiHarabaszIndex(Problem&)
  {
    const double expected = 25980.0 / 2303.0;
    Problem legacy = make_f22_problem();
    Problem canonical = make_f22_problem();
    double legacy_value = 0.0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value = scores::calinskiHarabaszIndex(legacy);
    DTWC_POP_NO_DEPRECATED
    const double canonical_value = scores::calinski_harabasz(canonical);
    ledger.behavior(
      f22_entity::calinski_harabasz,
      std::isfinite(legacy_value)
        && legacy_value > 0.0
        && f22_same_bits(legacy_value, expected)
        && f22_same_bits(legacy_value, canonical_value));
  }

  const std::vector<int> labels_true{ 1, 1, 1, 2, 2, 2, 3, 3 };
  const std::vector<int> labels_pred{ 1, 1, 2, 2, 2, 3, 3, 3 };

  // 24. scores::adjustedRandIndex(labels,labels)
  {
    const double expected = 5.0 / 21.0;
    double legacy_value = 0.0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value = scores::adjustedRandIndex(labels_true, labels_pred);
    DTWC_POP_NO_DEPRECATED
    const double canonical_value =
      scores::adjusted_rand(labels_true, labels_pred);
    ledger.behavior(
      f22_entity::adjusted_rand,
      std::isfinite(legacy_value)
        && legacy_value > 0.0
        && legacy_value < 1.0
        && f22_same_bits(legacy_value, expected)
        && f22_same_bits(legacy_value, canonical_value));
  }

  // 25. scores::normalizedMutualInformation(labels,labels)
  {
    constexpr double expected = 0.55887303821703238;
    double legacy_value = 0.0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_value =
      scores::normalizedMutualInformation(labels_true, labels_pred);
    DTWC_POP_NO_DEPRECATED
    const double canonical_value =
      scores::normalized_mutual_info(labels_true, labels_pred);
    ledger.behavior(
      f22_entity::normalized_mutual_info,
      std::isfinite(legacy_value)
        && legacy_value > 0.0
        && legacy_value < 1.0
        && std::abs(legacy_value - expected) <= 1e-15
        && f22_same_bits(legacy_value, canonical_value));
  }

  // 26. DataLoader::startColumn(int)
  {
    DataLoader legacy;
    DataLoader canonical;
    legacy.start_row(19);
    canonical.start_row(19);
    DataLoader *legacy_receiver = nullptr;
    DTWC_PUSH_NO_DEPRECATED
    legacy_receiver = &legacy.startColumn(7);
    DTWC_POP_NO_DEPRECATED
    DataLoader *canonical_receiver = &canonical.start_column(7);
    ledger.behavior(
      f22_entity::loader_start_column,
      legacy_receiver == &legacy
        && canonical_receiver == &canonical
        && legacy.startColumn() == 7
        && canonical.startColumn() == 7
        && legacy.startRow() == 19
        && canonical.startRow() == 19);
  }

  // 27. DataLoader::startRow(int)
  {
    DataLoader legacy;
    DataLoader canonical;
    legacy.start_column(23);
    canonical.start_column(23);
    DataLoader *legacy_receiver = nullptr;
    DTWC_PUSH_NO_DEPRECATED
    legacy_receiver = &legacy.startRow(11);
    DTWC_POP_NO_DEPRECATED
    DataLoader *canonical_receiver = &canonical.start_row(11);
    ledger.behavior(
      f22_entity::loader_start_row,
      legacy_receiver == &legacy
        && canonical_receiver == &canonical
        && legacy.startRow() == 11
        && canonical.startRow() == 11
        && legacy.startColumn() == 23
        && canonical.startColumn() == 23);
  }

  f22_path_settings_guard restore_paths;
  const auto poison_paths = [](
                              const settings::fs::path &data,
                              const settings::fs::path &results) {
    settings::paths::data = data;
    settings::paths::results = results;
  };

  // 28. settings::paths::setDataPath(const fs::path&)
  {
    poison_paths("poison-data-28", "poison-results-28");
    settings::paths::set_data_path(settings::fs::path{ "f22-data-path" });
    const auto canonical_data = settings::paths::data;
    const auto canonical_results = settings::paths::results;
    poison_paths("poison-data-28", "poison-results-28");
    DTWC_PUSH_NO_DEPRECATED
    settings::paths::setDataPath(settings::fs::path{ "f22-data-path" });
    DTWC_POP_NO_DEPRECATED
    ledger.behavior(
      f22_entity::set_data_path_path,
      settings::paths::data == canonical_data
        && settings::paths::results == canonical_results
        && settings::paths::data == "f22-data-path"
        && settings::paths::results == "poison-results-28");
  }

  // 29. settings::paths::setDataPath(const char*)
  {
    poison_paths("poison-data-29", "poison-results-29");
    {
      const std::string candidate = "f22-cstring-data";
      settings::paths::set_data_path(candidate.c_str());
    }
    const auto canonical_data = settings::paths::data;
    const auto canonical_results = settings::paths::results;
    poison_paths("poison-data-29", "poison-results-29");
    {
      const std::string candidate = "f22-cstring-data";
      DTWC_PUSH_NO_DEPRECATED
      settings::paths::setDataPath(candidate.c_str());
      DTWC_POP_NO_DEPRECATED
    }
    ledger.behavior(
      f22_entity::set_data_path_cstring,
      settings::paths::data == canonical_data
        && settings::paths::results == canonical_results
        && settings::paths::data == "f22-cstring-data"
        && settings::paths::results == "poison-results-29");
  }

  // 30. settings::paths::setResultsPath(const fs::path&)
  {
    poison_paths("poison-data-30", "poison-results-30");
    settings::paths::set_results_path(
      settings::fs::path{ "f22-results-path" });
    const auto canonical_data = settings::paths::data;
    const auto canonical_results = settings::paths::results;
    poison_paths("poison-data-30", "poison-results-30");
    DTWC_PUSH_NO_DEPRECATED
    settings::paths::setResultsPath(
      settings::fs::path{ "f22-results-path" });
    DTWC_POP_NO_DEPRECATED
    ledger.behavior(
      f22_entity::set_results_path_path,
      settings::paths::data == canonical_data
        && settings::paths::results == canonical_results
        && settings::paths::data == "poison-data-30"
        && settings::paths::results == "f22-results-path");
  }

  // 31. settings::paths::setResultsPath(const char*)
  {
    poison_paths("poison-data-31", "poison-results-31");
    {
      const std::string candidate = "f22-cstring-results";
      settings::paths::set_results_path(candidate.c_str());
    }
    const auto canonical_data = settings::paths::data;
    const auto canonical_results = settings::paths::results;
    poison_paths("poison-data-31", "poison-results-31");
    {
      const std::string candidate = "f22-cstring-results";
      DTWC_PUSH_NO_DEPRECATED
      settings::paths::setResultsPath(candidate.c_str());
      DTWC_POP_NO_DEPRECATED
    }
    ledger.behavior(
      f22_entity::set_results_path_cstring,
      settings::paths::data == canonical_data
        && settings::paths::results == canonical_results
        && settings::paths::data == "poison-data-31"
        && settings::paths::results == "f22-cstring-results");
  }

  // 32. Problem::maxIter
  {
    Problem canonical_write = make_f22_problem();
    canonical_write.set_n_repetitions(41);
    canonical_write.set_max_iter(17);
    int legacy_read = 0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_read = canonical_write.maxIter;
    DTWC_POP_NO_DEPRECATED
    const bool canonical_write_legacy_read =
      legacy_read == 17
      && canonical_write.n_repetitions() == 41;

    Problem legacy_write = make_f22_problem();
    legacy_write.set_n_repetitions(43);
    DTWC_PUSH_NO_DEPRECATED
    legacy_write.maxIter = 23;
    DTWC_POP_NO_DEPRECATED
    const bool legacy_write_canonical_read =
      legacy_write.max_iter() == 23
      && legacy_write.n_repetitions() == 43;
    ledger.field_route(
      f22_field_route::max_iter_canonical_write,
      canonical_write_legacy_read);
    ledger.field_route(
      f22_field_route::max_iter_legacy_write,
      legacy_write_canonical_read);
    ledger.behavior(
      f22_entity::max_iter_field,
      canonical_write_legacy_read && legacy_write_canonical_read);
  }

  // 33. Problem::N_repetition
  {
    Problem canonical_write = make_f22_problem();
    canonical_write.set_max_iter(47);
    canonical_write.set_n_repetitions(3);
    int legacy_read = 0;
    DTWC_PUSH_NO_DEPRECATED
    legacy_read = canonical_write.N_repetition;
    DTWC_POP_NO_DEPRECATED
    const bool canonical_write_legacy_read =
      legacy_read == 3
      && canonical_write.max_iter() == 47;

    Problem legacy_write = make_f22_problem();
    legacy_write.set_max_iter(53);
    DTWC_PUSH_NO_DEPRECATED
    legacy_write.N_repetition = 5;
    DTWC_POP_NO_DEPRECATED
    const bool legacy_write_canonical_read =
      legacy_write.n_repetitions() == 5
      && legacy_write.max_iter() == 53;
    ledger.field_route(
      f22_field_route::n_repetition_canonical_write,
      canonical_write_legacy_read);
    ledger.field_route(
      f22_field_route::n_repetition_legacy_write,
      legacy_write_canonical_read);
    ledger.behavior(
      f22_entity::n_repetition_field,
      canonical_write_legacy_read && legacy_write_canonical_read);
  }
  } // fixture_root is destroyed here and removes the whole validated root.

  std::error_code cleanup_error;
  const bool artifact_cleanup =
    !std::filesystem::exists(configured_root, cleanup_error)
    && !cleanup_error;
  CHECK(artifact_cleanup);

  const int inventory = ledger.inventory();
  const int behavior = ledger.behavior_count();
  const int field_routes = ledger.field_routes();
  const int io_routes = ledger.io_routes();
  const int file_identity = ledger.file_identity_count();
  const int stdout_identity = ledger.stdout_identity_count();
  const bool all_pass =
    ledger.all()
    && inventory == 33
    && behavior == 33
    && field_routes == 4
    && io_routes == 7
    && file_identity == 6
    && stdout_identity == 2
    && artifact_cleanup;
  const char *verdict = all_pass ? "PASS" : "FAIL";

  std::cout
    << "F22_CPP_COMPAT inventory=" << inventory << "/33"
    << " behavior=" << behavior << "/33"
    << " field_routes=" << field_routes << "/4"
    << " io_routes=" << io_routes << "/7"
    << " file_identity=" << file_identity << "/6"
    << " stdout_identity=" << stdout_identity << "/2"
    << " skips=0 verdict=" << verdict << '\n';
  CHECK(inventory == 33);
  CHECK(behavior == 33);
  CHECK(field_routes == 4);
  CHECK(io_routes == 7);
  CHECK(file_identity == 6);
  CHECK(stdout_identity == 2);
  REQUIRE(all_pass);
}
