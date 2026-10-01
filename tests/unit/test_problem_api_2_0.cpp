/**
 * @file test_problem_api_2_0.cpp
 * @brief Every 2.0 Problem / DataLoader entry point, and the v1.0.0 spelling that
 *        forwards to it, against a hand oracle.
 *
 * @details One table (docs/api-contract-2.0.md): each row runs an entry point by
 *          its 2.0 name and by its 1.x name (which compiles with a deprecation
 *          warning, suppressed locally) and compares one observation with one
 *          hand-derived oracle. test_deprecated_shims_warn proves the 1.x names
 *          still warn. The rows include the result write-back
 *          (scores::silhouette(prob) after fast_pam, no manual wiring) and the
 *          set_variant rebind of the bound DTW function.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <dtwc.hpp>
#include <mip/mip.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
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

using namespace dtwc;

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

} // namespace

// ===========================================================================
// One table over every 2.0 Problem / DataLoader entry point.
//
// Fixture: six scalar series 0, 2, 9, 40, 55, 100 (DTW distance |a - b|), two
// clusters {0,1,2} and {3,4,5} with medoids 1 and 4. Hand oracles:
//   max distance |100 - 0| = 100, d(2,5) = |9 - 100| = 91,
//   cost = (2 + 0 + 7) + (15 + 0 + 45) = 69 (the next best medoid pair costs 71),
//   s0 silhouette = (65 - 5.5) / 65 = 0.915385 (a = (2+9)/2, b = (40+55+100)/3).
// Each row runs the 2.0 name and, where v1.0.0 had a spelling, that spelling too,
// each on a fresh Problem with its own output folder, and compares one
// observation string with one oracle.
// ===========================================================================
namespace {

namespace fs = std::filesystem;

struct Row {
  const char *entry;
  bool clustered; // start from centroids {1,4}, labels {0,0,0,1,1,1}
  bool has_v1;    // v1.0.0 spelled this entry point differently
  std::string (*observe)(Problem &, const fs::path &dir, bool v1);
  std::string oracle;
};

template <class... T> std::string obs(const T &...values)
{
  std::ostringstream out;
  out << std::setprecision(17);
  ((out << values << ' '), ...);
  return out.str();
}

std::string ids(const std::vector<index_t> &values)
{
  std::string joined;
  for (const index_t value : values)
    joined += (joined.empty() ? "" : ",") + std::to_string(value);
  return joined;
}

std::string matrix_csv(Problem &problem)
{
  std::ostringstream out;
  out << std::setprecision(17);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      out << problem.dist_by_ind(i, j) << (j == 5 ? '\n' : ',');
  return out.str();
}

std::string file_text(const fs::path &path)
{
  return f22_normalize_crlf(
    f22_read_bytes(path).value_or("<missing " + path.filename().string() + ">"));
}

} // namespace

TEST_CASE("Problem 2.0 entry points match hand oracles, by 2.0 name and 1.x spelling",
          "[api_2_0][table][f22]")
{
  // Independent oracle for find_total_cost and cluster_by_mip: every medoid pair.
  using PairCost = std::pair<double, std::pair<std::size_t, std::size_t>>;
  std::vector<PairCost> pair_costs;
  for (std::size_t a = 0; a < f22_point_count; ++a)
    for (std::size_t b = a + 1; b < f22_point_count; ++b) {
      double cost = 0.0;
      for (std::size_t point = 0; point < f22_point_count; ++point)
        cost += std::min(f22_distance_oracle[point][a], f22_distance_oracle[point][b]);
      pair_costs.push_back({ cost, { a, b } });
    }
  std::sort(pair_costs.begin(), pair_costs.end());
  CHECK(pair_costs[0] == PairCost{ 69.0, { 1, 4 } });
  CHECK(pair_costs[1].first == 71.0);

  const auto configured_root = f22_validated_test_root();
  {
    f22_scoped_test_root root;
    DTWC_PUSH_NO_DEPRECATED
    const std::vector<Row> rows{
      { "set_n_clusters", false, true, [](Problem &p, const fs::path &, bool v1) {
          v1 ? p.set_numberOfClusters(2) : p.set_n_clusters(2);
          return obs(p.n_clusters(), p.labels().size(), p.medoids().size()); },
        "2 0 0 " },
      { "n_clusters", true, true, [](Problem &p, const fs::path &, bool v1) {
          return obs(v1 ? p.cluster_size() : p.n_clusters()); },
        "2 " },
      { "refresh_distance_matrix", true, true, [](Problem &p, const fs::path &, bool v1) {
          p.fill_distance_matrix();
          const bool filled = p.is_distance_matrix_filled();
          v1 ? p.refreshDistanceMatrix() : p.refresh_distance_matrix();
          const bool emptied = !p.is_distance_matrix_filled();
          p.fill_distance_matrix(); // dist_by_ind reads a matrix a fill prepared
          return obs(filled, emptied, p.dist_by_ind(2, 5)); },
        "1 1 91 " },
      { "read_distance_matrix", true, true, [](Problem &p, const fs::path &dir, bool v1) {
          const auto source = dir / "external.csv";
          f22_write_text(source, f22_external_matrix_csv);
          v1 ? p.readDistanceMatrix(source) : p.read_distance_matrix(source);
          const std::string loaded = matrix_csv(p);
          std::string failure; // a failed read propagates and leaves the matrix as it was
          try { p.read_distance_matrix(dir / "missing.csv"); }
          catch (const std::exception &e) { failure = e.what(); }
          const bool named = failure.find(f22_failed_read_message) != std::string::npos
                             && failure.find("missing.csv") != std::string::npos;
          return loaded + obs(named, matrix_csv(p) == loaded,
                              f22_read_bytes(source) == f22_external_matrix_csv); },
        std::string{ f22_external_matrix_csv } + "1 1 1 " },
      { "max_distance", true, true, [](Problem &p, const fs::path &, bool v1) {
          p.fill_distance_matrix();
          return obs(v1 ? p.maxDistance() : p.max_distance()); },
        "100 " },
      { "dist_by_ind", true, true, [](Problem &p, const fs::path &, bool v1) {
          // The 2.0 lookup reads a matrix a fill prepared; the 1.x one fills on its first call.
          if (!v1) p.fill_distance_matrix();
          const double d = v1 ? p.distByInd(2, 5) : p.dist_by_ind(2, 5);
          return obs(d, p.is_distance_matrix_filled()); },
        "91 1 " },
      { "is_distance_matrix_filled", true, true, [](Problem &p, const fs::path &, bool v1) {
          const bool before = v1 ? p.isDistanceMatrixFilled() : p.is_distance_matrix_filled();
          p.fill_distance_matrix();
          return obs(before, v1 ? p.isDistanceMatrixFilled() : p.is_distance_matrix_filled()); },
        "0 1 " },
      { "fill_distance_matrix", true, true, [](Problem &p, const fs::path &, bool v1) {
          v1 ? p.fillDistanceMatrix() : p.fill_distance_matrix();
          return obs(p.is_distance_matrix_filled()) + matrix_csv(p); },
        "1 " + std::string{ f22_distance_stdout } },
      { "print_distance_matrix", true, true, [](Problem &p, const fs::path &, bool v1) {
          p.fill_distance_matrix();
          return f22_capture_stdout(
            [&] { v1 ? p.printDistanceMatrix() : p.print_distance_matrix(); }); },
        std::string{ f22_distance_stdout } },
      { "write_distance_matrix(name)", true, true,
        [](Problem &p, const fs::path &dir, bool v1) {
          p.fill_distance_matrix();
          v1 ? p.writeDistanceMatrix("explicit.csv") : p.write_distance_matrix("explicit.csv");
          return file_text(dir / "explicit.csv"); },
        std::string{ f22_distance_stdout } },
      { "write_distance_matrix()", true, true, [](Problem &p, const fs::path &dir, bool v1) {
          p.fill_distance_matrix();
          v1 ? p.writeDistanceMatrix() : p.write_distance_matrix();
          return file_text(dir / "f22_asymmetric_distanceMatrix.csv"); },
        std::string{ f22_distance_stdout } },
      { "print_clusters", true, true, [](Problem &p, const fs::path &, bool v1) {
          return f22_capture_stdout([&] { v1 ? p.printClusters() : p.print_clusters(); }); },
        std::string{ f22_clusters_stdout } },
      { "write_clusters", true, true, [](Problem &p, const fs::path &dir, bool v1) {
          v1 ? p.writeClusters() : p.write_clusters();
          return file_text(dir / "f22_asymmetric_Nc_2.csv"); },
        "Cluster centroids:\ns1,s4\n\nData,its cluster\ns0,s1\ns1,s1\ns2,s1\ns3,s4\ns4,s4\ns5,s4\n"
        "Procedure is completed with cost: 69\n" },
      { "write_medoid_members", true, true, [](Problem &p, const fs::path &dir, bool v1) {
          v1 ? p.writeMedoidMembers(7, 3) : p.write_medoid_members(7, 3);
          return file_text(dir / "medoidMembers_Nc_2_rep_3_iter_7.csv"); },
        "s0,s1,s2,\ns3,s4,s5,\n" },
      { "write_silhouettes", true, true, [](Problem &p, const fs::path &dir, bool v1) {
          v1 ? p.writeSilhouettes() : p.write_silhouettes();
          return file_text(dir / "f22_asymmetric_silhouettes_Nc_2.csv"); },
        "Silhouettes:\ns0,0.915385\ns1,0.928571\ns2,0.857143\ns3,-0.0311111\ns4,0.415584\n"
        "s5,0.455017\n" },
      { "find_total_cost", true, true, [](Problem &p, const fs::path &, bool v1) {
          return obs(v1 ? p.findTotalCost() : p.find_total_cost()); },
        "69 " },
      { "assign_clusters", true, true, [](Problem &p, const fs::path &, bool v1) {
          p.clusters_ind.assign(6, -1);
          v1 ? p.assignClusters() : p.assign_clusters();
          return ids(p.labels()); },
        "0,0,0,1,1,1" },
      { "calculate_medoids", true, true, [](Problem &p, const fs::path &, bool v1) {
          p.fill_distance_matrix();
          p.centroids_ind = { 0, 3 }; // member sums: {0,1,2} -> 9 at point 1; {3,4,5} -> 60 at 4
          v1 ? p.calculateMedoids() : p.calculate_medoids();
          return ids(p.medoids()); },
        "1,4" },
      { "cluster_by_mip", true, true, [](Problem &p, const fs::path &, bool v1) {
          p.centroids_ind = { 0, 5 };
          p.clusters_ind = { 1, 1, 1, 0, 0, 0 };
          p.mip_settings.warm_start = false;
          p.mip_settings.mip_gap = 0.0;
          p.mip_settings.verbose_solver = false;
          REQUIRE(p.set_solver(Solver::HiGHS));
          try { v1 ? p.cluster_by_MIP() : p.cluster_by_mip(); }
          catch (const SolverError &e) { // typed, and the state is kept
            return std::string{ e.what() } + " / " + obs(ids(p.labels())); }
          return obs(ids(p.labels()), ids(p.medoids()), p.find_total_cost()); },
        highs_solver_available()
          ? "0,0,0,1,1,1 1,4 69 "
          : "HiGHS solver is unavailable; rebuild with -DDTWC_ENABLE_HIGHS=ON / 1,1,1,0,0,0 " },
      { "cluster_by_kmedoids_lloyd", true, true, [](Problem &p, const fs::path &, bool v1) {
          p.N_repetition = 0; // set_n_repetitions(0) throws; only the field reaches Lloyd's check
          try { v1 ? p.cluster_by_kMedoidsPAM() : p.cluster_by_kmedoids_lloyd(); }
          catch (const InvalidInput &e) {
            return std::string{ e.what() } + " / " + obs(ids(p.labels()), ids(p.medoids())); }
          return std::string{ "no InvalidInput" }; },
        "Lloyd k-medoids requires n_repetitions >= 1. / 0,0,0,1,1,1 1,4 " },
      { "DataLoader::start_column", true, true, [](Problem &, const fs::path &, bool v1) {
          DataLoader loader;
          loader.start_row(19);
          DataLoader &returned = v1 ? loader.startColumn(7) : loader.start_column(7);
          return obs(&returned == &loader, loader.startColumn(), loader.startRow()); },
        "1 7 19 " },
      { "DataLoader::start_row", true, true, [](Problem &, const fs::path &, bool v1) {
          DataLoader loader;
          loader.start_column(23);
          DataLoader &returned = v1 ? loader.startRow(11) : loader.start_row(11);
          return obs(&returned == &loader, loader.startRow(), loader.startColumn()); },
        "1 11 23 " },
      { "set_max_iter, set_n_repetitions", true, true,
        [](Problem &p, const fs::path &, bool v1) {
          if (v1) { p.maxIter = 17; p.N_repetition = 41; }
          else { p.set_n_repetitions(41); p.set_max_iter(17); }
          return obs(p.max_iter(), p.n_repetitions(), p.maxIter, p.N_repetition); },
        "17 41 17 41 " },
      { "set_variant rebinds the DTW function", true, false,
        [](Problem &, const fs::path &, bool) {
          // x = {0,0}, y = {0,1,2}, L1, full band: Standard = 3; ADTW with penalty 1 = 4.
          Problem two("rebind");
          two.set_data(Data(std::vector<std::vector<data_t>>{ { 0.0, 0.0 }, { 0.0, 1.0, 2.0 } },
                            std::vector<std::string>{ "x", "y" }));
          two.set_band(-1);
          const double standard = two.dtw_function()(two.series(0), two.series(1));
          core::DTWVariantParams adtw;
          adtw.variant = core::DTWVariant::ADTW;
          adtw.adtw_penalty = 1.0;
          two.set_variant(adtw);
          return obs(standard, two.dtw_function()(two.series(0), two.series(1))); },
        "3 4 " },
      { "fast_pam writes its result back", false, false,
        [](Problem &p, const fs::path &, bool) {
          const auto result = fast_pam(p, 2); // the cluster numbering is arbitrary, the partition is not
          std::vector<index_t> medoid_of_point;
          for (const index_t label : p.labels())
            medoid_of_point.push_back(p.medoids()[static_cast<std::size_t>(label)]);
          char s0[16];
          std::snprintf(s0, sizeof s0, "%.6f", scores::silhouette(p)[0]);
          return obs(result.medoid_indices == p.medoids(), result.labels == p.labels(),
                     ids(medoid_of_point)) + s0; },
        "1 1 1,1,1,4,4,4 0.915385" },
    };
    DTWC_POP_NO_DEPRECATED

    int leaf = 0;
    for (const Row &row : rows)
      for (const bool v1 : { false, true }) {
        if (v1 && !row.has_v1) continue;
        INFO(row.entry << (v1 ? ", 1.x spelling" : ", 2.0 name"));
        const auto dir = root.make_leaf("r" + std::to_string(leaf++));
        Problem problem = make_f22_problem(dir, row.clustered);
        CHECK(row.observe(problem, dir, v1) == row.oracle);
      }
  } // root removes the whole validated fixture tree
  CHECK_FALSE(fs::exists(configured_root));
}
