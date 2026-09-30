/**
 * @file test_problem_loud_contracts.cpp
 * @brief FX-3 / FX-4 residue: Problem requests that were silently wrong, or
 *        unsound, now fail with a typed error that names what to change.
 *
 * @details
 *   - O-06: set_max_iter / set_n_repetitions below 1 reported a clustering after
 *     zero iterations; the setters now refuse, and Lloyd refuses the value the
 *     deprecated public fields can still hold.
 *   - FX-3: read_distance_matrix kept a matrix of another size (later discarded
 *     and recomputed at the first lookup) and loaded nothing from an empty file.
 *   - F25: get_name / p_vec guarded storage they do not own only with assert, so
 *     a Release build read past an empty vector.
 *   - B-05 (POSIX): a write that fails after the file was opened (full disk,
 *     quota) left a truncated Tier-1 or run artefact behind a reported success.
 *     RLIMIT_FSIZE reproduces it in-process, restored on scope exit: a forked
 *     child would enter OpenMP again, which libgomp does not support after fork.
 */

#include <dtwc.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <sys/resource.h>
#endif

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

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
namespace fs = std::filesystem;
using dtwc::test_support::ScratchDirectory;

namespace {

dtwc::Data named(std::vector<std::vector<double>> series, std::vector<std::string> names)
{
  return dtwc::Data(std::move(series), std::move(names));
}

/// Six short series in two obvious groups, named s0..s5.
dtwc::Problem six_series(std::string name)
{
  dtwc::Problem prob(std::move(name));
  prob.set_data(named({ { 0, 1, 2, 1 }, { 0, 1, 2, 2 }, { 0, 1, 1, 1 },
                        { 9, 8, 9, 9 }, { 9, 9, 8, 9 }, { 8, 9, 9, 9 } },
                      { "s0", "s1", "s2", "s3", "s4", "s5" }));
  prob.set_verbose(false);
  return prob;
}

void write_text(const fs::path &path, std::string_view text)
{
  std::ofstream out(path, std::ios::binary);
  out << text;
}

std::string read_text(const fs::path &path)
{
  std::ifstream in(path, std::ios::binary);
  return { std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
}

} // namespace

TEST_CASE("O-06: the iteration setters refuse n < 1 and keep the value they had",
          "[problem][o06]")
{
  auto prob = six_series("o06_setters");
  prob.set_max_iter(7);
  prob.set_n_repetitions(3);
  for (const int bad : { 0, -1, std::numeric_limits<int>::min() }) {
    CAPTURE(bad);
    REQUIRE_THROWS_MATCHES(
      prob.set_max_iter(bad), dtwc::InvalidInput,
      MessageMatches(ContainsSubstring("Problem::set_max_iter: max_iter must be at "
                                       "least 1; got " + std::to_string(bad))));
    REQUIRE_THROWS_MATCHES(
      prob.set_n_repetitions(bad), dtwc::InvalidInput,
      MessageMatches(ContainsSubstring("Problem::set_n_repetitions: n_repetitions "
                                       "must be at least 1; got " + std::to_string(bad))));
  }
  CHECK(prob.max_iter() == 7);
  CHECK(prob.n_repetitions() == 3);

  prob.set_max_iter(1);
  prob.set_n_repetitions(1);
  CHECK(prob.max_iter() == 1);
  CHECK(prob.n_repetitions() == 1);
}

TEST_CASE("O-06: Lloyd refuses zero iterations held by the deprecated field",
          "[problem][o06][lloyd]")
{
  auto prob = six_series("o06_lloyd");
  prob.set_n_clusters(2);
  DTWC_PUSH_NO_DEPRECATED
  prob.maxIter = 0; // the one route left around set_max_iter
  DTWC_POP_NO_DEPRECATED
  REQUIRE_THROWS_MATCHES(prob.cluster_by_kmedoids_lloyd(), dtwc::InvalidInput,
                         MessageMatches(ContainsSubstring(
                           "Lloyd k-medoids requires max_iter >= 1.")));
  CHECK(prob.labels().empty()); // the refused run published nothing

  // Control: the same Problem clusters once the field holds a valid count.
  DTWC_PUSH_NO_DEPRECATED
  prob.maxIter = 5;
  DTWC_POP_NO_DEPRECATED
  REQUIRE_NOTHROW(prob.cluster_by_kmedoids_lloyd());
  CHECK(prob.last_iterations() >= 1);
}

TEST_CASE("FX-3: read_distance_matrix refuses a matrix of another size and an empty file",
          "[problem][fx3][io]")
{
  ScratchDirectory dir{ "fx3_read_matrix" };
  auto prob = six_series("fx3_read");
  prob.fill_distance_matrix();
  const double before = prob.dist_by_ind(0, 3);

  const auto four = dir.path / "four_by_four.csv";
  write_text(four, "0,1,2,3\n1,0,1,2\n2,1,0,1\n3,2,1,0\n");
  REQUIRE_THROWS_MATCHES(
    prob.read_distance_matrix(four), dtwc::InvalidInput,
    MessageMatches(ContainsSubstring("'" + four.string() + "' has 4 rows, but this "
                                     "Problem holds 6 series")));
  // The rejected file left the filled matrix as it was.
  CHECK(prob.is_distance_matrix_filled());
  CHECK(prob.dist_by_ind(0, 3) == before);

  const auto empty = dir.path / "empty.csv";
  write_text(empty, "");
  REQUIRE_THROWS_MATCHES(
    prob.read_distance_matrix(empty), dtwc::InvalidInput,
    MessageMatches(ContainsSubstring("'" + empty.string() + "' has 0 rows, but this "
                                     "Problem holds 6 series")));
  CHECK(prob.dist_by_ind(0, 3) == before);

  // The matching size loads (independent oracle: the file's own numbers).
  const auto six = dir.path / "six_by_six.csv";
  write_text(six, "0,1,2,3,4,5\n1,0,1,2,3,4\n2,1,0,1,2,3\n"
                  "3,2,1,0,1,2\n4,3,2,1,0,1\n5,4,3,2,1,0\n");
  prob.read_distance_matrix(six);
  CHECK(prob.dist_by_ind(0, 3) == 3.0);
  CHECK(prob.dist_by_ind(5, 1) == 4.0);

  // A Problem without series still takes a matrix of any size, as before.
  dtwc::Problem bare("fx3_bare");
  CHECK_NOTHROW(bare.read_distance_matrix(four));
}

TEST_CASE("F25: get_name and p_vec refuse storage they do not own, in Release too",
          "[problem][f25]")
{
  const std::vector<std::vector<double>> owner{ { 0, 1, 2 }, { 1, 2, 3 }, { 7, 8, 9 } };

  SECTION("a non-owning view")
  {
    std::vector<std::span<const double>> spans(owner.begin(), owner.end());
    dtwc::Problem view("f25_view");
    view.set_view_data(dtwc::Data(std::move(spans), { "a", "b", "c" }, 1));
    REQUIRE_THROWS_MATCHES(
      (void)view.get_name(0), dtwc::InvalidInput,
      MessageMatches(ContainsSubstring("Problem::get_name")
                     && ContainsSubstring("non-owning view")
                     && ContainsSubstring("series_name(i)")));
    const dtwc::Problem &const_view = view;
    REQUIRE_THROWS_AS((void)const_view.get_name(1), dtwc::InvalidInput);
    REQUIRE_THROWS_MATCHES(
      (void)view.p_vec(0), dtwc::InvalidInput,
      MessageMatches(ContainsSubstring("Problem::p_vec")
                     && ContainsSubstring("series(i)")));
    REQUIRE_THROWS_AS((void)const_view.p_vec(2), dtwc::InvalidInput);
    // The accessors that read every storage mode still do.
    CHECK(view.series_name(2) == "c");
    CHECK(view.series(2)[1] == 8.0);
  }

  SECTION("a Float32 store owns names but no Float64 values")
  {
    dtwc::Problem f32("f25_f32");
    f32.set_data(dtwc::Data(std::vector<std::vector<float>>{ { 0.f, 1.f }, { 2.f, 3.f } },
                            std::vector<std::string>{ "a", "b" }));
    CHECK(f32.get_name(1) == "b");
    REQUIRE_THROWS_MATCHES(
      (void)f32.p_vec(0), dtwc::InvalidInput,
      MessageMatches(ContainsSubstring("Problem::p_vec") && ContainsSubstring("Float32")));
  }

  SECTION("owned Float64 series keep both mutable accessors")
  {
    auto prob = six_series("f25_heap");
    prob.get_name(0) = "renamed";
    CHECK(prob.series_name(0) == "renamed");
    CHECK(prob.p_vec(3)[0] == 9.0);
  }
}

TEST_CASE("F25: the Problem writers name series in every storage mode",
          "[problem][f25][io]")
{
  // They read names through get_name, which a view does not own: the writers
  // read past an empty vector for a view (set_view_data, an mmap store).
  ScratchDirectory dir{ "f25_writers" };
  const std::vector<std::vector<double>> owner{ { 0, 1, 2 }, { 0, 1, 3 }, { 9, 8, 9 } };
  std::vector<std::span<const double>> spans(owner.begin(), owner.end());
  dtwc::Problem view("f25_writer");
  view.set_view_data(dtwc::Data(std::move(spans), { "alpha", "beta", "gamma" }, 1));
  view.set_output_folder(dir.path);
  view.set_n_clusters(2);
  view.centroids_ind = { 0, 2 };
  view.clusters_ind = { 0, 0, 1 };
  view.write_clusters();
  const auto text = read_text(dir.path / "f25_writer_Nc_2.csv");
  CHECK_THAT(text, ContainsSubstring("alpha,gamma"));
  CHECK_THAT(text, ContainsSubstring("beta,alpha"));
  CHECK_THAT(text, ContainsSubstring("gamma,gamma"));
}

#ifndef _WIN32
namespace {

/// Lowers RLIMIT_FSIZE's soft limit and ignores SIGXFSZ for its lifetime, so a
/// write past `bytes` fails with EFBIG after the file was opened, as a full disk
/// or a quota does. Only regular files are limited: the test runner's pipes are
/// not.
class FileSizeLimit
{
  rlimit previous_{};
  void (*previous_handler_)(int) = SIG_DFL;

public:
  explicit FileSizeLimit(rlim_t bytes)
  {
    REQUIRE(getrlimit(RLIMIT_FSIZE, &previous_) == 0);
    previous_handler_ = std::signal(SIGXFSZ, SIG_IGN);
    rlimit lowered = previous_;
    lowered.rlim_cur = bytes;
    REQUIRE(setrlimit(RLIMIT_FSIZE, &lowered) == 0);
  }
  ~FileSizeLimit()
  {
    setrlimit(RLIMIT_FSIZE, &previous_);
    std::signal(SIGXFSZ, previous_handler_);
  }
  FileSizeLimit(const FileSizeLimit &) = delete;
  FileSizeLimit &operator=(const FileSizeLimit &) = delete;
};

} // namespace

TEST_CASE("B-05: Result::save reports a write that fails after the file opened",
          "[api][tier1][b05][io]")
{
  // Six series named by 200-character file stems: labels.csv (> 1.2 kB) and
  // silhouettes.csv overflow a 1024-byte limit; medoids.csv and the 6 x 6
  // matrix fit. An open-only check returned normally with both truncated.
  ScratchDirectory dir{ "b05_save" };
  const auto input = dir.path / "series";
  fs::create_directories(input);
  const std::string stem(200, 'x');
  for (int s = 0; s < 6; ++s) {
    std::ostringstream text;
    text << "t,value\n";
    for (int t = 0; t < 8; ++t) text << t << ',' << (s < 3 ? t % 3 : 9 - t % 2) << '\n';
    write_text(input / (stem + "_" + std::to_string(s) + ".csv"), text.str());
  }
  const auto result = dtwc::cluster(dtwc::load(input, 1, 1), 2, "pam", -1, "cpu");
  const auto out = dir.path / "out";
  fs::create_directories(out);

  {
    FileSizeLimit limit(1024);
    REQUIRE_THROWS_MATCHES(result.save(out), dtwc::IOError,
                           MessageMatches(ContainsSubstring("series_labels.csv")
                                          && ContainsSubstring("incomplete")));
  }
  // Control: the same save succeeds without the limit.
  REQUIRE_NOTHROW(result.save(out));
  CHECK(fs::file_size(out / "series_labels.csv") > 1024);
}

TEST_CASE("B-05: a run artefact write that fails after the file opened is an IOError",
          "[problem][b05][io]")
{
  // cluster_and_process() writes the per-repetition medoids first; its writer
  // checked only the open, so the truncation went unnoticed. 600-character
  // names put one medoid line above the 1024-byte limit.
  ScratchDirectory dir{ "b05_artefacts" };
  auto prob = six_series("b05");
  for (int i = 0; i < 6; ++i)
    prob.get_name(static_cast<std::size_t>(i)) = std::string(600, 'n') + std::to_string(i);
  prob.set_output_folder(dir.path);
  prob.set_n_clusters(2);

  FileSizeLimit limit(1024);
  REQUIRE_THROWS_MATCHES(prob.cluster_and_process(), dtwc::IOError,
                         MessageMatches(ContainsSubstring("b05medoids_rep_0.csv")));
}
#endif
