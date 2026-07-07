/**
 * @file cpp_conformance.cpp
 * @brief C++ route of the cross-language conformance fixture (Phase 2 Task 2.4).
 *
 * This is the PERMANENT parity gate for docs/api-contract-2.0.md §9. It drives
 * the LIVE Tier-2 public pipeline in-process:
 *
 *     DataLoader::load()            (load the recorded CSV, the exact CLI path)
 *   -> Problem::set_band(3)         (fixed Sakoe-Chiba band -> banded DTW kernel)
 *   -> Problem::fillDistanceMatrix()
 *   -> dtwc::fast_pam(prob, 3, 100) (FastPAM k=3, writes labels/medoids back)
 *   -> scores::silhouette / davies_bouldin / dunn
 *
 * The C++ route is the REFERENCE producer: on first run (or with
 * DTWC_CONFORMANCE_REGEN=1) it records its canonical output into
 * tests/conformance/conformance_reference.txt; on every run it recomputes and
 * asserts digit-identical labels/medoids and scores within 1e-12 relative. The
 * Python, MATLAB and CLI routes assert against the SAME recorded reference, so
 * all four agree bit-for-bit on the clustering and to 1e-12 on the scores.
 *
 * Determinism: the recorded dataset (tests/conformance/data/
 * generate_conformance_data.py) is engineered so the FastPAM optimum is unique
 * and init-independent; the result is canonicalised (sorted medoid SET; each
 * point labelled by the rank of its assigned medoid) so RNG-dependent medoid
 * array order / cluster-id numbering cannot make identical clusterings compare
 * unequal.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

using Catch::Matchers::WithinRel;
namespace fs = std::filesystem;

namespace {

// The fixed pipeline parameters — SHARED verbatim by all four routes (must match
// tests/conformance/conformance.toml and the Python/MATLAB runners).
constexpr int kNClusters = 3;
constexpr int kBand = 3;
constexpr int kMaxIter = 100;
constexpr unsigned kSeed = 29; // dtwc::randGenerator default; reset for a literal fixed seed.

/// Repo-root/tests/conformance — located from the compile-time DTWC_TEST_DATA_DIR
/// (== "<repo>/data"), never a runtime-relative path (Global Constraint #1; same
/// pattern as tests/unit/test_supply_chain_pinning.cpp).
fs::path conformance_dir()
{
  return fs::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "conformance";
}

fs::path data_csv() { return conformance_dir() / "data" / "conformance_series.csv"; }
fs::path reference_file() { return conformance_dir() / "conformance_reference.txt"; }

/// 17 significant digits: round-trips an IEEE-754 double exactly across C++
/// operator>>, Python float() and MATLAB str2double.
std::string fmt17(double x)
{
  std::ostringstream os;
  os << std::setprecision(17) << x;
  return os.str();
}

struct CanonicalResult
{
  std::vector<int> labels;   ///< canonical 0-based label per series (rank of its medoid)
  std::vector<int> medoids;  ///< canonical medoid SET: series indices, sorted ascending
  double silhouette{};       ///< mean silhouette
  double davies_bouldin{};
  double dunn{};
};

/// Canonicalise a raw FastPAM result so RNG-dependent ordering cannot distinguish
/// identical clusterings: sort the medoid SET, and label each point by the rank of
/// its assigned medoid within that sorted set. `raw_labels[i]` indexes into
/// `raw_medoids` (FastPAM contract), `raw_medoids[m]` is a series index.
void canonicalise(const std::vector<int>& raw_labels,
                  const std::vector<int>& raw_medoids,
                  std::vector<int>& out_labels,
                  std::vector<int>& out_medoids)
{
  out_medoids = raw_medoids;
  std::sort(out_medoids.begin(), out_medoids.end());

  auto rank_of = [&](int series_idx) {
    auto it = std::lower_bound(out_medoids.begin(), out_medoids.end(), series_idx);
    return static_cast<int>(it - out_medoids.begin());
  };

  out_labels.resize(raw_labels.size());
  for (size_t i = 0; i < raw_labels.size(); ++i) {
    const int assigned_series = raw_medoids.at(static_cast<size_t>(raw_labels[i]));
    out_labels[i] = rank_of(assigned_series);
  }
}

/// Run the fixed conformance pipeline through the live C++ public surface.
CanonicalResult run_pipeline()
{
  // Load the recorded CSV via the exact code path the CLI uses (DataLoader,
  // no header/id columns), so C++ and CLI ingest identical series.
  dtwc::DataLoader dl{ data_csv() };
  dl.startColumn(0).startRow(0);

  dtwc::Problem prob{ "conformance" };
  prob.set_data(dl.load());
  prob.set_band(kBand);
  prob.fillDistanceMatrix();

  // Literal fixed seed. The data is init-independent, but resetting the global
  // RNG makes "fast_pam with a fixed seed" (Task 2.4) exact in this route.
  dtwc::randGenerator.seed(kSeed);
  const auto raw = dtwc::fast_pam(prob, kNClusters, kMaxIter);

  CanonicalResult r;
  canonicalise(raw.labels, raw.medoid_indices, r.labels, r.medoids);

  // Scores read prob state (written back by fast_pam). Silhouette score = the
  // mean of the per-point vector (the Tier-1 "silhouette" contract, §1.4). All
  // three metrics are label-permutation invariant, so no canonicalisation needed.
  const auto sil = dtwc::scores::silhouette(prob);
  r.silhouette = std::reduce(sil.begin(), sil.end(), 0.0) / static_cast<double>(sil.size());
  r.davies_bouldin = dtwc::scores::davies_bouldin(prob);
  r.dunn = dtwc::scores::dunn(prob);
  return r;
}

void write_reference(const CanonicalResult& r)
{
  std::ofstream out(reference_file());
  if (!out)
    throw std::runtime_error("cannot write reference: " + reference_file().string());

  out << "# DTWC++ 2.0 cross-language conformance reference (Phase 2 Task 2.4).\n"
      << "# Recorded by the C++ route (tests/conformance/cpp_conformance.cpp) which\n"
      << "# runs the LIVE pipeline: DataLoader -> set_band(3) -> fillDistanceMatrix\n"
      << "# -> fast_pam(k=3, seed=29) -> silhouette/davies_bouldin/dunn. The Python,\n"
      << "# MATLAB and CLI routes assert digit-identical labels/medoids and scores\n"
      << "# within 1e-12 rel against THIS file.\n"
      << "# REGENERATE: run this test with env DTWC_CONFORMANCE_REGEN=1 (rebuild the\n"
      << "# CSV first only if data/generate_conformance_data.py changed). Values are\n"
      << "# canonical: medoids = sorted series indices; labels[i] = rank of series i's\n"
      << "# medoid; silhouette = mean of the per-point vector.\n";
  out << "labels";
  for (int v : r.labels) out << "," << v;
  out << "\n";
  out << "medoids";
  for (int v : r.medoids) out << "," << v;
  out << "\n";
  out << "silhouette," << fmt17(r.silhouette) << "\n";
  out << "davies_bouldin," << fmt17(r.davies_bouldin) << "\n";
  out << "dunn," << fmt17(r.dunn) << "\n";
}

CanonicalResult read_reference()
{
  std::ifstream in(reference_file());
  if (!in)
    throw std::runtime_error("cannot read reference: " + reference_file().string());

  CanonicalResult r;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string key, tok;
    std::getline(ls, key, ',');
    if (key == "labels" || key == "medoids") {
      std::vector<int>& dst = (key == "labels") ? r.labels : r.medoids;
      while (std::getline(ls, tok, ',')) dst.push_back(std::stoi(tok));
    } else if (key == "silhouette") {
      std::getline(ls, tok, ','); r.silhouette = std::stod(tok);
    } else if (key == "davies_bouldin") {
      std::getline(ls, tok, ','); r.davies_bouldin = std::stod(tok);
    } else if (key == "dunn") {
      std::getline(ls, tok, ','); r.dunn = std::stod(tok);
    }
  }
  return r;
}

} // namespace

// Drives dtwc::fast_pam() Tier-2 + scores::* on the recorded conformance dataset.
TEST_CASE("Cross-language conformance: C++ route matches recorded reference",
          "[Phase2][conformance]")
{
  REQUIRE(fs::exists(data_csv()));

  const CanonicalResult live = run_pipeline();

  const bool regen =
    !fs::exists(reference_file())
    || (std::getenv("DTWC_CONFORMANCE_REGEN") != nullptr
        && std::string(std::getenv("DTWC_CONFORMANCE_REGEN")) == "1");

  if (regen) {
    write_reference(live);
    WARN("Recorded conformance reference -> " << reference_file().string());
  }

  const CanonicalResult ref = read_reference();

  // Labels + medoids: DIGIT-IDENTICAL (canonical integer vectors).
  REQUIRE(live.labels == ref.labels);
  REQUIRE(live.medoids == ref.medoids);
  REQUIRE(static_cast<int>(live.medoids.size()) == kNClusters);

  // Scores: equal to 1e-12 relative.
  CHECK_THAT(live.silhouette, WithinRel(ref.silhouette, 1e-12));
  CHECK_THAT(live.davies_bouldin, WithinRel(ref.davies_bouldin, 1e-12));
  CHECK_THAT(live.dunn, WithinRel(ref.dunn, 1e-12));
}
