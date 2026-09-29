/**
 * @file test_names.cpp
 * @brief Pins every string <-> enum table (base/names.hpp) against the spellings
 *        the front ends accept today.
 *
 * @details The oracle is transcribed from the parsers the tables replace, not from
 * the tables: dtwc_cl's CLI11 CheckedTransformer maps (dtwc_cl.cpp, the options
 * block of run_cli_main) and MATLAB's hand-written parsers (dtwc_mex.cpp,
 * parse_missing_strategy and its siblings). Per table it checks the
 * canonical name of every value, that every alias reads as its value, that the
 * table holds nothing else, that case is ignored, and the error text.
 *
 * @date 24 Sep 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "algorithms/hierarchical.hpp"
#include "algorithms/one_batch_pam.hpp"
#include "base/error.hpp"
#include "base/names.hpp"
#include "cli/config.hpp"
#include "core/dtw_options.hpp"
#include "core/storage.hpp"
#include "enums/Method.hpp"
#include "enums/Solver.hpp"

#include <cctype>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string upper(std::string text)
{
  for (char &c : text) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return text;
}

std::string capitalised(std::string text)
{
  if (!text.empty()) text.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(text.front())));
  return text;
}

/// `canonical`: each value with its canonical name, in table order.
/// `aliases`: every other accepted spelling onto its canonical name, as CLI11's maps list them.
template <class E, std::size_t N>
void check_table(const dtwc::Name<E> (&table)[N], const std::vector<std::pair<std::string, E>> &canonical,
                 const std::map<std::string, std::string> &aliases)
{
  std::map<std::string, E> value_of;
  std::string valid;
  for (const auto &[text, value] : canonical) {
    INFO("canonical name '" << text << "'");
    CHECK(dtwc::name_of(table, value) == text);
    CHECK(dtwc::parse_name(table, text, "thing") == value);
    value_of[text] = value;
    valid += (valid.empty() ? "" : ", ") + text;
  }

  std::vector<std::string> spellings;
  for (const auto &entry : canonical) spellings.push_back(entry.first);
  for (const auto &[alias, target] : aliases) {
    INFO("alias '" << alias << "' -> '" << target << "'");
    REQUIRE(value_of.count(target) == 1);
    CHECK(dtwc::parse_name(table, alias, "thing") == value_of[target]);
    spellings.push_back(alias);
  }
  // Every spelling above parses, so a table of this size holds exactly them.
  CHECK(N == spellings.size());

  for (const std::string &spelling : spellings) {
    INFO("spelling '" << spelling << "'");
    const E value = dtwc::parse_name(table, spelling, "thing");
    CHECK(dtwc::parse_name(table, upper(spelling), "thing") == value);
    CHECK(dtwc::parse_name(table, capitalised(spelling), "thing") == value);
  }

  CHECK_THROWS_MATCHES(dtwc::parse_name(table, "no-such-name", "thing"), dtwc::InvalidInput,
                       Catch::Matchers::Message("unknown thing 'no-such-name'. Valid: " + valid + "."));
}

} // namespace

TEST_CASE("Method and Solver tables", "[names]")
{
  using dtwc::Method;
  // MATLAB's parse_method also read 'pam' and 'auto' as Kmedoids, i.e. Lloyd: not names of it.
  check_table(dtwc::method_names,
              { { "kmedoids", Method::Kmedoids }, { "mip", Method::MIP }, { "lrcore", Method::LRCore },
                { "tadpole", Method::TADPole } },
              {});
  check_table(dtwc::solver_names, { { "highs", dtwc::Solver::HiGHS }, { "gurobi", dtwc::Solver::Gurobi } }, {});
}

TEST_CASE("distance tables: metric, variant, mv-mode, missing strategy", "[names]")
{
  using namespace dtwc::core;
  check_table(metric_names, { { "l1", MetricType::L1 }, { "squared_euclidean", MetricType::SquaredL2 } },
              { { "sqeuclidean", "squared_euclidean" }, { "l2sq", "squared_euclidean" } });
  check_table(variant_names,
              { { "standard", DTWVariant::Standard }, { "ddtw", DTWVariant::DDTW }, { "wdtw", DTWVariant::WDTW },
                { "adtw", DTWVariant::ADTW }, { "softdtw", DTWVariant::SoftDTW }, { "msm", DTWVariant::MSM },
                { "twe", DTWVariant::TWE } },
              { { "soft-dtw", "softdtw" } });
  check_table(mv_mode_names, { { "dependent", MVMode::Dependent }, { "independent", MVMode::Independent } }, {});
  check_table(missing_strategy_names,
              { { "error", MissingStrategy::Error }, { "zero_cost", MissingStrategy::ZeroCost },
                { "arow", MissingStrategy::AROW }, { "interpolate", MissingStrategy::Interpolate } },
              { { "zero-cost", "zero_cost" }, { "zerocost", "zero_cost" } });

  // L2 is a kernel metric that no front end spells: asking for its name is an error, not "".
  CHECK_THROWS_AS(dtwc::name_of(metric_names, MetricType::L2), dtwc::InvalidInput);
}

TEST_CASE("dtype, linkage and OneBatchPAM weighting tables", "[names]")
{
  using dtwc::core::Precision;
  check_table(dtwc::core::precision_names, { { "float32", Precision::Float32 }, { "float64", Precision::Float64 } },
              { { "f32", "float32" }, { "fp32", "float32" }, { "float", "float32" }, { "f64", "float64" },
                { "fp64", "float64" }, { "double", "float64" } });
  using dtwc::algorithms::Linkage;
  check_table(dtwc::algorithms::linkage_names,
              { { "single", Linkage::Single }, { "complete", Linkage::Complete }, { "average", Linkage::Average } },
              {});
  using dtwc::algorithms::OneBatchWeighting;
  check_table(dtwc::algorithms::one_batch_weighting_names,
              { { "uniform", OneBatchWeighting::Uniform }, { "debiased", OneBatchWeighting::Debiased },
                { "nniw", OneBatchWeighting::NearestNeighbor } },
              { { "debias", "debiased" } });
}

TEST_CASE("config tables: cluster method and GPU precision", "[names]")
{
  using dtwc::ClusterMethod;
  check_table(dtwc::cluster_method_names,
              { { "auto", ClusterMethod::Auto }, { "pam", ClusterMethod::PAM }, { "onebatch", ClusterMethod::OneBatch },
                { "clara", ClusterMethod::CLARA }, { "kmedoids", ClusterMethod::Kmedoids },
                { "mip", ClusterMethod::MIP }, { "lrcore", ClusterMethod::LRCore },
                { "tadpole", ClusterMethod::TADPole }, { "hierarchical", ClusterMethod::Hierarchical } },
              { { "obp", "onebatch" }, { "lr", "lrcore" }, { "hclust", "hierarchical" } });
  check_table(dtwc::gpu_precision_names, { { "auto", 0 }, { "fp32", 1 }, { "fp64", 2 } },
              { { "float32", "fp32" }, { "f32", "fp32" }, { "float", "fp32" }, { "float64", "fp64" },
                { "f64", "fp64" }, { "double", "fp64" } });
}

TEST_CASE("parse_name matches ASCII case only", "[names]")
{
  // The Python and CLI error text both start "unknown method" (pytest matches on it).
  CHECK_THROWS_MATCHES(dtwc::parse_name(dtwc::cluster_method_names, "k-means", "method"), dtwc::InvalidInput,
                       Catch::Matchers::Message("unknown method 'k-means'. Valid: auto, pam, onebatch, clara, "
                                                "kmedoids, mip, lrcore, tadpole, hierarchical."));
  // No trimming and no '-' / '_' folding: an alias exists only because a table lists it.
  CHECK_THROWS_AS(dtwc::parse_name(dtwc::cluster_method_names, " pam", "method"), dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_name(dtwc::cluster_method_names, "one_batch", "method"), dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_name(dtwc::core::missing_strategy_names, "zero cost", "missing strategy"),
                  dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_name(dtwc::core::metric_names, "squared-euclidean", "metric"),
                  dtwc::InvalidInput);
}
