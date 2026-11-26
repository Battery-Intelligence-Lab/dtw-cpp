/** @file test_cdist_and_workflow.cpp
 *  @brief Unit tests for cross-collection dtwc::cdist and for the files and the checkpoint
 *         cluster(Dataset, Config) writes.
 *  @author Volkan Kumtepeli
 */

#include <dtwc.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using dtwc::test_support::ScratchDirectory;

namespace {

std::vector<std::vector<double>> make_equal_series(std::size_t n, std::size_t len, double offset)
{
  std::vector<std::vector<double>> out(n, std::vector<double>(len));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < len; ++t) {
      out[i][t] = offset + 0.3 * static_cast<double>(i) + std::sin(0.25 * static_cast<double>(t + i));
    }
  }
  return out;
}

} // namespace

TEST_CASE("cdist matches per-pair dtw oracle across rectangular, ragged, banded, variants, and multivariate",
          "[api][cdist]")
{
  // N_Y = 19 deliberately avoids multiples of SIMD lane width (8 or 16) to exercise tail-panel padding.
  const auto X = make_equal_series(7, 24, 0.0);
  const auto Y = make_equal_series(19, 24, 2.5);

  SECTION("equal-length standard L1 and SquaredL2, unbanded and banded")
  {
    for (const auto metric : { dtwc::core::MetricType::L1, dtwc::core::MetricType::SquaredL2 }) {
      for (const int band : { -1, 4 }) {
        dtwc::core::DistanceConfig dcfg;
        dcfg.metric = metric;
        dcfg.band = band;
        const auto D = dtwc::distance::cdist(X, Y, dcfg);
        REQUIRE(D.size() == X.size() * Y.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
          for (std::size_t j = 0; j < Y.size(); ++j) {
            const double ref = dtwc::distance::dtw(X[i], Y[j], band, metric);
            REQUIRE_THAT(D[i * Y.size() + j], Catch::Matchers::WithinRel(ref, 1e-12));
          }
        }
      }
    }
  }

  SECTION("ragged series unbanded and feasible Sakoe-Chiba band")
  {
    const std::vector<std::vector<double>> X_rag = {
      { 0.0, 1.0, 2.0, 1.5, 0.5 },
      { 3.0, 3.2, 2.8, 3.1, 3.0, 2.9, 3.3 },
      { -1.0, -0.5, 0.0, 0.5 },
    };
    const std::vector<std::vector<double>> Y_rag = {
      { 0.1, 0.9, 2.1, 1.4, 0.4, 0.2 },
      { 3.1, 3.0, 2.9, 3.2, 3.1 },
      { -0.8, -0.4, 0.1, 0.4, 0.6 },
      { 5.0, 5.5, 6.0, 5.8, 5.2, 5.1 },
      { 1.0, 1.2, 1.1, 0.9 },
    };
    for (const int band : { -1, 3 }) {
      dtwc::core::DistanceConfig dcfg;
      dcfg.band = band;
      const auto D = dtwc::distance::cdist(X_rag, Y_rag, dcfg);
      REQUIRE(D.size() == X_rag.size() * Y_rag.size());
      for (std::size_t i = 0; i < X_rag.size(); ++i) {
        for (std::size_t j = 0; j < Y_rag.size(); ++j) {
          const double ref = dtwc::distance::dtw(X_rag[i], Y_rag[j], band);
          REQUIRE_THAT(D[i * Y_rag.size() + j], Catch::Matchers::WithinRel(ref, 1e-12));
        }
      }
    }

    // Band narrower than |len(x) - len(y)| = |7 - 4| = 3 must throw InvalidInput.
    dtwc::core::DistanceConfig narrow;
    narrow.band = 2;
    REQUIRE_THROWS_AS(dtwc::distance::cdist(X_rag, Y_rag, narrow), dtwc::InvalidInput);
    // A band below -1 is refused, as Problem::set_band refuses it, not run as full DTW.
    narrow.band = -2;
    REQUIRE_THROWS_AS(dtwc::distance::cdist(X_rag, Y_rag, narrow), dtwc::InvalidInput);
  }

  SECTION("DDTW and WDTW variants")
  {
    dtwc::core::DistanceConfig ddtw_cfg;
    ddtw_cfg.variant.variant = dtwc::core::DTWVariant::DDTW;
    ddtw_cfg.band = 3;
    const auto D_ddtw = dtwc::distance::cdist(X, Y, ddtw_cfg);
    for (std::size_t i = 0; i < X.size(); ++i) {
      for (std::size_t j = 0; j < Y.size(); ++j) {
        const double ref = dtwc::distance::ddtw(X[i], Y[j], 3);
        REQUIRE_THAT(D_ddtw[i * Y.size() + j], Catch::Matchers::WithinRel(ref, 1e-12));
      }
    }

    dtwc::core::DistanceConfig wdtw_cfg;
    wdtw_cfg.variant.variant = dtwc::core::DTWVariant::WDTW;
    wdtw_cfg.variant.wdtw_g = 0.08;
    const auto D_wdtw = dtwc::distance::cdist(X, Y, wdtw_cfg);
    for (std::size_t i = 0; i < X.size(); ++i) {
      for (std::size_t j = 0; j < Y.size(); ++j) {
        const double ref = dtwc::distance::wdtw(X[i], Y[j], -1, 0.08);
        REQUIRE_THAT(D_wdtw[i * Y.size() + j], Catch::Matchers::WithinRel(ref, 1e-12));
      }
    }
  }

  SECTION("multivariate Dependent and Independent match Problem::fill_distance_matrix")
  {
    // 24 flat values = 8 timesteps x 3 channels
    std::vector<std::string> x_names(X.size()), y_names(Y.size());
    for (std::size_t i = 0; i < X.size(); ++i) x_names[i] = "x" + std::to_string(i);
    for (std::size_t j = 0; j < Y.size(); ++j) y_names[j] = "y" + std::to_string(j);
    dtwc::Data X_mv(std::vector<std::vector<double>>(X), std::move(x_names), 3);
    dtwc::Data Y_mv(std::vector<std::vector<double>>(Y), std::move(y_names), 3);

    std::vector<std::vector<double>> all_series;
    std::vector<std::string> all_names;
    for (std::size_t i = 0; i < X.size(); ++i) {
      all_series.push_back(X[i]);
      all_names.push_back("x" + std::to_string(i));
    }
    for (std::size_t j = 0; j < Y.size(); ++j) {
      all_series.push_back(Y[j]);
      all_names.push_back("y" + std::to_string(j));
    }

    for (const auto mode : { dtwc::core::MVMode::Dependent,
                             dtwc::core::MVMode::Independent }) {
      dtwc::core::DistanceConfig mv_cfg;
      mv_cfg.ndim = 3;
      mv_cfg.variant.mv_mode = mode;
      mv_cfg.band = 2;
      const auto D_mv = dtwc::cdist(X_mv, Y_mv, mv_cfg);

      dtwc::Problem prob("mv_oracle");
      prob.set_data(dtwc::Data(std::vector<std::vector<double>>(all_series),
                               std::vector<std::string>(all_names), 3));
      prob.set_distance(mv_cfg);
      prob.fill_distance_matrix();

      for (std::size_t i = 0; i < X.size(); ++i) {
        for (std::size_t j = 0; j < Y.size(); ++j) {
          const double ref = prob.dist_by_ind(static_cast<dtwc::index_t>(i),
                                              static_cast<dtwc::index_t>(X.size() + j));
          REQUIRE_THAT(D_mv[i * Y.size() + j], Catch::Matchers::WithinRel(ref, 1e-12));
        }
      }
    }
  }

  SECTION("Float32 and mixed Float32/Float64 inputs")
  {
    std::vector<std::vector<float>> X_f32(X.size()), Y_f32(Y.size());
    std::vector<std::string> xn(X.size(), "x"), yn(Y.size(), "y");
    for (std::size_t i = 0; i < X.size(); ++i) {
      X_f32[i].resize(X[i].size());
      for (std::size_t t = 0; t < X[i].size(); ++t) X_f32[i][t] = static_cast<float>(X[i][t]);
    }
    for (std::size_t j = 0; j < Y.size(); ++j) {
      Y_f32[j].resize(Y[j].size());
      for (std::size_t t = 0; t < Y[j].size(); ++t) Y_f32[j][t] = static_cast<float>(Y[j][t]);
    }
    const auto D_f32 = dtwc::distance::cdist<float>(X_f32, Y_f32);
    REQUIRE(D_f32.size() == X.size() * Y.size());
    for (std::size_t i = 0; i < X.size(); ++i) {
      for (std::size_t j = 0; j < Y.size(); ++j) {
        const float ref = dtwc::distance::dtw(X_f32[i], Y_f32[j]);
        REQUIRE_THAT(D_f32[i * Y.size() + j], Catch::Matchers::WithinRel(static_cast<double>(ref), 1e-5));
      }
    }

    // Mixed precision: X in Float32, Y in Float64 -> widens X to Float64 cleanly.
    dtwc::Data X_data_f32(std::move(X_f32), std::move(xn), 1);
    dtwc::Data Y_data_f64(std::vector<std::vector<double>>(Y), std::move(yn), 1);
    const auto D_mixed = dtwc::cdist(X_data_f32, Y_data_f64);
    REQUIRE(D_mixed.size() == X.size() * Y.size());

    dtwc::Config f32_cfg;
    f32_cfg.matrix_dtype = dtwc::core::Precision::Float32;
    REQUIRE_THROWS_AS(dtwc::cdist(X_data_f32, Y_data_f64, f32_cfg), dtwc::InvalidInput);
    const auto D_out_f32 = dtwc::cdist_f32(X_data_f32, Y_data_f64, f32_cfg);
    REQUIRE(D_out_f32.size() == D_mixed.size());
    for (std::size_t idx = 0; idx < D_mixed.size(); ++idx)
      REQUIRE(D_out_f32[idx] == static_cast<float>(D_mixed[idx]));
  }

  SECTION("NaN rejected under Error strategy and accepted under ZeroCost")
  {
    auto X_nan = X;
    X_nan[1][3] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(dtwc::distance::cdist(X_nan, Y), dtwc::InvalidInput);

    dtwc::core::DistanceConfig zc;
    zc.missing = dtwc::core::MissingStrategy::ZeroCost;
    const auto D_zc = dtwc::distance::cdist(X_nan, Y, zc);
    REQUIRE(D_zc.size() == X.size() * Y.size());
    REQUIRE_THAT(D_zc[1 * Y.size() + 0],
                 Catch::Matchers::WithinRel(dtwc::distance::missing(X_nan[1], Y[0]), 1e-12));
  }

#ifdef DTWC_HAS_CUDA
  SECTION("GPU and CPU_GPU hybrid match CPU oracle when a CUDA device is present")
  {
    if (dtwc::gpu_available()) {
      std::vector<std::string> xn(X.size(), "x"), yn(Y.size(), "y");
      dtwc::Data Xd(std::vector<std::vector<double>>(X), std::move(xn));
      dtwc::Data Yd(std::vector<std::vector<double>>(Y), std::move(yn));
      dtwc::core::DistanceConfig dcfg;
      dcfg.band = 4;

      const auto D_cpu = dtwc::cdist(Xd, Yd, dcfg, dtwc::Device::CPU);
      const auto D_gpu_f64 = dtwc::cdist(Xd, Yd, dcfg, dtwc::Device::GPU, 0, dtwc::GpuPrecision::FP64);
      const auto D_hyb_f64 = dtwc::cdist(Xd, Yd, dcfg, dtwc::Device::CPU_GPU, 0, dtwc::GpuPrecision::FP64);
      const auto D_gpu_f32 = dtwc::cdist(Xd, Yd, dcfg, dtwc::Device::GPU, 0, dtwc::GpuPrecision::FP32);

      REQUIRE(D_gpu_f64.size() == D_cpu.size());
      REQUIRE(D_hyb_f64.size() == D_cpu.size());
      for (std::size_t idx = 0; idx < D_cpu.size(); ++idx) {
        REQUIRE_THAT(D_gpu_f64[idx], Catch::Matchers::WithinRel(D_cpu[idx], 1e-12));
        REQUIRE_THAT(D_hyb_f64[idx], Catch::Matchers::WithinRel(D_cpu[idx], 1e-12));
        REQUIRE_THAT(D_gpu_f32[idx], Catch::Matchers::WithinRel(D_cpu[idx], 1e-4));
      }
    }
  }
#endif
}

TEST_CASE("cluster(Dataset, Config) writes the files and checkpoint its Config names, and nothing else",
          "[api][cluster]")
{
  ScratchDirectory scratch("dtwc_cluster_config");
  const fs::path csv_path = scratch.path / "series.csv";
  {
    std::ofstream out(csv_path);
    out << "id,t0,t1,t2,t3\n"
        << "s0,0.0,0.1,0.2,0.1\n"
        << "s1,0.1,0.0,0.1,0.2\n"
        << "s2,0.05,0.15,0.1,0.05\n"
        << "s3,10.0,10.2,10.1,9.9\n"
        << "s4,10.1,10.0,9.95,10.1\n"
        << "s5,9.9,10.1,10.05,10.0\n";
  }

  const fs::path auto_out = scratch.path / "auto_results";
  const fs::path ckpt_dir = scratch.path / "ckpt";
  const fs::path manual_out = scratch.path / "manual_results";
  const auto entries = [&] {
    return std::distance(fs::recursive_directory_iterator(scratch.path), fs::recursive_directory_iterator{});
  };

  // The Dataset holds the reader options (a header line, an id column); the Config holds every other setting.
  auto ds = dtwc::load(csv_path, 1, 1);
  dtwc::Config cfg;
  cfg.k = 2;
  cfg.method = dtwc::Method::PAM;
  cfg.band = 2;
  cfg.skip_cols = 1; // the Dataset's own: equal, so not refused
  cfg.skip_rows = 1;
  cfg.name = "sensor";
  cfg.output = auto_out.string();
  cfg.checkpoint = ckpt_dir.string();

  // 1. The result files and the report go to cfg.output, the distance matrix to cfg.checkpoint.
  const auto res = dtwc::cluster(ds, cfg);
  REQUIRE(res.labels().size() == 6);
  REQUIRE(res.medoids().size() == 2);
  REQUIRE(res.labels()[0] == res.labels()[1]);
  REQUIRE(res.labels()[0] == res.labels()[2]);
  REQUIRE(res.labels()[3] == res.labels()[4]);
  REQUIRE(res.labels()[0] != res.labels()[3]);
  REQUIRE(fs::is_regular_file(auto_out / "sensor_labels.csv"));
  REQUIRE(fs::is_regular_file(auto_out / "sensor_medoids.csv"));
  REQUIRE(fs::is_regular_file(auto_out / "sensor_report.txt"));
#ifdef DTWC_HAS_MMAP
  REQUIRE(fs::is_regular_file(ckpt_dir / "sensor.dtwm"));
  // A second run with the same checkpoint resumes from it.
  const auto resumed = dtwc::cluster(ds, cfg);
  REQUIRE(resumed.labels() == res.labels());
  REQUIRE(std::regex_search(resumed.report(), std::regex(R"(the checkpoint\s+resumed)"))); // the report wraps its notes
#endif

  // 2. A Config with no output and no checkpoint writes nothing, and clusters alike; Result::save writes on request.
  dtwc::Config in_memory = cfg;
  in_memory.output.clear();
  in_memory.checkpoint.clear();
  const auto written = entries();
  const auto mem_res = dtwc::cluster(ds, in_memory);
  REQUIRE(entries() == written);
  REQUIRE(mem_res.labels() == res.labels());
  REQUIRE_THAT(mem_res.cost(), Catch::Matchers::WithinRel(res.cost(), 1e-12));

  mem_res.save(manual_out);
  REQUIRE(fs::is_regular_file(manual_out / "sensor_labels.csv"));
  REQUIRE(fs::is_regular_file(manual_out / "sensor_medoids.csv"));

  // 3. Dataset-level cdist with the same Config.
  const auto D_ds = dtwc::cdist(ds, ds, cfg);
  REQUIRE(D_ds.size() == 36);
  for (std::size_t i = 0; i < 6; ++i) {
    REQUIRE_THAT(D_ds[i * 6 + i], Catch::Matchers::WithinAbs(0.0, 1e-12));
  }

  // 4. A Dataset is read with the options it was made with: a Config that gives another value is refused, naming
  //    it; the same value runs.
  using Catch::Matchers::ContainsSubstring;
  using Catch::Matchers::MessageMatches;
  dtwc::Config other = in_memory;
  other.skip_rows = 2;
  REQUIRE_THROWS_MATCHES(dtwc::cluster(ds, other), dtwc::InvalidInput,
                         MessageMatches(ContainsSubstring("skip_rows 1 and skip_rows 2 was given")));
  REQUIRE_THROWS_MATCHES(dtwc::cdist(dtwc::load(make_equal_series(3, 4, 0.0)), ds, cfg), dtwc::InvalidInput,
                         MessageMatches(ContainsSubstring("skip_cols 0 and skip_cols 1 was given")));
  REQUIRE(dtwc::cluster(ds, in_memory).labels() == res.labels());
}

