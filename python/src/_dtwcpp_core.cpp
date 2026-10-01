/**
 * @file _dtwcpp_core.cpp
 * @brief nanobind Python bindings for DTWC++.
 *
 * @details Exposes DTW distance functions, clustering algorithms,
 *          distance matrix, and scoring to Python with zero-copy
 *          numpy integration where possible.
 *
 * @author Volkan Kumtepeli
 * 
 * @date 28 Mar 2026
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/pair.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <dtwc.hpp>
#include <base/env.hpp>
#include <base/error.hpp>
#include <io/arrow_c_data.hpp>
#include <checkpoint.hpp>
#include <warping.hpp>
#include <warping_ddtw.hpp>
#include <soft_dtw.hpp>
#include <algorithms/fast_pam.hpp>
#include <algorithms/fast_clara.hpp>
#include <algorithms/one_batch_pam.hpp>
#include <algorithms/barycenter.hpp>
#include <algorithms/hierarchical.hpp>
#include <scores.hpp>
#include <core/z_normalize.hpp>
#include <core/dtw_options.hpp>
#include <core/matrix_io.hpp>
#include <test_api.hpp> // dtwc::test::parallelisation()/gpu() introspection (Task 3.3)
#include <mip/mip.hpp>


#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals; // for _a arg names

namespace {

/// Hand an owning buffer to numpy with no leak window and no nested GIL scope.
///
/// The vector is moved onto the heap, the capsule is constructed while a
/// unique_ptr still owns it (so a throwing capsule allocation frees it), and
/// only then is ownership released to the capsule. Callers hold the GIL, so no
/// `gil_scoped_acquire` is nested inside a live release.
///
/// `T` is double for the distance matrices and `index_t` (int64) for labels and
/// medoids; a getter that hands out a Problem's or a result's indices passes a
/// copy, so the array never aliases state C++ goes on to change.
template <class T>
nb::ndarray<nb::numpy, T> adopt_as_ndarray(
  std::vector<T> &&values, std::initializer_list<size_t> shape) {
  // numpy never dereferences a zero-sized array, but nanobind still wants a
  // real address; an empty vector may report data() == nullptr.
  if (values.empty()) values.reserve(1);
  auto owned = std::make_unique<std::vector<T>>(std::move(values));
  T *ptr = owned->data();
  nb::capsule owner(owned.get(), [](void *p) noexcept {
    std::unique_ptr<std::vector<T>>(static_cast<std::vector<T> *>(p));
  });
  owned.release(); // the capsule owns the buffer from here on
  return nb::ndarray<nb::numpy, T>(ptr, shape, owner);
}

/// The matrix bindings' input check: every series once, before any pair is
/// computed, since the per-pair kernels do not check (warping.hpp). NaN or
/// ±inf raises InvalidInput naming the series index and the position.
void require_finite_series(const std::vector<std::vector<double>> &series,
                           const char *where) {
  for (size_t i = 0; i < series.size(); ++i)
    dtwc::detail::require_finite<double>(
      series[i], "series[" + std::to_string(i) + "]", where);
}

/// The range check every binding runs before it hands an index to C++ that does
/// not check it (Data::series, Data::name, Problem::dist_by_ind and
/// centroids_ind[clusters_ind[i]] sit in hot loops): an index outside [0, n) is
/// InvalidInput naming the index and n, not a read past the storage.
void require_index(const char *who, const char *name, std::int64_t index, size_t n) {
  if (index < 0 || static_cast<std::uint64_t>(index) >= n)
    throw dtwc::InvalidInput(
      std::string(who) + ": " + name + " = " + std::to_string(index)
      + " is outside [0, N) with N = " + std::to_string(n) + ".");
}

/// A distance configuration by the names dtwc_cl takes, read with the C++ name
/// tables; core::validate checks it where it is used (distance::dtw, Problem).
dtwc::core::DistanceConfig distance_config(const std::string &variant, int band, const std::string &metric,
                                           const std::string &missing_strategy, double wdtw_g,
                                           double adtw_penalty, double sdtw_gamma, double msm_c, double twe_nu,
                                           double twe_lambda) {
  using namespace dtwc::core;
  return { .variant = { .variant = dtwc::parse_name(variant_names, variant, "variant"),
                        .wdtw_g = wdtw_g,
                        .adtw_penalty = adtw_penalty,
                        .sdtw_gamma = sdtw_gamma,
                        .msm_c = msm_c,
                        .twe_nu = twe_nu,
                        .twe_lambda = twe_lambda },
           .metric = dtwc::parse_name(metric_names, metric, "metric"),
           .missing = dtwc::parse_name(missing_strategy_names, missing_strategy, "missing_strategy"),
           .band = band };
}

} // namespace

NB_MODULE(_dtwcpp_core, m) {
  m.attr("__version__") = DTWC_VERSION_STRING;
  m.attr("DEFAULT_RANDOM_SEED") = dtwc::settings::DEFAULT_RANDOM_SEED;
  m.attr("HIGHS_AVAILABLE") = dtwc::highs_solver_available();
  m.doc() = "DTWC++ — Fast Dynamic Time Warping and Clustering (C++ core)";

  // =========================================================================
  // Error taxonomy (api-contract-2.0.md §5)
  // =========================================================================
  // One base (DtwcError) + four leaves. Each leaf subclasses BOTH DtwcError AND
  // the closest built-in (ValueError / RuntimeError / OSError) so idiomatic
  // `except ValueError:` and `except dtwcpp.InvalidInput:` both catch it. The
  // types are function-local statics (module lifetime) referenced by the single
  // captureless translator below (registered so it runs before the default one).
  // UndefinedScore is the one sub-leaf: it mirrors dtwc::UndefinedScore, which
  // derives from dtwc::InvalidInput, so a caller that only wants to skip an
  // unwritable score file can distinguish it while `except InvalidInput` (and
  // `except ValueError`) keep catching it.
  static PyObject *g_exc_base = PyErr_NewException("dtwcpp.DtwcError", PyExc_Exception, nullptr);
  static PyObject *g_exc_invalid = nullptr;
  static PyObject *g_exc_undefined_score = nullptr;
  static PyObject *g_exc_solver = nullptr;
  static PyObject *g_exc_device = nullptr;
  static PyObject *g_exc_io = nullptr;
  {
    auto make_leaf = [](const char *qualname, PyObject *builtin) -> PyObject * {
      PyObject *bases = PyTuple_Pack(2, g_exc_base, builtin);
      PyObject *exc = PyErr_NewException(qualname, bases, nullptr);
      Py_XDECREF(bases);
      return exc;
    };
    g_exc_invalid = make_leaf("dtwcpp.InvalidInput", PyExc_ValueError);
    g_exc_undefined_score =
      PyErr_NewException("dtwcpp.UndefinedScore", g_exc_invalid, nullptr);
    g_exc_solver = make_leaf("dtwcpp.SolverError", PyExc_RuntimeError);
    g_exc_device = make_leaf("dtwcpp.DeviceError", PyExc_RuntimeError);
    g_exc_io = make_leaf("dtwcpp.IOError", PyExc_OSError);
  }
  m.attr("DtwcError") = nb::borrow(g_exc_base);
  m.attr("InvalidInput") = nb::borrow(g_exc_invalid);
  m.attr("UndefinedScore") = nb::borrow(g_exc_undefined_score);
  m.attr("SolverError") = nb::borrow(g_exc_solver);
  m.attr("DeviceError") = nb::borrow(g_exc_device);
  m.attr("IOError") = nb::borrow(g_exc_io);

  nb::register_exception_translator(
    [](const std::exception_ptr &p, void * /*payload*/) {
      try {
        std::rethrow_exception(p);
      } catch (const dtwc::UndefinedScore &e) {
        // Must precede InvalidInput: UndefinedScore derives from it.
        PyErr_SetString(g_exc_undefined_score, e.what());
      } catch (const dtwc::InvalidInput &e) {
        PyErr_SetString(g_exc_invalid, e.what());
      } catch (const dtwc::SolverError &e) {
        PyErr_SetString(g_exc_solver, e.what());
      } catch (const dtwc::DeviceError &e) {
        PyErr_SetString(g_exc_device, e.what());
      } catch (const dtwc::IOError &e) {
        PyErr_SetString(g_exc_io, e.what());
      } catch (const dtwc::Error &e) {
        PyErr_SetString(g_exc_base, e.what());
      }
    });

  // =========================================================================
  // Device (api-contract-2.0.md §6)
  // =========================================================================

  nb::enum_<dtwc::Device>(m, "Device")
    .value("CPU", dtwc::Device::CPU)
    .value("GPU", dtwc::Device::GPU);

  m.def("device_to_string", [](dtwc::Device d) { return dtwc::to_string(d); }, "device"_a,
        "Canonical lower-case name of a Device ('cpu'/'gpu').");

  m.def("parse_device", [](const std::string &name) {
        const auto [device, index] = dtwc::detail::parse_device(name);
        return std::make_pair(dtwc::to_string(device), index);
      }, "name"_a,
        "Parse a device name with the one C++ grammar (dtwc::detail::parse_device)\n"
        "and return (canonical name, GPU ordinal): 'CUDA:3' -> ('gpu', 3). Grammar\n"
        "only, the build is not checked; an unknown name raises DeviceError.");

  m.def("device", [](const std::string &name) {
        return dtwc::device(name);
      }, "name"_a,
        "Set the process-wide device and return its CANONICAL name\n"
        "('cpu'/'gpu'/'gpu:N'), exactly as dtwc::device(name) does.");

  m.def("device", []() { return dtwc::device(); },
        "Canonical name of the process-wide device (dtwc::device()).");

  // =========================================================================
  // Tier-1 file parsing (api-contract-2.0.md §1.2)
  // =========================================================================

  m.def("_read_data",
        [](const std::filesystem::path &source, dtwc::index_t skip_cols,
           dtwc::index_t skip_rows, const std::string &delimiter) {
    if (skip_cols < 0) throw dtwc::InvalidInput("load: skip_cols must be non-negative.");
    if (skip_rows < 0) throw dtwc::InvalidInput("load: skip_rows must be non-negative.");
    if (delimiter.size() > 1)
      throw dtwc::InvalidInput("load: delimiter must be a single character.");
    // File I/O and parsing touch no Python object, so the GIL is released for
    // the whole read exactly as every other I/O binding here does.
    nb::gil_scoped_release release;
    return dtwc::read_data(source, skip_cols, skip_rows, delimiter.empty() ? '\0' : delimiter[0]);
  }, "source"_a, "skip_cols"_a = 0, "skip_rows"_a = 0, "delimiter"_a = std::string{},
     "Read a path with dtwc::read_data, the reader dtwc_cl and C++ dtwc::load\n"
     "use (CSV/TSV and a folder of them; the wheel links no Arrow C++, so\n"
     "dtwcpp.load reads Parquet through pyarrow instead), and return the owning\n"
     "dtwc::Data (series + names) with no intermediate Python objects. Backs\n"
     "dtwcpp.Dataset, whose handle is handed straight to Problem.set_data(Data):\n"
     "skip_cols drops leading FIELDS before numeric parsing, skip_rows drops\n"
     "leading LINES, an empty delimiter means infer from the extension, and\n"
     "variable-length rows are preserved. The names are the reader's own, so\n"
     "Tier-1 output carries the series names the CLI writes.");

  m.def("_parquet_files", &dtwc::parquet_files, "path"_a,
        "The Parquet files a path names, listed as dtwc::read_data lists them:\n"
        "the file itself, or a folder's .parquet/.pq files, sorted, hidden files\n"
        "skipped; empty for any other input.");

  // =========================================================================
  // Enums
  // =========================================================================

  nb::enum_<dtwc::Method>(m, "Method")
    .value("Kmedoids", dtwc::Method::Kmedoids)
    .value("MIP", dtwc::Method::MIP)
    .value("LRCore", dtwc::Method::LRCore)
    .value("TADPole", dtwc::Method::TADPole);

  nb::enum_<dtwc::Solver>(m, "Solver")
    .value("Gurobi", dtwc::Solver::Gurobi)
    .value("HiGHS", dtwc::Solver::HiGHS);

  nb::enum_<dtwc::core::MetricType>(m, "MetricType")
    .value("L1", dtwc::core::MetricType::L1)
    .value("L2", dtwc::core::MetricType::L2)
    .value("SquaredL2", dtwc::core::MetricType::SquaredL2);

  nb::enum_<dtwc::core::DTWVariant>(m, "DTWVariant")
    .value("Standard", dtwc::core::DTWVariant::Standard)
    .value("DDTW", dtwc::core::DTWVariant::DDTW)
    .value("WDTW", dtwc::core::DTWVariant::WDTW)
    .value("ADTW", dtwc::core::DTWVariant::ADTW)
    .value("SoftDTW", dtwc::core::DTWVariant::SoftDTW)
    .value("MSM", dtwc::core::DTWVariant::MSM)
    .value("TWE", dtwc::core::DTWVariant::TWE);

  nb::enum_<dtwc::core::MVMode>(m, "MVMode")
    .value("Dependent", dtwc::core::MVMode::Dependent)
    .value("Independent", dtwc::core::MVMode::Independent);

  nb::enum_<dtwc::core::MissingStrategy>(m, "MissingStrategy")
    .value("Error", dtwc::core::MissingStrategy::Error)
    .value("ZeroCost", dtwc::core::MissingStrategy::ZeroCost)
    .value("AROW", dtwc::core::MissingStrategy::AROW)
    .value("Interpolate", dtwc::core::MissingStrategy::Interpolate);

  nb::enum_<dtwc::DistanceMatrixStrategy>(m, "DistanceMatrixStrategy")
    .value("Auto", dtwc::DistanceMatrixStrategy::Auto)
    .value("BruteForce", dtwc::DistanceMatrixStrategy::BruteForce)
    .value("CUDA", dtwc::DistanceMatrixStrategy::CUDA)
    .value("Metal", dtwc::DistanceMatrixStrategy::Metal);

  // =========================================================================
  // CUDASettings
  // =========================================================================

  nb::enum_<dtwc::GpuPrecision>(m, "GpuPrecision")
    .value("Auto", dtwc::GpuPrecision::Auto)
    .value("FP32", dtwc::GpuPrecision::FP32)
    .value("FP64", dtwc::GpuPrecision::FP64);

  nb::class_<dtwc::CUDASettings>(m, "CUDASettings")
    .def(nb::init<>())
    .def_rw("device_id", &dtwc::CUDASettings::device_id, "CUDA device index (default 0).")
    .def_rw("precision", &dtwc::CUDASettings::precision,
            "Compute precision: GpuPrecision.Auto (default), FP32 or FP64.")
    .def("__repr__", [](const dtwc::CUDASettings &s) {
      return "CUDASettings(device_id=" + std::to_string(s.device_id) + ", precision="
             + std::string(dtwc::name_of(dtwc::gpu_precision_names, s.precision)) + ")";
    });

  // =========================================================================
  // Linkage (hierarchical clustering)
  // =========================================================================

  nb::enum_<dtwc::algorithms::Linkage>(m, "Linkage")
    .value("Single", dtwc::algorithms::Linkage::Single)
    .value("Complete", dtwc::algorithms::Linkage::Complete)
    .value("Average", dtwc::algorithms::Linkage::Average);

  nb::enum_<dtwc::algorithms::BarycenterMethod>(m, "BarycenterMethod")
    .value("SSG", dtwc::algorithms::BarycenterMethod::SSG)
    .value("DBA", dtwc::algorithms::BarycenterMethod::DBA)
    .value("SoftDTW", dtwc::algorithms::BarycenterMethod::SoftDTW);

  nb::class_<dtwc::algorithms::OneBatchPAMOptions>(m, "OneBatchPAMOptions")
    .def(nb::init<>())
    .def_rw("n_clusters", &dtwc::algorithms::OneBatchPAMOptions::n_clusters)
    .def_rw("batch_size", &dtwc::algorithms::OneBatchPAMOptions::batch_size)
    .def_rw("max_iter", &dtwc::algorithms::OneBatchPAMOptions::max_iter)
    .def_rw("random_seed", &dtwc::algorithms::OneBatchPAMOptions::random_seed)
    .def_rw("relative_tolerance", &dtwc::algorithms::OneBatchPAMOptions::relative_tolerance);

  nb::class_<dtwc::algorithms::BarycenterOptions>(m, "BarycenterOptions")
    .def(nb::init<>())
    .def_rw("method", &dtwc::algorithms::BarycenterOptions::method)
    .def_rw("max_iter", &dtwc::algorithms::BarycenterOptions::max_iter)
    .def_rw("learning_rate", &dtwc::algorithms::BarycenterOptions::learning_rate)
    .def_rw("learning_rate_decay", &dtwc::algorithms::BarycenterOptions::learning_rate_decay)
    .def_rw("gamma", &dtwc::algorithms::BarycenterOptions::gamma)
    .def_rw("tolerance", &dtwc::algorithms::BarycenterOptions::tolerance)
    .def_rw("random_seed", &dtwc::algorithms::BarycenterOptions::random_seed);

  nb::class_<dtwc::algorithms::BarycenterClusteringOptions>(m, "BarycenterClusteringOptions")
    .def(nb::init<>())
    .def_rw("n_clusters", &dtwc::algorithms::BarycenterClusteringOptions::n_clusters)
    .def_rw("max_iter", &dtwc::algorithms::BarycenterClusteringOptions::max_iter)
    .def_rw("barycenter_max_iter", &dtwc::algorithms::BarycenterClusteringOptions::barycenter_max_iter)
    .def_rw("target_length", &dtwc::algorithms::BarycenterClusteringOptions::target_length)
    .def_rw("method", &dtwc::algorithms::BarycenterClusteringOptions::method)
    .def_rw("learning_rate", &dtwc::algorithms::BarycenterClusteringOptions::learning_rate)
    .def_rw("learning_rate_decay", &dtwc::algorithms::BarycenterClusteringOptions::learning_rate_decay)
    .def_rw("gamma", &dtwc::algorithms::BarycenterClusteringOptions::gamma)
    .def_rw("tolerance", &dtwc::algorithms::BarycenterClusteringOptions::tolerance)
    .def_rw("random_seed", &dtwc::algorithms::BarycenterClusteringOptions::random_seed);

  nb::class_<dtwc::algorithms::BarycenterClusteringResult>(m, "BarycenterClusteringResult")
    .def(nb::init<>())
    .def_prop_ro("labels", [](const dtwc::algorithms::BarycenterClusteringResult &r) {
      return adopt_as_ndarray(std::vector<dtwc::index_t>(r.labels), {r.labels.size()});
    }, nb::rv_policy::move)
    .def_ro("barycenters", &dtwc::algorithms::BarycenterClusteringResult::barycenters)
    .def_ro("total_cost", &dtwc::algorithms::BarycenterClusteringResult::total_cost)
    .def_ro("iterations", &dtwc::algorithms::BarycenterClusteringResult::iterations)
    .def_ro("converged", &dtwc::algorithms::BarycenterClusteringResult::converged);

  // =========================================================================
  // DTWVariantParams
  // =========================================================================

  using Params = dtwc::core::DTWVariantParams;
  // A field takes a value only when the whole parameter set stays valid (core::validate).
  const auto checked = [](double Params::*field) {
    return [field](Params &p, double value) {
      auto candidate = p;
      candidate.*field = value;
      dtwc::core::validate({ candidate }, false);
      p = candidate;
    };
  };
  nb::class_<Params>(m, "DTWVariantParams")
    .def(nb::init<>())
    .def_rw("variant", &Params::variant)
    .def_prop_rw("wdtw_g", [](const Params &p) { return p.wdtw_g; }, checked(&Params::wdtw_g))
    .def_prop_rw("adtw_penalty", [](const Params &p) { return p.adtw_penalty; }, checked(&Params::adtw_penalty))
    .def_prop_rw("sdtw_gamma", [](const Params &p) { return p.sdtw_gamma; }, checked(&Params::sdtw_gamma))
    .def_prop_rw("msm_c", [](const Params &p) { return p.msm_c; }, checked(&Params::msm_c))
    .def_prop_rw("twe_nu", [](const Params &p) { return p.twe_nu; }, checked(&Params::twe_nu))
    .def_prop_rw("twe_lambda", [](const Params &p) { return p.twe_lambda; }, checked(&Params::twe_lambda))
    .def_rw("mv_mode", &Params::mv_mode);

  // =========================================================================
  // MIPSettings
  // =========================================================================

  nb::class_<dtwc::MIPSettings>(m, "MIPSettings")
    .def(nb::init<>())
    .def_rw("mip_gap", &dtwc::MIPSettings::mip_gap,
            "Relative MIP gap tolerance (default 1e-5).")
    .def_rw("time_limit_sec", &dtwc::MIPSettings::time_limit_sec,
            "Solver time limit in seconds (-1 = unlimited).")
    .def_rw("warm_start", &dtwc::MIPSettings::warm_start,
            "Run FastPAM first and feed as MIP start (default True).")
    .def_rw("numeric_focus", &dtwc::MIPSettings::numeric_focus,
            "Gurobi NumericFocus (0-3, default 1).")
    .def_rw("mip_focus", &dtwc::MIPSettings::mip_focus,
            "Gurobi MIPFocus (0=balanced, 1=feasible, 2=optimal, 3=bound).")
    .def_rw("verbose_solver", &dtwc::MIPSettings::verbose_solver,
            "Show solver log output (default False).")
    .def_rw("lr_max_nodes", &dtwc::MIPSettings::lr_max_nodes,
            "Method.LRCore branch-and-bound node cap (>= 1, default 2000000).")
    .def("__repr__", [](const dtwc::MIPSettings &s) {
      return "MIPSettings(gap=" + std::to_string(s.mip_gap)
             + ", time_limit=" + std::to_string(s.time_limit_sec)
             + ", warm_start=" + (s.warm_start ? "True" : "False")
             + ", numeric_focus=" + std::to_string(s.numeric_focus)
             + ", mip_focus=" + std::to_string(s.mip_focus)
             + ", lr_max_nodes=" + std::to_string(s.lr_max_nodes)
             + ", verbose=" + (s.verbose_solver ? "True" : "False") + ")";
    });

  // =========================================================================
  // DendrogramStep
  // =========================================================================

  nb::class_<dtwc::algorithms::DendrogramStep>(m, "DendrogramStep")
    .def(nb::init<>())
    .def_rw("cluster_a", &dtwc::algorithms::DendrogramStep::cluster_a,
            "First merged cluster index.")
    .def_rw("cluster_b", &dtwc::algorithms::DendrogramStep::cluster_b,
            "Second merged cluster index.")
    .def_rw("distance", &dtwc::algorithms::DendrogramStep::distance,
            "Merge distance.")
    .def_rw("new_size", &dtwc::algorithms::DendrogramStep::new_size,
            "Size of the merged cluster.")
    .def("__repr__", [](const dtwc::algorithms::DendrogramStep &s) {
      return "DendrogramStep(a=" + std::to_string(s.cluster_a)
             + ", b=" + std::to_string(s.cluster_b)
             + ", dist=" + std::to_string(s.distance)
             + ", size=" + std::to_string(s.new_size) + ")";
    });

  // =========================================================================
  // Dendrogram
  // =========================================================================

  nb::class_<dtwc::algorithms::Dendrogram>(m, "Dendrogram")
    .def(nb::init<>())
    .def_rw("merges", &dtwc::algorithms::Dendrogram::merges,
            "List of N-1 DendrogramStep merge records.")
    .def_rw("n_points", &dtwc::algorithms::Dendrogram::n_points,
            "Number of original data points.")
    .def("__repr__", [](const dtwc::algorithms::Dendrogram &d) {
      return "Dendrogram(n_points=" + std::to_string(d.n_points)
             + ", merges=" + std::to_string(d.merges.size()) + ")";
    });

  // =========================================================================
  // HierarchicalOptions
  // =========================================================================

  nb::class_<dtwc::algorithms::HierarchicalOptions>(m, "HierarchicalOptions")
    .def(nb::init<>())
    .def_rw("linkage", &dtwc::algorithms::HierarchicalOptions::linkage,
            "Linkage criterion (Single, Complete, or Average).")
    .def_rw("max_points", &dtwc::algorithms::HierarchicalOptions::max_points,
            "Hard guard: throws if N exceeds this (default 2000).")
    .def("__repr__", [](const dtwc::algorithms::HierarchicalOptions &o) {
      std::string linkage_str;
      switch (o.linkage) {
        case dtwc::algorithms::Linkage::Single: linkage_str = "Single"; break;
        case dtwc::algorithms::Linkage::Complete: linkage_str = "Complete"; break;
        case dtwc::algorithms::Linkage::Average: linkage_str = "Average"; break;
      }
      return "HierarchicalOptions(linkage=" + linkage_str
             + ", max_points=" + std::to_string(o.max_points) + ")";
    });

  // =========================================================================
  // ClusteringResult
  // =========================================================================

  nb::class_<dtwc::core::ClusteringResult>(m, "ClusteringResult")
    .def(nb::init<>())
    .def_prop_rw("labels",
      [](const dtwc::core::ClusteringResult &r) {
        return adopt_as_ndarray(std::vector<dtwc::index_t>(r.labels), {r.labels.size()});
      },
      [](dtwc::core::ClusteringResult &r, std::vector<dtwc::index_t> labels) {
        r.labels = std::move(labels);
      }, nb::rv_policy::move)
    .def_prop_rw("medoid_indices",
      [](const dtwc::core::ClusteringResult &r) {
        return adopt_as_ndarray(std::vector<dtwc::index_t>(r.medoid_indices),
                                {r.medoid_indices.size()});
      },
      [](dtwc::core::ClusteringResult &r, std::vector<dtwc::index_t> medoids) {
        r.medoid_indices = std::move(medoids);
      }, nb::rv_policy::move)
    .def_rw("total_cost", &dtwc::core::ClusteringResult::total_cost)
    .def_rw("iterations", &dtwc::core::ClusteringResult::iterations)
    .def_rw("converged", &dtwc::core::ClusteringResult::converged)
    .def_prop_ro("n_clusters", &dtwc::core::ClusteringResult::n_clusters)
    .def_prop_ro("n_points", &dtwc::core::ClusteringResult::n_points)
    .def("__repr__", [](const dtwc::core::ClusteringResult &r) {
      return "ClusteringResult(k=" + std::to_string(r.n_clusters())
             + ", cost=" + std::to_string(r.total_cost)
             + ", iters=" + std::to_string(r.iterations)
             + ", converged=" + (r.converged ? "True" : "False") + ")";
    });

  // =========================================================================
  // DTW distance
  // =========================================================================

  // dtwc::distance::dtw, the checked boundary: core::validate refuses a
  // configuration no kernel implements, then x and y are scanned once (NaN or
  // ±inf raises InvalidInput naming x or y and the position; a missing-data
  // strategy reads NaN as missing). The arrays are read in place, and the GIL
  // is released for the computation.
  const dtwc::core::DTWVariantParams defaults{};
  m.def("dtw", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                  nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y, const std::string &variant, int band,
                  const std::string &metric, const std::string &missing_strategy, double wdtw_g,
                  double adtw_penalty, double sdtw_gamma, double msm_c, double twe_nu, double twe_lambda) {
    const auto c = distance_config(variant, band, metric, missing_strategy, wdtw_g, adtw_penalty, sdtw_gamma,
                                   msm_c, twe_nu, twe_lambda);
    nb::gil_scoped_release release;
    return dtwc::distance::dtw<double>(std::span<const double>(x.data(), x.size()),
                                       std::span<const double>(y.data(), y.size()), c.variant, c.band,
                                       c.metric, c.missing);
  }, "x"_a, "y"_a, nb::kw_only(), "variant"_a = "standard", "band"_a = dtwc::settings::DEFAULT_BAND,
     "metric"_a = "l1", "missing_strategy"_a = "error", "wdtw_g"_a = defaults.wdtw_g,
     "adtw_penalty"_a = defaults.adtw_penalty, "sdtw_gamma"_a = defaults.sdtw_gamma, "msm_c"_a = defaults.msm_c,
     "twe_nu"_a = defaults.twe_nu, "twe_lambda"_a = defaults.twe_lambda,
     "DTW-family distance of two float64 series (dtwc::distance::dtw).\n\n"
     "variant: standard, ddtw, wdtw, adtw, softdtw, msm or twe; each reads its own\n"
     "parameter (wdtw_g, adtw_penalty, sdtw_gamma, msm_c, twe_nu and twe_lambda).\n"
     "band: -1 for full DTW, b >= 0 for a Sakoe-Chiba half-width.\n"
     "metric: l1 or squared_euclidean (Standard DTW and DDTW).\n"
     "missing_strategy: error, zero_cost, arow or interpolate (Standard DTW); NaN\n"
     "is a missing value under the last three.\n"
     "Raises InvalidInput for an unknown name, a parameter outside its domain, a\n"
     "combination no kernel implements, or a value the strategy does not take.");

  m.def("soft_dtw_gradient", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                                 nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                                 double gamma) {
    nb::gil_scoped_release release;
    return dtwc::soft_dtw_gradient<double>(std::span<const double>(x.data(), x.size()),
                                           std::span<const double>(y.data(), y.size()), gamma);
  }, "x"_a, "y"_a, "gamma"_a = 1.0,
     "Compute Soft-DTW gradient w.r.t. first series x (zero-copy from numpy).\n\n"
     "NaN or +-inf in x or y raises InvalidInput.");

  // =========================================================================
  // Utility functions
  // =========================================================================

  m.def("derivative_transform", [](const std::vector<double> &x) {
    return dtwc::derivative_transform<double>(x);
  }, "x"_a, "Compute derivative transform for DDTW.");

  m.def("z_normalize", [](std::vector<double> x) {
    dtwc::core::z_normalize(x.data(), x.size());
    return x;
  }, "x"_a, "Return z-normalized copy (zero mean, unit stddev).");

  // =========================================================================
  // Data
  // =========================================================================

  nb::class_<dtwc::Data>(m, "Data")
    .def(nb::init<>())
    .def(nb::init<std::vector<std::vector<dtwc::data_t>> &&,
                   std::vector<std::string> &&>(),
         "series"_a, "names"_a)
    .def(nb::init<std::vector<std::vector<dtwc::data_t>> &&,
                   std::vector<std::string> &&, size_t>(),
         "series"_a, "names"_a, "ndim"_a,
         "Float64 heap-mode data with multivariate interleaved layout (ndim>1).")
    .def_static("from_float32",
                [](std::vector<std::vector<float>> series,
                   std::vector<std::string> names, size_t ndim) {
                  return dtwc::Data(std::move(series), std::move(names), ndim);
                },
                "series"_a, "names"_a, "ndim"_a = 1,
                "Float32 heap-mode data (explicit opt-in; halves storage, distances\n"
                "are still accumulated and returned in double).")
    .def_rw("p_vec", &dtwc::Data::p_vec)
    .def_rw("p_names", &dtwc::Data::p_names)
    .def_rw("ndim", &dtwc::Data::ndim,
            "Number of features (dimensions) per timestep (default 1).")
    .def_prop_ro("size", &dtwc::Data::size)
    .def("is_f32", &dtwc::Data::is_f32, "True if series are stored as float32.")
    .def("is_view", &dtwc::Data::is_view, "True if data is a non-owning view (spans).")
    .def("series_length", &dtwc::Data::series_length, "i"_a,
         "Return the number of timesteps for series i (flat size / ndim).")
    .def("validate_ndim", &dtwc::Data::validate_ndim,
         "Validate that all series flat sizes are divisible by ndim.\n\n"
         "Raises InvalidInput (a ValueError) if any series has incompatible size.");

  // Zero-copy Arrow C Data interface ingest (Task 5.7). Consumes the PyCapsule
  // protocol `__arrow_c_array__` (polars / DuckDB / pyarrow / pandas) via the
  // vendored nanoarrow — NO pyarrow dependency. Each list element becomes one
  // (univariate) series; see dtwc::io::data_from_arrow for accepted layouts.
  m.def("data_from_arrow_c_array", [](nb::object obj) -> dtwc::Data {
    // Producers expose one of two PyCapsule protocols. pyarrow.Array / DuckDB
    // give a single array (__arrow_c_array__); polars / pandas give a batch
    // stream (__arrow_c_stream__). Support both; prefer the single-array form.
    if (!nb::hasattr(obj, "__arrow_c_array__") && nb::hasattr(obj, "__arrow_c_stream__")) {
      nb::object cap = obj.attr("__arrow_c_stream__")();
      auto *stream = static_cast<ArrowArrayStream *>(
        PyCapsule_GetPointer(cap.ptr(), "arrow_array_stream"));
      if (stream == nullptr) {
        if (PyErr_Occurred()) throw nb::python_error();
        throw dtwc::InvalidInput("data_from_arrow_c_array: null Arrow stream capsule.");
      }
      return dtwc::io::data_from_arrow_stream(stream);
    }

    if (!nb::hasattr(obj, "__arrow_c_array__"))
      throw dtwc::InvalidInput(
        "data_from_arrow_c_array: object does not implement the Arrow C Data "
        "interface (__arrow_c_array__ / __arrow_c_stream__). Pass a "
        "polars/DuckDB/pyarrow/pandas array.");

    nb::object capsules = obj.attr("__arrow_c_array__")();
    nb::tuple pair = nb::cast<nb::tuple>(capsules);
    if (pair.size() != 2)
      throw dtwc::InvalidInput("data_from_arrow_c_array: __arrow_c_array__ did not "
                               "return a (schema, array) capsule pair.");

    auto *schema = static_cast<ArrowSchema *>(
      PyCapsule_GetPointer(pair[0].ptr(), "arrow_schema"));
    auto *array = static_cast<ArrowArray *>(
      PyCapsule_GetPointer(pair[1].ptr(), "arrow_array"));
    if (schema == nullptr || array == nullptr) {
      if (PyErr_Occurred()) throw nb::python_error();
      throw dtwc::InvalidInput("data_from_arrow_c_array: null Arrow capsule pointer.");
    }

    // Read (copies into owning Data), then release the borrowed structs. The
    // capsule destructors see release==nullptr afterwards and become no-ops, so
    // there is no double free.
    dtwc::Data data;
    try {
      data = dtwc::io::data_from_arrow(schema, array);
    } catch (...) {
      dtwc::io::release_arrow(schema, array);
      throw;
    }
    dtwc::io::release_arrow(schema, array);
    return data;
  }, "obj"_a,
     "Build a Data object from an Arrow C Data interface source (zero-copy,\n"
     "no pyarrow). `obj` must implement __arrow_c_array__ over a list/large_list\n"
     "of float32/float64 (each element one series) or a struct containing one.");

  // =========================================================================
  // Problem class
  // =========================================================================

  // Labels and medoids leave as int64 copies: clusters_ind / labels() and
  // centroids_ind / medoids() are the same data under two spellings.
  const auto labels_of = [](const dtwc::Problem &p) {
    return adopt_as_ndarray(std::vector<dtwc::index_t>(p.labels()), {p.labels().size()});
  };
  const auto medoids_of = [](const dtwc::Problem &p) {
    return adopt_as_ndarray(std::vector<dtwc::index_t>(p.medoids()), {p.medoids().size()});
  };

  nb::class_<dtwc::Problem>(m, "Problem",
    "A clustering problem: data, configuration, distance matrix and results.\n\n"
    "Threading: a Problem instance must not be used concurrently from multiple\n"
    "Python threads; the GIL is released during C++ work so that other threads\n"
    "can run, but two threads calling methods on the same Problem race on its\n"
    "lazily-filled distance cache. Use one Problem per thread, or call\n"
    "fill_distance_matrix() first and only read afterwards.")
    .def("__init__", [](dtwc::Problem *p, const std::string &name,
                        const std::string &device) {
      // Select the device before constructing in place, so a rejected name
      // never leaves nanobind holding a half-built Problem.
      const auto [selected, index] = dtwc::detail::parse_device(device);
      dtwc::Problem prob(name);
      prob.set_device(selected, index);
      new (p) dtwc::Problem(std::move(prob));
    }, "name"_a = std::string(), nb::kw_only(), "device"_a = "cpu",
       "Create a Problem that computes on `device`: 'cpu' (default), 'gpu',\n"
       "'gpu:N', 'cuda' or 'cuda:N', as for dtwcpp.device(). A Problem does not\n"
       "follow the process-wide dtwcpp.device(). 'gpu' without a GPU backend\n"
       "raises DeviceError; 'hpc' raises InvalidInput (it is a cluster() option).")
    .def("set_device", [](dtwc::Problem &p, const std::string &device) {
      const auto [selected, index] = dtwc::detail::parse_device(device);
      p.set_device(selected, index);
    }, "device"_a,
       "Compute on `device` (the names dtwcpp.device() accepts). 'cpu' keeps a\n"
       "CPU distance_strategy you chose and moves a GPU one to Auto; 'gpu' selects\n"
       "this build's GPU backend (CUDA, else Metal). A request the device cannot\n"
       "honour (a variant, missing-data strategy, multivariate data or precision\n"
       "its kernels lack) raises DeviceError when distances are computed.")
    // ---- config properties (canonical names) ----
    .def_prop_rw("method", &dtwc::Problem::method, &dtwc::Problem::set_method)
    .def_prop_rw("max_iter", &dtwc::Problem::max_iter,
                 &dtwc::Problem::set_max_iter)
    .def_prop_rw("n_repetitions", &dtwc::Problem::n_repetitions,
                 &dtwc::Problem::set_n_repetitions,
                 "Repetitions for iterative methods.")
    .def_prop_rw("random_seed", &dtwc::Problem::random_seed,
                 &dtwc::Problem::set_random_seed,
                 "Invocation-local seed for Lloyd and MIP warm starts.")
    .def_prop_rw("band",
                 [](const dtwc::Problem &p) { return p.band; },
                 [](dtwc::Problem &p, int value) { p.set_band(value); })
    .def_prop_rw("variant_params",
                 [](const dtwc::Problem &p) { return p.variant_params(); },
                 [](dtwc::Problem &p, dtwc::core::DTWVariantParams value) {
                   p.set_variant(value);
                 })
    .def_prop_rw("missing_strategy",
                 [](const dtwc::Problem &p) { return p.missing_strategy(); },
                 [](dtwc::Problem &p, dtwc::core::MissingStrategy value) {
                   p.set_missing_strategy(value);
                 },
                 "Strategy for handling NaN values (Error, ZeroCost, AROW, Interpolate).")
    .def_prop_rw("distance_strategy",
                 [](const dtwc::Problem &p) { return p.distance_strategy(); },
                 [](dtwc::Problem &p, dtwc::DistanceMatrixStrategy value) {
                   p.set_distance_strategy(value);
                 },
                 "Distance matrix computation strategy (Auto, BruteForce, CUDA, Metal).")
    .def_prop_rw("cuda_settings",
                 [](const dtwc::Problem &p) { return p.cuda_settings(); },
                 [](dtwc::Problem &p, dtwc::CUDASettings value) {
                   p.set_cuda_settings(value);
                 },
                 "GPU compute options (device_id, precision), read by the CUDA and\n"
                 "Metal routes; set_device('gpu:N') sets device_id.")
    .def_rw("mip_settings", &dtwc::Problem::mip_settings,
            "MIP solver tuning parameters.")
    .def_rw("checkpoint", &dtwc::Problem::checkpoint,
            nb::rv_policy::reference_internal,
            "Automatic mid-fill checkpointing options (CheckpointOptions),\n"
            "consumed by fill_distance_matrix(). The getter returns a view of\n"
            "the member, so prob.checkpoint.enabled = True mutates the Problem.")
    .def_prop_rw("verbose", &dtwc::Problem::verbose,
                 &dtwc::Problem::set_verbose,
                 "Print progress messages for long-running operations.")
    .def_prop_rw("name", &dtwc::Problem::name, &dtwc::Problem::set_name)
    .def_prop_rw("output_folder", &dtwc::Problem::output_folder,
                 &dtwc::Problem::set_output_folder,
                 "Output folder for results written by the write_* methods.")
    .def_prop_ro("clusters_ind", labels_of, nb::rv_policy::move,
            "Cluster label of each series, an int64 array. Read-only: set_result is the write route.")
    .def_prop_ro("centroids_ind", medoids_of, nb::rv_policy::move,
            "Medoid series indices, an int64 array. Read-only: set_result is the write route.")
    .def("set_result", &dtwc::Problem::set_result, "result"_a,
         "Publish a ClusteringResult on this Problem: k is the number of medoids,\n"
         "which are distinct indices in [0, N), and every series has one label in\n"
         "[0, k). Anything else raises InvalidInput and leaves the Problem unchanged.")
    // ---- read accessors ----
    .def_prop_ro("size", &dtwc::Problem::size)
    .def("n_clusters", &dtwc::Problem::n_clusters, "Number of clusters.")
    .def("cluster_size", &dtwc::Problem::n_clusters,
         "Number of clusters: the v1.0.0 spelling of n_clusters().")
    .def("labels", labels_of,
         "Cluster label of each series, an int64 array (reads clusters_ind; parity with\n"
         "Result.labels).")
    .def("medoids", medoids_of,
         "Medoid series indices, an int64 array (reads centroids_ind; parity with\n"
         "Result.medoids).")
    .def("series", [](const dtwc::Problem &p, std::int64_t i) {
      require_index("series", "i", i, p.size());
      auto s = p.series(static_cast<size_t>(i));
      return std::vector<double>(s.begin(), s.end());
    }, "i"_a,
       "Copy of series i as a list of doubles.\n\n"
       "Raises InvalidInput if i is outside [0, N).")
    .def("series_name", [](const dtwc::Problem &p, std::int64_t i) {
      require_index("series_name", "i", i, p.size());
      return std::string(p.series_name(static_cast<size_t>(i)));
    }, "i"_a,
       "Name of series i.\n\n"
       "Raises InvalidInput if i is outside [0, N).")
    .def("centroid_of", [](const dtwc::Problem &p, std::int64_t i) {
      require_index("centroid_of", "i", i, p.size());
      p.require_clustered("centroid_of"); // Problem::centroid_of reads both vectors unchecked
      return p.centroid_of(i);
    }, "i"_a,
       "Medoid index of the cluster that series i belongs to.\n\n"
       "Raises InvalidInput if i is outside [0, N) or the Problem holds no clustering.")
    .def("is_distance_matrix_filled", &dtwc::Problem::is_distance_matrix_filled)
    .def("max_distance", &dtwc::Problem::max_distance)
    .def("dist_by_ind", [](dtwc::Problem &p, dtwc::index_t i, dtwc::index_t j) {
      // Problem::dist_by_ind is the unchecked hot path: this boundary owns the range check.
      require_index("dist_by_ind", "i", i, p.size());
      require_index("dist_by_ind", "j", j, p.size());
      nb::gil_scoped_release release;
      p.fill_distance_matrix(); // a no-op once filled: Problem::dist_by_ind reads the matrix
      return p.dist_by_ind(i, j);
    }, "i"_a, "j"_a,
       "Distance between series i and j.\n\n"
       "Raises InvalidInput if i or j is outside [0, N).\n\n"
       "The first call on a Problem whose matrix is not filled fills it\n"
       "(fill_distance_matrix()), which MUTATES this Problem: do not make it\n"
       "concurrently from several Python threads on the same object (see the\n"
       "Problem class docstring).")
    // ---- config setters ----
    .def("set_n_clusters", &dtwc::Problem::set_n_clusters, "n_clusters"_a)
    .def("set_method", &dtwc::Problem::set_method, "method"_a)
    .def("set_band", &dtwc::Problem::set_band, "band"_a)
    .def("set_max_iter", &dtwc::Problem::set_max_iter, "max_iter"_a)
    .def("set_n_repetitions", &dtwc::Problem::set_n_repetitions, "n_repetitions"_a)
    .def("set_random_seed", &dtwc::Problem::set_random_seed, "random_seed"_a)
    .def("set_variant", nb::overload_cast<dtwc::core::DTWVariant>(&dtwc::Problem::set_variant), "variant"_a)
    .def("set_variant_params",
         nb::overload_cast<dtwc::core::DTWVariantParams>(&dtwc::Problem::set_variant), "params"_a,
         "Set the DTW variant + parameters and rebind the distance function.")
    .def("set_distance",
         [](dtwc::Problem &p, const std::string &variant, int band, const std::string &metric,
            const std::string &missing_strategy, const std::string &mv_mode, double wdtw_g, double adtw_penalty,
            double sdtw_gamma, double msm_c, double twe_nu, double twe_lambda) {
           auto c = distance_config(variant, band, metric, missing_strategy, wdtw_g, adtw_penalty, sdtw_gamma,
                                    msm_c, twe_nu, twe_lambda);
           c.variant.mv_mode = dtwc::parse_name(dtwc::core::mv_mode_names, mv_mode, "mv_mode");
           p.set_distance(c);
         }, nb::kw_only(), "variant"_a = "standard", "band"_a = dtwc::settings::DEFAULT_BAND, "metric"_a = "l1",
         "missing_strategy"_a = "error", "mv_mode"_a = "dependent", "wdtw_g"_a = defaults.wdtw_g,
         "adtw_penalty"_a = defaults.adtw_penalty, "sdtw_gamma"_a = defaults.sdtw_gamma,
         "msm_c"_a = defaults.msm_c, "twe_nu"_a = defaults.twe_nu, "twe_lambda"_a = defaults.twe_lambda,
         "Set every distance setting at once, by the names distance.dtw takes plus\n"
         "mv_mode (dependent or independent, for multivariate series); a setting not\n"
         "given takes its default. A change drops the distance matrix and the\n"
         "clustering. An invalid configuration raises InvalidInput and changes nothing.")
    .def("set_solver", &dtwc::Problem::set_solver, "solver"_a,
         "Select the MIP solver (Gurobi/HiGHS). Returns True if the solver is available.")
    .def("set_data", [](dtwc::Problem &p, std::vector<std::vector<double>> series,
                         std::vector<std::string> names, size_t ndim) {
      dtwc::Data d(std::move(series), std::move(names), ndim);
      p.set_data(std::move(d));
    }, "series"_a, "names"_a, "ndim"_a = 1,
       "Set time series data (ndim>1 for multivariate interleaved layout).")
    .def("set_data", [](dtwc::Problem &p, dtwc::Data d) { p.set_data(std::move(d)); },
         "data"_a, "Set time series data from a Data object (enables f32 storage).")
    // ---- distance matrix ----
    .def("fill_distance_matrix", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.fill_distance_matrix();
    }, "Compute all pairwise DTW distances.")
    // Always a COPY: the C++ store keeps only the upper triangle, so a zero-copy
    // view into a full NxN layout is structurally impossible (§2.2 ‡).
    .def("distance_matrix", [](dtwc::Problem &prob) {
           // Size is only known after the fill, so both happen inside one release.
           std::vector<double> values;
           size_t n = 0;
           {
             nb::gil_scoped_release release;
             prob.fill_distance_matrix();
             const auto &dm = std::as_const(prob).distance_matrix(); // on the heap or mapped
             n = dm.size();
             values = dtwc::io::to_full_matrix(dm); // row-major, expanded from the triangle
           }
           return adopt_as_ndarray(std::move(values), {n, n});
         },
         "Fill (if needed) and return the full NxN distance matrix as a numpy\n"
         "array (independent copy; use set_distance_matrix to write).")
    .def("set_distance_matrix",
         [](dtwc::Problem &p,
            nb::ndarray<const double, nb::ndim<2>, nb::c_contig> dm) {
           const size_t n = dm.shape(0);
           if (dm.shape(1) != n)
             throw dtwc::InvalidInput("Expected square distance matrix");
           if (n != p.size())
             throw dtwc::InvalidInput("Matrix size doesn't match Problem data size");
           auto &mat = p.distance_matrix();
           // Values written into a mapped matrix would persist in its file under the
           // Problem's fingerprint, whatever they were computed from.
           if (mat.is_mapped())
             throw dtwc::InvalidInput("Problem.set_distance_matrix: this Problem's distance matrix is "
                                      "memory-mapped (use_mmap_distance_matrix), and a supplied matrix is "
                                      "kept in RAM only; call refresh_distance_matrix() first.");
           mat.resize(n);
           const double *data = dm.data();
           for (size_t i = 0; i < n; ++i)
             for (size_t j = i; j < n; ++j)
               mat.set(i, j, data[i * n + j]);
           // A complete matrix is filled; NaN entries are computed on first use.
           if (mat.all_computed("Problem.set_distance_matrix")) p.fill_distance_matrix();
         }, "dm"_a,
         "Load a precomputed NxN distance matrix (e.g. from a GPU compute). NaN marks\n"
         "a pair to compute; a ±inf entry raises InvalidInput naming the pair.")
    .def("refresh_distance_matrix", &dtwc::Problem::refresh_distance_matrix)
    .def("read_distance_matrix", [](dtwc::Problem &p, const std::filesystem::path &path) {
      nb::gil_scoped_release release;
      p.read_distance_matrix(path);
    }, "path"_a,
         "Read a distance matrix from a CSV file (REPLACES this Problem's matrix).")
    .def("print_distance_matrix", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.print_distance_matrix();
    })
    .def("use_mmap_distance_matrix",
         [](dtwc::Problem &p, const std::filesystem::path &cache_path) {
           p.use_mmap_distance_matrix(cache_path);
         }, "cache_path"_a,
         "Back the distance matrix with a memory-mapped cache file (big-N / resume).")
    // ---- clustering ----
    .def("cluster", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.cluster();
    }, "Run clustering (Lloyd k-medoids or MIP).")
    .def("find_total_cost", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      return p.find_total_cost();
    }, "Total cost of the current cluster assignment.\n\n"
       "Raises InvalidInput if the Problem holds no clustering.")
    .def("assign_clusters", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.assign_clusters();
    })
    .def("calculate_medoids", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.calculate_medoids();
    })
    // ---- I/O ----
    .def("print_clusters", &dtwc::Problem::print_clusters)
    .def("write_clusters", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.write_clusters();
    }, "Write the cluster-assignment CSV.\n\n"
       "Raises InvalidInput if the Problem holds no clustering.")
    .def("write_medoid_members", &dtwc::Problem::write_medoid_members, "iter"_a, "rep"_a = 0)
    .def("write_distance_matrix", [](const dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.write_distance_matrix();
    })
    .def("write_silhouettes", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.write_silhouettes();
    }, "Write per-series silhouette scores.")
    .def("__repr__", [](const dtwc::Problem &p) {
      return "Problem(name='" + p.name() + "', n=" + std::to_string(p.size())
             + ", k=" + std::to_string(p.n_clusters()) + ")";
    });

  // =========================================================================
  // Distance matrix convenience function
  // =========================================================================

  m.def("compute_distance_matrix", [](const std::vector<std::vector<double>> &series,
                                        int band, const std::string &metric) {
    const auto mt = dtwc::parse_name(dtwc::core::metric_names, metric, "metric");
    require_finite_series(series, "compute_distance_matrix");

    // Warn once under OMP_NUM_THREADS=1, deterministically, before either branch
    // (the pruned branch also warns via get_max_threads; this covers the
    // unpruned one).
    dtwc::warn_if_single_threaded();

    const size_t n = series.size();
    // Owned buffer instead of a raw new[]: anything throwing between the
    // allocation and the capsule used to leak the whole N^2 matrix.
    std::vector<double> values(n * n, 0.0);
    double *ptr = values.data();

    // One exception_ptr slot per OpenMP thread. Each thread writes only its
    // own slot, so the error path stays lock-free: an `omp critical` around a
    // shared flag would serialise the hot loop and is the wrong fix.
#ifdef _OPENMP
    const int n_error_slots = std::max(1, omp_get_max_threads());
#else
    const int n_error_slots = 1;
#endif
    std::vector<std::exception_ptr> errors(static_cast<size_t>(n_error_slots));

    // Release GIL only for the compute-heavy section
    {
      nb::gil_scoped_release release;

      // Lock-free by design: each thread owns a disjoint set of rows (outer loop i).
      // Writes to ptr[i*n+j] and ptr[j*n+i] never collide across threads because
      // no two threads share the same i value.
      // num_threads pins the team to the number of slots sized above, so
      // omp_get_thread_num() can never index past `errors`.
      #ifdef _OPENMP
      #pragma omp parallel for schedule(dynamic, 16) num_threads(n_error_slots)
      #endif
      for (dtwc::index_t i = 0; i < static_cast<dtwc::index_t>(n); ++i) {
#ifdef _OPENMP
          const size_t slot = static_cast<size_t>(omp_get_thread_num());
#else
          const size_t slot = 0;
#endif
          // The input was checked above and the kernels do not check it, but
          // an exception escaping an OpenMP region is undefined behaviour and
          // terminates the process — a hard interpreter crash instead of a
          // Python exception — so any that does throw is carried out per thread.
          if (errors[slot]) continue;
          try {
            for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
                double d = (band >= 0)
                    ? dtwc::dtwBanded<double>(series[i], series[j], band, -1.0, mt)
                    : dtwc::dtwFull_L<double>(series[i], series[j], -1.0, mt);
                ptr[i * n + j] = d;
                ptr[j * n + i] = d;
            }
          } catch (...) {
            errors[slot] = std::current_exception();
          }
      }
    }  // GIL re-acquired here

    for (const auto &error : errors)
      if (error) std::rethrow_exception(error);

    return adopt_as_ndarray(std::move(values), {n, n});
  }, "series"_a, "band"_a = -1, "metric"_a = "l1",
     "Compute pairwise DTW distance matrix entirely in C++.\n\n"
     "Returns NxN numpy array. Uses OpenMP parallelism when available.\n"
     "This avoids a Python-level pair loop.\n"
     "NaN or +-inf in a series raises InvalidInput.");

  // =========================================================================
  // FastPAM
  // =========================================================================

  m.def("fast_pam", [](dtwc::Problem &prob, dtwc::index_t n_clusters, int max_iter) {
    nb::gil_scoped_release release;
    return dtwc::fast_pam(prob, n_clusters, max_iter);
  }, "prob"_a, "n_clusters"_a, "max_iter"_a = 100,
     "Run FastPAM k-medoids clustering (Schubert & Rousseeuw 2021).\n\n"
     "The C++ core writes labels/medoids/k back into prob (since 1.6), so\n"
     "silhouette(prob) and davies_bouldin(prob) work after this call with no\n"
     "wrapper-side wiring (api-contract-2.0.md §2.5).\n\n"
     "max_iter is the SWAP budget: 0 returns the BUILD medoids without a SWAP\n"
     "(converged is False); a negative count raises InvalidInput.");

  m.def("fast_pam_seeded",
        [](dtwc::Problem &prob, dtwc::index_t n_clusters, std::uint64_t seed, int max_iter) {
    nb::gil_scoped_release release;
    return dtwc::fast_pam_seeded(prob, n_clusters, seed, max_iter);
  }, "prob"_a, "n_clusters"_a, "seed"_a, "max_iter"_a = 100,
     "Run FastPAM with an invocation-local deterministic BUILD seed.\n\n"
     "max_iter reads as in fast_pam: 0 is BUILD only, a negative count raises\n"
     "InvalidInput.");

  // =========================================================================
  // FastCLARA
  // =========================================================================

  nb::class_<dtwc::algorithms::CLARAOptions>(m, "CLARAOptions")
    .def(nb::init<>())
    .def_rw("n_clusters", &dtwc::algorithms::CLARAOptions::n_clusters)
    .def_rw("sample_size", &dtwc::algorithms::CLARAOptions::sample_size)
    .def_rw("n_samples", &dtwc::algorithms::CLARAOptions::n_samples)
    .def_rw("max_iter", &dtwc::algorithms::CLARAOptions::max_iter)
    .def_rw("random_seed", &dtwc::algorithms::CLARAOptions::random_seed)
    .def("__repr__", [](const dtwc::algorithms::CLARAOptions &o) {
      return "CLARAOptions(k=" + std::to_string(o.n_clusters)
             + ", sample_size=" + std::to_string(o.sample_size)
             + ", n_samples=" + std::to_string(o.n_samples)
             + ", max_iter=" + std::to_string(o.max_iter)
             + ", seed=" + std::to_string(o.random_seed) + ")";
    });

  m.def("fast_clara", [](dtwc::Problem &prob, dtwc::index_t n_clusters,
                           dtwc::index_t sample_size, int n_samples, int max_iter,
                           std::uint64_t seed) {
    dtwc::algorithms::CLARAOptions opts;
    opts.n_clusters = n_clusters;
    opts.sample_size = sample_size;
    opts.n_samples = n_samples;
    opts.max_iter = max_iter;
    opts.random_seed = seed;
    nb::gil_scoped_release release;
    return dtwc::algorithms::fast_clara(prob, opts);
  }, "prob"_a, "n_clusters"_a, "sample_size"_a = -1,
     "n_samples"_a = 5, "max_iter"_a = 100,
     "seed"_a = dtwc::settings::DEFAULT_RANDOM_SEED,
     "Run FastCLARA scalable k-medoids clustering.\n\n"
     "Runs FastPAM on random subsamples and assigns all points to the\n"
     "best medoids found. Avoids O(N^2) memory of full PAM.\n\n"
     "The C++ core writes labels/medoids/k back into prob (since 1.6), so\n"
     "silhouette(prob) and davies_bouldin(prob) work after this call (§2.5).\n\n"
     "Parameters:\n"
     "  prob: Problem with data loaded.\n"
     "  n_clusters: Number of clusters (k).\n"
     "  sample_size: Subsample size (-1 = auto: max(40+2*k, min(N, 10*k+100))).\n"
     "  n_samples: Number of subsamples to try (default 5).\n"
     "  max_iter: Max PAM iterations per subsample (default 100).\n"
     "  seed: Random seed for reproducibility (default 42).");

  // =========================================================================
  // OneBatchPAM
  // =========================================================================

  m.def("one_batch_pam", [](dtwc::Problem &prob, dtwc::index_t n_clusters,
                              dtwc::index_t batch_size, int max_iter, std::uint64_t seed) {
    dtwc::algorithms::OneBatchPAMOptions options;
    options.n_clusters = n_clusters;
    options.batch_size = batch_size;
    options.max_iter = max_iter;
    options.random_seed = seed;
    nb::gil_scoped_release release;
    return dtwc::algorithms::one_batch_pam(prob, options);
  }, "prob"_a, "n_clusters"_a, "batch_size"_a = -1, "max_iter"_a = 100,
     "seed"_a = dtwc::settings::DEFAULT_RANDOM_SEED,
     "Run OneBatchPAM using one fixed N-by-m distance table (AAAI 2025).");

  // =========================================================================
  // DTW barycenters
  // =========================================================================

  m.def("dtw_barycenter",
        [](const dtwc::Problem &prob, const std::vector<dtwc::index_t> &indices,
           std::size_t target_length, const dtwc::algorithms::BarycenterOptions &options) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::dtw_barycenter(prob, indices, target_length, options);
  }, "prob"_a, "series_indices"_a, "target_length"_a,
     "options"_a = dtwc::algorithms::BarycenterOptions{},
     "Compute an SSG, DBA, or soft-DTW barycenter for selected series.");

  m.def("barycenter_kmeans",
        [](const dtwc::Problem &prob,
           const dtwc::algorithms::BarycenterClusteringOptions &options) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::barycenter_kmeans(prob, options);
  }, "prob"_a, "options"_a = dtwc::algorithms::BarycenterClusteringOptions{},
     "Cluster univariate series around sequence-valued DTW barycenters.");

  // =========================================================================
  // Checkpointing
  // =========================================================================

  nb::class_<dtwc::CheckpointOptions>(m, "CheckpointOptions",
    "Automatic mid-fill checkpointing options, read by\n"
    "Problem.fill_distance_matrix() through Problem.checkpoint.")
    .def(nb::init<>())
    .def_rw("directory", &dtwc::CheckpointOptions::directory,
            "Checkpoint directory. Empty with enabled raises InvalidInput.")
    .def_rw("save_interval", &dtwc::CheckpointOptions::save_interval,
            "Completed matrix ROWS between automatic saves (>= 1). The fill\n"
            "runs consecutive row blocks of this size and saves after each\n"
            "block, the last included, so a completed fill leaves a complete\n"
            "checkpoint. A value below 1 with enabled raises InvalidInput.")
    .def_rw("enabled", &dtwc::CheckpointOptions::enabled,
            "Enable automatic mid-fill checkpointing to <directory>/<name>.dtwm.\n"
            "A matrix mapped to that file is flushed in place. An empty\n"
            "directory raises InvalidInput before any distance is computed.")
    .def("__repr__", [](const dtwc::CheckpointOptions &o) {
      return "CheckpointOptions(dir='" + o.directory
             + "', interval=" + std::to_string(o.save_interval)
             + ", enabled=" + (o.enabled ? "True" : "False") + ")";
    });

  m.def("save_checkpoint", [](const dtwc::Problem &prob,
                              const std::string &path,
                              dtwc::core::MetricType metric) {
        // An N^2 write; `prob` is const here and the writer only reads it.
        nb::gil_scoped_release release;
        dtwc::save_checkpoint(prob, path, metric);
      }, "prob"_a, "path"_a, "metric"_a = dtwc::core::MetricType::L1,
        "Save the distance matrix to <path>/<name>.dtwm ('distances' for an\n"
        "unnamed Problem), replacing any previous checkpoint there. The\n"
        "directory is created if it does not exist; IOError if it cannot be.\n\n"
        "`metric` is the pointwise metric the stored distances were computed\n"
        "with. It is part of the identity fingerprint, so a SquaredL2 matrix is\n"
        "no longer accepted by a later L1 load. Mirrors the CLI's --metric and\n"
        "defaults to L1 for backward compatibility.");

  m.def("load_checkpoint", [](dtwc::Problem &prob,
                              const std::string &path,
                              dtwc::core::MetricType metric) {
        nb::gil_scoped_release release;
        return dtwc::load_checkpoint(prob, path, metric);
      }, "prob"_a, "path"_a, "metric"_a = dtwc::core::MetricType::L1,
        "Load <path>/<name>.dtwm into the Problem's distance matrix.\n\n"
        "Returns True when loaded, False only when there is no such file.\n"
        "Raises InvalidInput for a checkpoint of other series or other\n"
        "distance settings, and IOError for a file that is not a whole .dtwm\n"
        "file; neither changes the Problem.\n\n"
        "`metric` is the pointwise metric THIS run computes with: a checkpoint\n"
        "written under a different metric does not match the identity\n"
        "fingerprint (InvalidInput). Mirrors the CLI's --metric; defaults to\n"
        "L1 for backward compatibility.\n\n"
        "MUTATES `prob`: do not run it concurrently with any other method on\n"
        "the same Problem (see the Problem class docstring).");

  // =========================================================================
  // Scores
  // =========================================================================

  // Score names drop the `Index`/`Information` noun (api-contract-2.0.md §2.4).
  m.def("silhouette", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::silhouette(prob);
  }, "prob"_a, "Compute silhouette score for each data point.");

  m.def("davies_bouldin", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::davies_bouldin(prob);
  }, "prob"_a, "Compute Davies-Bouldin index (lower is better).");

  m.def("dunn", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::dunn(prob);
  }, "prob"_a,
     "Compute Dunn index (min inter-cluster distance / max intra-cluster diameter).");

  m.def("inertia", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::inertia(prob);
  }, "prob"_a,
     "Compute inertia (total within-cluster distance sum to medoids).");

  m.def("calinski_harabasz", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::calinski_harabasz(prob);
  }, "prob"_a, "Compute Calinski-Harabasz index (medoid-adapted; higher is better).");

  m.def("adjusted_rand", [](const std::vector<dtwc::index_t> &labels_true,
                            const std::vector<dtwc::index_t> &labels_pred) {
    return dtwc::scores::adjusted_rand(labels_true, labels_pred);
  }, "labels_true"_a, "labels_pred"_a,
     "Adjusted Rand index between two label assignments (1.0 = perfect agreement).");

  m.def("normalized_mutual_info", [](const std::vector<dtwc::index_t> &labels_true,
                                      const std::vector<dtwc::index_t> &labels_pred) {
    return dtwc::scores::normalized_mutual_info(labels_true, labels_pred);
  }, "labels_true"_a, "labels_pred"_a,
     "Normalized Mutual Information between two label assignments ([0,1]).");

  // =========================================================================
  // Hierarchical clustering
  // =========================================================================

  m.def("build_dendrogram", [](dtwc::Problem &prob,
                                 const dtwc::algorithms::HierarchicalOptions &opts) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::build_dendrogram(prob, opts);
  }, "prob"_a, "opts"_a = dtwc::algorithms::HierarchicalOptions{},
     "Build a hierarchical dendrogram from a Problem.\n\n"
     "Fills the distance matrix first if it is not filled.\n"
     "Returns a Dendrogram containing N-1 merge steps in merge order.\n"
     "Raises InvalidInput (a ValueError) if N > opts.max_points (default 2000).");

  m.def("cut_dendrogram", [](const dtwc::algorithms::Dendrogram &dend,
                               dtwc::Problem &prob, dtwc::index_t k) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::cut_dendrogram(dend, prob, k);
  }, "dendrogram"_a, "prob"_a, "k"_a,
     "Cut a dendrogram to produce k flat clusters.\n\n"
     "Returns a ClusteringResult with labels, medoid_indices, and total_cost.\n"
     "The C++ core also writes labels/medoids/k back into prob (since 1.6), so\n"
     "silhouette(prob) etc. work after this call with no wrapper wiring (§2.5).");


  // =========================================================================
  // CUDA (optional)
  // =========================================================================

#ifdef DTWC_HAS_CUDA
  m.def("cuda_available", &dtwc::cuda::cuda_available,
        "Check if a CUDA-capable GPU is available.");

  m.def("cuda_device_info", &dtwc::cuda::cuda_device_info,
        "device_id"_a = 0,
        "Get a human-readable string describing the CUDA device.");

  m.def("compute_distance_matrix_cuda",
        [](const std::vector<std::vector<double>> &series,
           int band, bool use_squared_l2, int device_id, bool verbose) {
          dtwc::cuda::CUDADistMatOptions opts;
          opts.band = band;
          opts.use_squared_l2 = use_squared_l2;
          opts.device_id = device_id;
          opts.verbose = verbose;
          require_finite_series(series, "compute_distance_matrix_cuda");
          std::vector<double> matrix;
          const size_t n = series.size();
          {
            nb::gil_scoped_release release;
            dtwc::core::DistanceMatrix packed;
            dtwc::cuda::compute_distance_matrix_cuda(series, opts, packed);
            matrix = dtwc::io::to_full_matrix(packed); // row-major, expanded from the triangle
          }
          return adopt_as_ndarray(std::move(matrix), {n, n});
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "device_id"_a = 0, "verbose"_a = false,
        "Compute NxN DTW distance matrix on CUDA GPU.\n\n"
        "Returns NxN numpy array of DTW distances. A pair with no warping\n"
        "path under `band` reads the finite double-max sentinel, not IEEE\n"
        "infinity.\n"
        "NaN or +-inf in a series raises InvalidInput.");

  m.attr("CUDA_AVAILABLE") = true;
#else
  m.def("cuda_available", []() { return false; },
        "Check if CUDA GPU is available.");

  m.def("cuda_device_info", [](int) { return std::string("CUDA not available (not compiled)"); },
        "device_id"_a = 0,
        "Get CUDA device info string.");

  m.def("compute_distance_matrix_cuda",
        [](const std::vector<std::vector<double>> &, int, bool, int, bool) -> nb::object {
          throw dtwc::DeviceError("CUDA support not compiled. Rebuild with -DDTWC_ENABLE_CUDA=ON");
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "device_id"_a = 0, "verbose"_a = false,
        "Compute NxN DTW distance matrix on CUDA GPU (requires CUDA build).");

  m.attr("CUDA_AVAILABLE") = false;
#endif

  // =========================================================================
  // Metal (optional, Apple GPU)
  // =========================================================================

#ifdef DTWC_HAS_METAL
  m.def("metal_available", &dtwc::metal::metal_available,
        "Check if a Metal-capable GPU is available (macOS only).");

  m.def("metal_device_info", &dtwc::metal::metal_device_info,
        "Get a human-readable string describing the Metal device.");

  m.def("compute_distance_matrix_metal",
        [](const std::vector<std::vector<double>> &series,
           int band, bool use_squared_l2, bool verbose) {
          dtwc::metal::MetalDistMatOptions opts;
          opts.band = band;
          opts.use_squared_l2 = use_squared_l2;
          opts.verbose = verbose;
          require_finite_series(series, "compute_distance_matrix_metal");
          std::vector<double> matrix;
          const size_t n = series.size();
          {
            nb::gil_scoped_release release;
            dtwc::core::DistanceMatrix packed;
            dtwc::metal::compute_distance_matrix_metal(series, opts, packed);
            matrix = dtwc::io::to_full_matrix(packed); // row-major, expanded from the triangle
          }
          return adopt_as_ndarray(std::move(matrix), {n, n});
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "verbose"_a = false,
        "Compute NxN DTW distance matrix on Apple GPU via Metal.\n\n"
        "Returns NxN numpy array of DTW distances.\n"
        "NaN or +-inf in a series raises InvalidInput.");

  m.attr("METAL_AVAILABLE") = true;
#else
  m.def("metal_available", []() { return false; },
        "Check if Metal GPU is available.");
  m.def("metal_device_info", []() { return std::string("Metal not available (not compiled)"); },
        "Get Metal device info string.");
  m.def("compute_distance_matrix_metal",
        [](const std::vector<std::vector<double>> &, int, bool, bool) -> nb::object {
          throw dtwc::DeviceError("Metal support not compiled. Rebuild on macOS with -DDTWC_ENABLE_METAL=ON");
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "verbose"_a = false,
        "Compute NxN DTW distance matrix on Apple GPU (requires Metal build).");
  m.attr("METAL_AVAILABLE") = false;
#endif

  // =========================================================================
  // Capability detection: OpenMP
  // =========================================================================

#ifdef _OPENMP
  m.attr("OPENMP_AVAILABLE") = true;
  m.def("openmp_max_threads", []() {
    return omp_get_max_threads();
  }, "Return the maximum number of OpenMP threads available.");
#else
  m.attr("OPENMP_AVAILABLE") = false;
  m.def("openmp_max_threads", []() { return 1; },
        "Return 1 (OpenMP not compiled in).");
#endif

  m.def("system_info", []() {
    std::string info;
    info += "DTWC++ System Information\n";
#ifdef _OPENMP
    info += "  OpenMP: available (" + std::to_string(omp_get_max_threads()) + " threads)\n";
#else
    info += "  OpenMP: not available\n";
#endif
#ifdef DTWC_HAS_CUDA
    if (dtwc::cuda::cuda_available())
      info += "  CUDA:   available (" + dtwc::cuda::cuda_device_info(0) + ")\n";
    else
      info += "  CUDA:   compiled but no GPU detected\n";
#else
    info += "  CUDA:   not compiled (rebuild with -DDTWC_ENABLE_CUDA=ON)\n";
#endif
#ifdef DTWC_HAS_METAL
    if (dtwc::metal::metal_available())
      info += "  Metal:  available (" + dtwc::metal::metal_device_info() + ")\n";
    else
      info += "  Metal:  compiled but no GPU detected\n";
#else
    info += "  Metal:  not compiled (macOS only)\n";
#endif
    return info;
  }, "Return a string summarizing available backends and capabilities.");

  // =========================================================================
  // dtwc.test introspection API (Task 3.3) — SAME schema/field names as the C++
  // dtwc::test::* structs and the MATLAB dtwc_mex('test_*') structs. Each returns
  // a dict so `except`-free front-end code reads identical keys in all three
  // languages. The heavy probe (a real OpenMP region / a real GPU kernel) runs
  // with the GIL released, then the dict is built under the GIL.
  // =========================================================================

  m.def("test_parallelisation", []() {
    dtwc::test::ParallelReport r;
    {
      nb::gil_scoped_release release;
      r = dtwc::test::parallelisation();
    }
    nb::dict d;
    d["available"] = r.available;
    d["max_threads"] = r.max_threads;
    d["threads_engaged"] = r.threads_engaged;
    d["pass"] = r.pass;
    d["reason"] = r.reason;
    return d;
  }, "Run a REAL OpenMP parallel region and report DISTINCT engaged thread ids.\n\n"
     "Returns a dict {available, max_threads, threads_engaged, pass, reason} — the\n"
     "same schema as C++ dtwc::test::parallelisation() and MATLAB\n"
     "dtwc_mex('test_parallelisation'). Proof-of-engagement, not a flag read: on a\n"
     "sequential build available=False with a non-empty reason (never raises).");

  m.def("test_gpu", []() {
    dtwc::test::GpuReport r;
    {
      nb::gil_scoped_release release;
      r = dtwc::test::gpu();
    }
    nb::dict d;
    d["available"] = r.available;
    d["backend"] = r.backend;
    d["device_name"] = r.device_name;
    d["validated"] = r.validated;
    d["pass"] = r.pass;
    d["reason"] = r.reason;
    return d;
  }, "Execute a tiny REAL GPU kernel and validate it against a CPU oracle.\n\n"
     "Returns a dict {available, backend, device_name, validated, pass, reason} —\n"
     "the same schema as C++ dtwc::test::gpu() and MATLAB dtwc_mex('test_gpu').\n"
     "When no GPU backend is compiled in (or no device is present) available=False\n"
     "and reason names exactly what is missing (never raises, never silently\n"
     "degrades).");
}
