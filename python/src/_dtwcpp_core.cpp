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
#include <mip/decode_assignment.hpp>


#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
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

/// Raise `type` with `what`: a reader's message quotes the token it refused,
/// which need not be UTF-8 (a Latin-1 0xA0), so invalid bytes are replaced
/// instead of turning the error into a UnicodeDecodeError.
void set_error(PyObject *type, const char *what)
{
  PyObject *message = PyUnicode_DecodeUTF8(what, static_cast<Py_ssize_t>(std::strlen(what)), "replace");
  if (message == nullptr) return; // the decoder's own error stays set
  PyErr_SetObject(type, message);
  Py_DECREF(message);
}

/// Problem.variant_params' type: its fields are bound read-only.
struct FrozenParams : dtwc::core::DTWVariantParams
{};

/// A Config field reached through `ref`, read and written as itself.
template <class T, class Ref>
void def_field(nb::class_<dtwc::Config> &config, const char *key, Ref ref)
{
  config.def_prop_rw(key, [ref](dtwc::Config &c) -> T { return ref(c); },
                     [ref](dtwc::Config &c, T value) { ref(c) = value; });
}

/// A Config enum reached through `ref`, read as its canonical name and written
/// by any spelling `table` lists; an unknown name raises InvalidInput.
template <class E, std::size_t N, class Ref>
void def_named(nb::class_<dtwc::Config> &config, const char *key, Ref ref, const dtwc::Name<E> (&table)[N])
{
  config.def_prop_rw(
    key, [ref, &table](dtwc::Config &c) { return std::string(dtwc::name_of(table, ref(c))); },
    [ref, &table, key](dtwc::Config &c, const std::string &text) { ref(c) = dtwc::parse_name(table, text, key); });
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
  // Error taxonomy
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
        set_error(g_exc_undefined_score, e.what());
      } catch (const dtwc::InvalidInput &e) {
        set_error(g_exc_invalid, e.what());
      } catch (const dtwc::SolverError &e) {
        set_error(g_exc_solver, e.what());
      } catch (const dtwc::DeviceError &e) {
        set_error(g_exc_device, e.what());
      } catch (const dtwc::IOError &e) {
        set_error(g_exc_io, e.what());
      } catch (const dtwc::Error &e) {
        set_error(g_exc_base, e.what());
      }
    });

  // =========================================================================
  // Device
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
  // Tier-1 file parsing
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
     "dtwcpp.load reads Parquet and Arrow IPC through pyarrow), and return the owning\n"
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

  m.def("_default_name", &dtwc::detail::default_name, "path"_a,
        "The name dtwc_cl gives a run of `path` when none is given: the file's name\n"
        "without its extension, or the folder's name (a trailing separator\n"
        "included); 'dataset' for an empty path (series in memory).");

  // =========================================================================
  // Enums
  // =========================================================================

  nb::enum_<dtwc::Method>(m, "Method")
    .value("Kmedoids", dtwc::Method::Kmedoids)
    .value("MIP", dtwc::Method::MIP)
    .value("LRCore", dtwc::Method::LRCore)
    .value("TADPole", dtwc::Method::TADPole)
    .value("Auto", dtwc::Method::Auto)
    .value("PAM", dtwc::Method::PAM)
    .value("OneBatch", dtwc::Method::OneBatch)
    .value("CLARA", dtwc::Method::CLARA)
    .value("Hierarchical", dtwc::Method::Hierarchical);

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

  nb::enum_<dtwc::GpuPrecision>(m, "GpuPrecision")
    .value("Auto", dtwc::GpuPrecision::Auto)
    .value("FP32", dtwc::GpuPrecision::FP32)
    .value("FP64", dtwc::GpuPrecision::FP64);

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
    .def_rw("target_length", &dtwc::algorithms::BarycenterClusteringOptions::target_length)
    .def_rw("barycenter", &dtwc::algorithms::BarycenterClusteringOptions::barycenter);

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

  // What Problem.variant_params returns: a DTWVariantParams whose fields cannot be
  // written, so `prob.variant_params.wdtw_g = 0.1` raises instead of editing a copy
  // the Problem never sees. Assigning it back to variant_params works.
  nb::class_<FrozenParams, Params>(m, "FrozenDTWVariantParams")
    .def_prop_ro("variant", [](const FrozenParams &p) { return p.variant; })
    .def_prop_ro("wdtw_g", [](const FrozenParams &p) { return p.wdtw_g; })
    .def_prop_ro("adtw_penalty", [](const FrozenParams &p) { return p.adtw_penalty; })
    .def_prop_ro("sdtw_gamma", [](const FrozenParams &p) { return p.sdtw_gamma; })
    .def_prop_ro("msm_c", [](const FrozenParams &p) { return p.msm_c; })
    .def_prop_ro("twe_nu", [](const FrozenParams &p) { return p.twe_nu; })
    .def_prop_ro("twe_lambda", [](const FrozenParams &p) { return p.twe_lambda; })
    .def_prop_ro("mv_mode", [](const FrozenParams &p) { return p.mv_mode; });

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
  // Config: the settings of a clustering, keyed by the CLI long names in
  // snake_case, read and checked by C++ (dtwc/config.hpp). The file options
  // (input, output, checkpoint, ...) are dtwc_cl's.
  // =========================================================================

  using dtwc::Config;
  nb::class_<Config> config(m, "Config");
  config.def(nb::init<>())
    .def_rw("n_clusters", &Config::k)
    .def_rw("max_iter", &Config::max_iter)
    .def_rw("n_init", &Config::n_init)
    .def_rw("seed", &Config::seed)
    .def_rw("sample_size", &Config::sample_size)
    .def_rw("n_samples", &Config::n_samples)
    .def_rw("batch_size", &Config::batch_size)
    .def_rw("dc", &Config::tadpole_dc)
    .def_rw("band", &Config::band)
    .def_rw("name", &Config::name)
    .def_rw("verbose", &Config::verbose)
    .def_prop_rw("device", [](const Config &c) { return dtwc::device_text(c); },
                 [](Config &c, const std::string &name) {
                   std::tie(c.device, c.device_index) = dtwc::detail::parse_device(name);
                 })
    .def_prop_rw("no_warm_start", [](const Config &c) { return !c.mip.warm_start; },
                 [](Config &c, bool value) { c.mip.warm_start = !value; });
  def_named(config, "method", [](Config &c) -> dtwc::Method & { return c.method; }, dtwc::method_names);
  def_named(config, "metric", [](Config &c) -> dtwc::core::MetricType & { return c.metric; },
            dtwc::core::metric_names);
  def_named(config, "variant", [](Config &c) -> dtwc::core::DTWVariant & { return c.variant.variant; },
            dtwc::core::variant_names);
  def_named(config, "mv_mode", [](Config &c) -> dtwc::core::MVMode & { return c.variant.mv_mode; },
            dtwc::core::mv_mode_names);
  def_named(config, "missing_strategy", [](Config &c) -> dtwc::core::MissingStrategy & { return c.missing; },
            dtwc::core::missing_strategy_names);
  def_named(config, "linkage", [](Config &c) -> dtwc::algorithms::Linkage & { return c.linkage; },
            dtwc::algorithms::linkage_names);
  def_named(config, "solver", [](Config &c) -> dtwc::Solver & { return c.solver; }, dtwc::solver_names);
  def_named(config, "gpu_precision", [](Config &c) -> dtwc::GpuPrecision & { return c.gpu_precision; },
            dtwc::gpu_precision_names);
  def_field<double>(config, "wdtw_g", [](Config &c) -> double & { return c.variant.wdtw_g; });
  def_field<double>(config, "adtw_penalty", [](Config &c) -> double & { return c.variant.adtw_penalty; });
  def_field<double>(config, "sdtw_gamma", [](Config &c) -> double & { return c.variant.sdtw_gamma; });
  def_field<double>(config, "msm_c", [](Config &c) -> double & { return c.variant.msm_c; });
  def_field<double>(config, "twe_nu", [](Config &c) -> double & { return c.variant.twe_nu; });
  def_field<double>(config, "twe_lambda", [](Config &c) -> double & { return c.variant.twe_lambda; });
  def_field<double>(config, "mip_gap", [](Config &c) -> double & { return c.mip.mip_gap; });
  def_field<int>(config, "time_limit", [](Config &c) -> int & { return c.mip.time_limit_sec; });
  def_field<int>(config, "numeric_focus", [](Config &c) -> int & { return c.mip.numeric_focus; });
  def_field<int>(config, "mip_focus", [](Config &c) -> int & { return c.mip.mip_focus; });
  def_field<bool>(config, "verbose_solver", [](Config &c) -> bool & { return c.mip.verbose_solver; });
  def_field<std::int64_t>(config, "lr_max_nodes", [](Config &c) -> std::int64_t & { return c.mip.lr_max_nodes; });

  m.def("apply", &dtwc::apply, "config"_a, "prob"_a,
        "Hand prob the clustering settings of config: the distance, the method and its\n"
        "controls, the MIP solver, the device and verbose. The Problem checks each as\n"
        "it takes it (InvalidInput, SolverError, DeviceError).");

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
  // configuration no kernel implements, then x and y are scanned once (an empty
  // series, NaN or ±inf raises InvalidInput naming x or y and the position; a
  // missing-data strategy reads NaN as missing). The arrays are read in place,
  // and the GIL is released for the computation.
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
     "combination no kernel implements, an empty series, or a value the strategy\n"
     "does not take.");

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
                "Float32 heap-mode data (explicit opt-in; halves storage). The series\n"
                "are rounded to float32 and the DTW recurrence runs in float32; the\n"
                "distances are returned and stored as double.")
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
       "Compute on `device` (the names dtwcpp.device() accepts): 'gpu:N' is GPU N\n"
       "of this build's GPU backend (CUDA, else Metal, which has GPU 0 only). A\n"
       "request the device cannot honour (a variant, missing-data strategy,\n"
       "multivariate data or precision its kernels lack) raises DeviceError when\n"
       "distances are computed.")
    .def("set_gpu_precision", &dtwc::Problem::set_gpu_precision, "precision"_a,
         "What a GPU computes in: GpuPrecision.Auto (the default; FP32 on consumer\n"
         "CUDA GPUs and on Metal), FP32 or FP64. A change drops the distance matrix.")
    // ---- config properties (canonical names) ----
    .def_prop_rw("method", &dtwc::Problem::method, &dtwc::Problem::set_method)
    .def_prop_ro("solver", &dtwc::Problem::solver,
                 "The MIP solver of Method.MIP: Solver.HiGHS unless set_solver chose another.")
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
                 [](const dtwc::Problem &p) { return FrozenParams{ p.variant_params() }; },
                 [](dtwc::Problem &p, dtwc::core::DTWVariantParams value) {
                   p.set_variant(value);
                 },
                 "The DTW variant and its parameters, read-only: assign a DTWVariantParams\n"
                 "(or call set_variant_params) to change them, which drops the matrix.")
    .def_prop_rw("missing_strategy",
                 [](const dtwc::Problem &p) { return p.missing_strategy(); },
                 [](dtwc::Problem &p, dtwc::core::MissingStrategy value) {
                   p.set_missing_strategy(value);
                 },
                 "Strategy for handling NaN values (Error, ZeroCost, AROW, Interpolate).")
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
    // view into a full NxN layout is structurally impossible.
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
           // Values written into a mapped matrix would persist in its file under the
           // Problem's fingerprint, whatever they were computed from.
           if (p.distance_matrix().is_mapped())
             throw dtwc::InvalidInput("Problem.set_distance_matrix: this Problem's distance matrix is "
                                      "memory-mapped (use_mmap_distance_matrix), and a supplied matrix is "
                                      "kept in RAM only; call refresh_distance_matrix() first.");
           auto &mat = p.writable_distance_matrix();
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
      return p.cluster();
    }, "Cluster the series into n_clusters() by Problem.method (Auto: PAM on a GPU and\n"
       "for up to 5000 series, else CLARA) and return the ClusteringResult; the labels\n"
       "and medoids are published on the Problem too.")
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

  m.def("_write_result_files", [](dtwc::Problem &prob, const std::filesystem::path &directory) {
    nb::gil_scoped_release release;
    dtwc::detail::write_result_files(prob, directory, true);
  }, "prob"_a, "directory"_a,
     "Write a clustered Problem's four result files into directory, as\n"
     "dtwc_cl and C++ Result::save write them (detail::write_result_files).");

  // =========================================================================
  // Method.MIP without linked HiGHS (the wheel): dtwcpp._mip hands the model's
  // arrays to highspy as read-only NumPy views of the C++ model, which each
  // view keeps alive, and _mip_set_solution decodes highspy's solution.
  // =========================================================================

  using dtwc::mip::PMedianModel;
  const auto view = [](auto member) {
    return [member](const PMedianModel &model) {
      const auto &values = model.*member;
      using T = typename std::remove_cvref_t<decltype(values)>::value_type;
      return nb::ndarray<nb::numpy, const T, nb::ndim<1>>(values.data(), { values.size() });
    };
  };
  constexpr auto internal = nb::rv_policy::reference_internal;
  nb::class_<PMedianModel>(m, "_MIPModel",
                           "The compact p-median MIP of a Problem as the arrays HiGHS takes\n"
                           "(dtwc::mip::PMedianModel); start is empty without a warm start.")
    .def_ro("num_col", &PMedianModel::num_col)
    .def_ro("num_row", &PMedianModel::num_row)
    .def_prop_ro("col_cost", view(&PMedianModel::col_cost), internal)
    .def_prop_ro("col_lower", view(&PMedianModel::col_lower), internal)
    .def_prop_ro("col_upper", view(&PMedianModel::col_upper), internal)
    .def_prop_ro("row_lower", view(&PMedianModel::row_lower), internal)
    .def_prop_ro("row_upper", view(&PMedianModel::row_upper), internal)
    .def_prop_ro("a_start", view(&PMedianModel::a_start), internal)
    .def_prop_ro("a_index", view(&PMedianModel::a_index), internal)
    .def_prop_ro("a_value", view(&PMedianModel::a_value), internal)
    .def_prop_ro("integrality", view(&PMedianModel::integrality), internal)
    .def_prop_ro("start", view(&PMedianModel::start), internal);

  m.def("_mip_model", [](dtwc::Problem &prob) {
    dtwc::require_clusterable(prob.n_clusters(), static_cast<std::size_t>(prob.size())); // as Problem::cluster()
    dtwc::validate_mip_settings(prob.mip_settings);                                       // and its MIP route
    nb::gil_scoped_release release;
    return dtwc::mip::build_p_median_model(prob);
  }, "prob"_a,
     "Fill prob's distance matrix and return its compact p-median MIP as arrays\n"
     "(dtwc::mip::build_p_median_model), the model linked HiGHS solves.");

  m.def("_mip_set_solution", [](dtwc::Problem &prob, nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x) {
    const auto n = static_cast<std::size_t>(prob.size());
    if (x.size() != n * n)
      throw dtwc::InvalidInput("_mip_set_solution: expected N * N = " + std::to_string(n * n)
                               + " values, got " + std::to_string(x.size()) + ".");
    auto result = dtwc::mip::decode_assignment({ x.data(), x.size() }, n, prob.n_clusters(), false, "highspy");
    prob.set_result(result);
    result.total_cost = prob.find_total_cost();
    result.converged = true; // as Problem::cluster() returns an exact method's result
    return result;
  }, "prob"_a, "x"_a,
     "Decode a solution of prob's _mip_model (dtwc::mip::decode_assignment), publish\n"
     "it on prob and return the ClusteringResult Problem.cluster() returns.");

  // =========================================================================
  // FastPAM
  // =========================================================================

  m.def("fast_pam",
        [](dtwc::Problem &prob, dtwc::index_t n_clusters, int max_iter, std::uint64_t seed) {
    nb::gil_scoped_release release;
    return dtwc::fast_pam(prob, n_clusters, max_iter, seed);
  }, "prob"_a, "n_clusters"_a, "max_iter"_a = 100,
     "seed"_a = dtwc::settings::DEFAULT_RANDOM_SEED,
     "Run FastPAM k-medoids clustering (Schubert & Rousseeuw 2021).\n\n"
     "The C++ core writes labels/medoids/k back into prob (since 1.6), so\n"
     "silhouette(prob) and davies_bouldin(prob) work after this call with no\n"
     "wrapper-side wiring.\n\n"
     "max_iter is the SWAP budget: 0 returns the BUILD medoids without a SWAP\n"
     "(converged is False); a negative count raises InvalidInput. seed is the\n"
     "BUILD (k-medoids++) seed: one seed gives one result on every platform.");

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
     "silhouette(prob) and davies_bouldin(prob) work after this call.\n\n"
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

  // Score names drop the `Index`/`Information` noun.
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

  m.def("score", [](dtwc::Problem &prob, const std::string &name) {
    nb::gil_scoped_release release;
    return dtwc::scores::score(prob, name);
  }, "prob"_a, "name"_a,
     "The score `name` names: silhouette (the mean over the series), davies_bouldin,\n"
     "dunn, calinski_harabasz or inertia; any other name raises InvalidInput.");

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
     "silhouette(prob) etc. work after this call with no wrapper wiring.");


  // =========================================================================
  // GPU discovery: this build's backend, CUDA else Metal
  // =========================================================================

  m.def("gpu_available", &dtwc::gpu_available,
        "True when this build's GPU backend (CUDA, else Metal) finds a GPU, so\n"
        "device='gpu' can compute here.");
  m.def("gpu_info", &dtwc::gpu_info,
        "One line naming this build's GPU backend and the GPU device='gpu'\n"
        "computes on ('CUDA: <device>', 'Metal: <device>'), or why there is none.");

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
