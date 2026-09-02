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
#include <env.hpp>
#include <error.hpp>
#include <io/arrow_c_data.hpp>
#include <checkpoint.hpp>
#include <warping.hpp>
#include <warping_ddtw.hpp>
#include <warping_wdtw.hpp>
#include <warping_adtw.hpp>
#include <warping_missing.hpp>
#include <warping_missing_arow.hpp>
#include <soft_dtw.hpp>
#include <algorithms/fast_pam.hpp>
#include <algorithms/fast_clara.hpp>
#include <algorithms/one_batch_pam.hpp>
#include <algorithms/barycenter.hpp>
#include <algorithms/clarans.hpp>
#include <algorithms/hierarchical.hpp>
#include <scores.hpp>
#include <core/z_normalize.hpp>
#include <core/dtw_options.hpp>
#include <core/distance_semantics.hpp>
#include <core/variant_validation.hpp>
#include <core/pruned_distance_matrix.hpp>
#include <core/matrix_io.hpp>
#include <test_api.hpp> // dtwc::test::parallelisation()/gpu() introspection (Task 3.3)
#include <mip/mip.hpp>
#include <mip/pdlp_lp.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals; // for _a arg names

namespace {

void warn_deprecated_alias(const char *old_name, const char *new_name) {
  std::string message(old_name);
  message += " is deprecated; use ";
  message += new_name;
  if (PyErr_WarnEx(PyExc_DeprecationWarning, message.c_str(), 1) < 0)
    throw nb::python_error();
}

/// Hand an owning buffer to numpy with no leak window and no nested GIL scope.
///
/// The vector is moved onto the heap, the capsule is constructed while a
/// unique_ptr still owns it (so a throwing capsule allocation frees it), and
/// only then is ownership released to the capsule. Callers hold the GIL, so no
/// `gil_scoped_acquire` is nested inside a live release.
nb::ndarray<nb::numpy, double> adopt_as_ndarray(
  std::vector<double> &&values, std::initializer_list<size_t> shape) {
  // numpy never dereferences a zero-sized array, but nanobind still wants a
  // real address; an empty vector may report data() == nullptr.
  if (values.empty()) values.reserve(1);
  auto owned = std::make_unique<std::vector<double>>(std::move(values));
  double *ptr = owned->data();
  nb::capsule owner(owned.get(), [](void *p) noexcept {
    std::unique_ptr<std::vector<double>>(static_cast<std::vector<double> *>(p));
  });
  owned.release(); // the capsule owns the buffer from here on
  return nb::ndarray<nb::numpy, double>(ptr, shape, owner);
}

/// Move an N*N GPU result into an owned buffer, refusing a size mismatch.
/// A backend that returns fewer elements than n*n must not be zero-padded into
/// something that reads as a valid distance matrix.
std::vector<double> checked_square_matrix(
  std::vector<double> &&matrix, size_t n, const char *backend) {
  if (matrix.size() != n * n)
    throw dtwc::DeviceError(
      std::string(backend) + " returned " + std::to_string(matrix.size())
      + " distances for " + std::to_string(n) + " series; expected "
      + std::to_string(n * n) + ".");
  return std::move(matrix);
}

std::string utf8_path_text(const std::filesystem::path &path) {
  const std::u8string encoded = path.u8string();
  return std::string(
    reinterpret_cast<const char *>(encoded.data()), encoded.size());
}

} // namespace

NB_MODULE(_dtwcpp_core, m) {
  m.attr("__version__") = DTWC_VERSION_STRING;
  m.attr("_F22_DEPRECATION_POLICY") = true;
  m.attr("DEFAULT_RANDOM_SEED") = dtwc::settings::DEFAULT_RANDOM_SEED;
  m.attr("HIGHS_AVAILABLE") = dtwc::highs_solver_available();
  m.attr("PDLP_GPU_AVAILABLE") = dtwc::mip::pdlp_gpu_available();
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
  // Env / Device registry (api-contract-2.0.md §6)
  // =========================================================================

  nb::enum_<dtwc::Device>(m, "Device")
    .value("CPU", dtwc::Device::CPU)
    .value("GPU", dtwc::Device::GPU)
    .value("HPC", dtwc::Device::HPC);

  nb::class_<dtwc::Env>(m, "Env")
    .def("set_device", [](dtwc::Env &e, const std::string &name) { e.set_device(name); }, "name"_a,
         "Select the compute device (cpu/gpu/gpu:N/cuda/cuda:N/hpc). Raises\n"
         "DeviceError on an unknown name, gpu without a GPU backend, or any of the\n"
         "three device='hpc' .env failures — never a silent CPU fallback.")
    .def("device", &dtwc::Env::device, "Currently selected Device.")
    .def("device_index", &dtwc::Env::device_index, "GPU ordinal from the last gpu:N/cuda:N selection.")
    .def("threads", &dtwc::Env::threads, "Resolved thread count for parallel regions.")
    .def("set_env_file_dir",
         [](dtwc::Env &e, const std::filesystem::path &d) { e.set_env_file_dir(d); }, "dir"_a,
         "Directory searched for the device='hpc' .env file.")
    .def("env_file_dir", [](const dtwc::Env &e) { return e.env_file_dir(); });

  m.def("env", &dtwc::env, nb::rv_policy::reference,
        "Return the process-wide Env singleton (the shared device registry, §6).");

  m.def("device_to_string", [](dtwc::Device d) { return dtwc::to_string(d); }, "device"_a,
        "Canonical lower-case name of a Device ('cpu'/'gpu'/'hpc').");

  m.def("device", [](const std::string &name) {
        // Env::set_device probes GPU/HPC availability (device query, .env read,
        // sinfo) without touching Python; hold no GIL across it.
        nb::gil_scoped_release release;
        return dtwc::device(name);
      }, "name"_a,
        "Set the process-wide device and return its CANONICAL name\n"
        "('cpu'/'gpu'/'gpu:N'/'hpc'), exactly as dtwc::device(name) does.");

  m.def("device", []() { return dtwc::device(); },
        "Canonical name of the process-wide device (dtwc::device()).");

  // =========================================================================
  // Tier-1 file parsing (api-contract-2.0.md §1.2)
  // =========================================================================

  m.def("_read_data",
        [](const std::filesystem::path &source, int skip_cols, int skip_rows,
           const std::string &delimiter) {
    if (skip_cols < 0) throw dtwc::InvalidInput("load: skip_cols must be non-negative.");
    if (skip_rows < 0) throw dtwc::InvalidInput("load: skip_rows must be non-negative.");
    if (delimiter.size() > 1)
      throw dtwc::InvalidInput("load: delimiter must be a single character.");
    // File I/O and parsing touch no Python object, so the GIL is released for
    // the whole read exactly as every other I/O binding here does.
    nb::gil_scoped_release release;
    dtwc::DataLoader loader(source);
    loader.start_column(skip_cols).start_row(skip_rows).verbosity(0);
    if (!delimiter.empty()) loader.delimiter(delimiter[0]);
    try {
      return loader.load_local();
    } catch (const dtwc::Error &) {
      throw;
    } catch (const std::exception &e) {
      throw dtwc::IOError("load: failed to read '" + source.string() + "': " + e.what());
    }
  }, "source"_a, "skip_cols"_a = 0, "skip_rows"_a = 0, "delimiter"_a = std::string{},
     "Read a batch file or folder with the C++ DataLoader and return the owning\n"
     "dtwc::Data (series + names) with no intermediate Python objects. Backs\n"
     "dtwcpp.Dataset, whose handle is handed straight to Problem.set_data(Data),\n"
     "so Python and C++ parse a path with one implementation: skip_cols drops\n"
     "leading FIELDS before numeric parsing, skip_rows drops leading LINES, an\n"
     "empty delimiter means infer from the extension, and variable-length rows\n"
     "are preserved. The names are the loader's own -- file stem per file for a\n"
     "folder, 1-based row number for a batch file -- so Tier-1 output carries\n"
     "the same series names the CLI writes.");

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

  nb::enum_<dtwc::core::ConstraintType>(m, "ConstraintType")
    .value("NONE", dtwc::core::ConstraintType::None)
    .value("SakoeChibaBand", dtwc::core::ConstraintType::SakoeChibaBand);

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
    .value("Pruned", dtwc::DistanceMatrixStrategy::Pruned)
    .value("CUDA", dtwc::DistanceMatrixStrategy::CUDA)
    .value("Metal", dtwc::DistanceMatrixStrategy::Metal);

  nb::enum_<dtwc::core::StoragePolicy>(m, "StoragePolicy")
    .value("Auto", dtwc::core::StoragePolicy::Auto)
    .value("Heap", dtwc::core::StoragePolicy::Heap)
    .value("Mmap", dtwc::core::StoragePolicy::Mmap);

  nb::enum_<dtwc::LowerBoundStrategy>(m, "LowerBoundStrategy")
    .value("Auto", dtwc::LowerBoundStrategy::Auto)
    .value("None", dtwc::LowerBoundStrategy::None)
    .value("Kim", dtwc::LowerBoundStrategy::Kim)
    .value("Keogh", dtwc::LowerBoundStrategy::Keogh)
    .value("KimKeogh", dtwc::LowerBoundStrategy::KimKeogh)
    .value("Enhanced", dtwc::LowerBoundStrategy::Enhanced)
    .value("Webb", dtwc::LowerBoundStrategy::Webb);

  // =========================================================================
  // CUDASettings
  // =========================================================================

  nb::class_<dtwc::CUDASettings>(m, "CUDASettings")
    .def(nb::init<>())
    .def_rw("device_id", &dtwc::CUDASettings::device_id, "CUDA device index (default 0).")
    .def_prop_rw("precision",
      [](const dtwc::CUDASettings &s) { return s.precision; },
      [](dtwc::CUDASettings &s, int value) {
        dtwc::validate_cuda_settings_precision(value);
        s.precision = value;
      },
      "Compute precision: 0=Auto, 1=FP32, 2=FP64 (default 0).")
    .def("__repr__", [](const dtwc::CUDASettings &s) {
      return "CUDASettings(device_id=" + std::to_string(s.device_id)
             + ", precision=" + std::to_string(s.precision) + ")";
    });

  // =========================================================================
  // Linkage (hierarchical clustering)
  // =========================================================================

  nb::enum_<dtwc::algorithms::Linkage>(m, "Linkage")
    .value("Single", dtwc::algorithms::Linkage::Single)
    .value("Complete", dtwc::algorithms::Linkage::Complete)
    .value("Average", dtwc::algorithms::Linkage::Average);

  nb::enum_<dtwc::algorithms::OneBatchWeighting>(m, "OneBatchWeighting")
    .value("Uniform", dtwc::algorithms::OneBatchWeighting::Uniform)
    .value("Debiased", dtwc::algorithms::OneBatchWeighting::Debiased)
    .value("NearestNeighbor", dtwc::algorithms::OneBatchWeighting::NearestNeighbor);

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
    .def_rw("weighting", &dtwc::algorithms::OneBatchPAMOptions::weighting)
    .def_rw("relative_tolerance", &dtwc::algorithms::OneBatchPAMOptions::relative_tolerance);

  nb::class_<dtwc::algorithms::OneBatchPAMStats>(m, "OneBatchPAMStats")
    .def(nb::init<>())
    .def_ro("batch_size", &dtwc::algorithms::OneBatchPAMStats::batch_size)
    .def_ro("distance_evaluations", &dtwc::algorithms::OneBatchPAMStats::distance_evaluations)
    .def_ro("full_matrix_fraction", &dtwc::algorithms::OneBatchPAMStats::full_matrix_fraction)
    .def_ro("estimated_objective", &dtwc::algorithms::OneBatchPAMStats::estimated_objective)
    .def_ro("accepted_swaps", &dtwc::algorithms::OneBatchPAMStats::accepted_swaps);

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
    .def_ro("labels", &dtwc::algorithms::BarycenterClusteringResult::labels)
    .def_ro("barycenters", &dtwc::algorithms::BarycenterClusteringResult::barycenters)
    .def_ro("total_cost", &dtwc::algorithms::BarycenterClusteringResult::total_cost)
    .def_ro("iterations", &dtwc::algorithms::BarycenterClusteringResult::iterations)
    .def_ro("converged", &dtwc::algorithms::BarycenterClusteringResult::converged);

  // =========================================================================
  // DTWVariantParams
  // =========================================================================

  nb::class_<dtwc::core::DTWVariantParams>(m, "DTWVariantParams")
    .def(nb::init<>())
    .def_prop_rw("variant",
      [](const dtwc::core::DTWVariantParams &p) { return p.variant; },
      [](dtwc::core::DTWVariantParams &p, dtwc::core::DTWVariant value) {
        dtwc::core::validate_dtw_variant(value);
        p.variant = value;
      })
    .def_prop_rw("wdtw_g",
      [](const dtwc::core::DTWVariantParams &p) { return p.wdtw_g; },
      [](dtwc::core::DTWVariantParams &p, double value) {
        dtwc::core::validate_wdtw_g(value);
        p.wdtw_g = value;
      })
    .def_prop_rw("adtw_penalty",
      [](const dtwc::core::DTWVariantParams &p) { return p.adtw_penalty; },
      [](dtwc::core::DTWVariantParams &p, double value) {
        dtwc::core::validate_adtw_penalty(value);
        p.adtw_penalty = value;
      })
    .def_prop_rw("sdtw_gamma",
      [](const dtwc::core::DTWVariantParams &p) { return p.sdtw_gamma; },
      [](dtwc::core::DTWVariantParams &p, double value) {
        dtwc::core::validate_sdtw_gamma(value);
        p.sdtw_gamma = value;
      })
    .def_prop_rw("msm_c",
      [](const dtwc::core::DTWVariantParams &p) { return p.msm_c; },
      [](dtwc::core::DTWVariantParams &p, double value) {
        dtwc::core::validate_msm_c(value);
        p.msm_c = value;
      })
    .def_prop_rw("twe_nu",
      [](const dtwc::core::DTWVariantParams &p) { return p.twe_nu; },
      [](dtwc::core::DTWVariantParams &p, double value) {
        dtwc::core::validate_twe_nu(value);
        p.twe_nu = value;
      })
    .def_prop_rw("twe_lambda",
      [](const dtwc::core::DTWVariantParams &p) { return p.twe_lambda; },
      [](dtwc::core::DTWVariantParams &p, double value) {
        dtwc::core::validate_twe_lambda(value);
        p.twe_lambda = value;
      })
    .def_prop_rw("mv_mode",
      [](const dtwc::core::DTWVariantParams &p) { return p.mv_mode; },
      [](dtwc::core::DTWVariantParams &p, dtwc::core::MVMode value) {
        dtwc::core::validate_mv_mode(value);
        p.mv_mode = value;
      });

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
    .def_rw("max_benders_iter", &dtwc::MIPSettings::max_benders_iter,
            "Maximum Benders iterations (default 200).")
    .def_rw("benders", &dtwc::MIPSettings::benders,
            "Benders decomposition mode: 'auto' (N>200), 'on', or 'off'.")
    .def_rw("lr_max_nodes", &dtwc::MIPSettings::lr_max_nodes,
            "Method.LRCore branch-and-bound node cap (>= 1, default 2000000).")
    .def("__repr__", [](const dtwc::MIPSettings &s) {
      return "MIPSettings(gap=" + std::to_string(s.mip_gap)
             + ", time_limit=" + std::to_string(s.time_limit_sec)
             + ", warm_start=" + (s.warm_start ? "True" : "False")
             + ", numeric_focus=" + std::to_string(s.numeric_focus)
             + ", mip_focus=" + std::to_string(s.mip_focus)
             + ", benders=" + s.benders
             + ", max_benders_iter=" + std::to_string(s.max_benders_iter)
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
  // CLARANSOptions
  // =========================================================================

  nb::class_<dtwc::algorithms::CLARANSOptions>(m, "CLARANSOptions")
    .def(nb::init<>())
    .def_rw("n_clusters", &dtwc::algorithms::CLARANSOptions::n_clusters,
            "Number of clusters (k).")
    .def_rw("num_local", &dtwc::algorithms::CLARANSOptions::num_local,
            "Number of random restarts (default 2).")
    .def_rw("max_neighbor", &dtwc::algorithms::CLARANSOptions::max_neighbor,
            "Max non-improving swaps per restart (-1 = auto).")
    .def_rw("max_dtw_evals", &dtwc::algorithms::CLARANSOptions::max_dtw_evals,
            "Hard budget on total DTW computations (-1 = no limit).")
    .def_rw("random_seed", &dtwc::algorithms::CLARANSOptions::random_seed,
            "RNG seed for determinism (default 42).")
    .def("__repr__", [](const dtwc::algorithms::CLARANSOptions &o) {
      return "CLARANSOptions(k=" + std::to_string(o.n_clusters)
             + ", num_local=" + std::to_string(o.num_local)
             + ", max_neighbor=" + std::to_string(o.max_neighbor)
             + ", max_dtw_evals=" + std::to_string(o.max_dtw_evals)
             + ", seed=" + std::to_string(o.random_seed) + ")";
    });

  // =========================================================================
  // ClusteringResult
  // =========================================================================

  nb::class_<dtwc::core::ClusteringResult>(m, "ClusteringResult")
    .def(nb::init<>())
    .def_rw("labels", &dtwc::core::ClusteringResult::labels)
    .def_rw("medoid_indices", &dtwc::core::ClusteringResult::medoid_indices)
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
  // DenseDistanceMatrix
  // =========================================================================

  nb::class_<dtwc::core::DenseDistanceMatrix>(m, "DenseDistanceMatrix")
    .def(nb::init<>())
    .def(nb::init<size_t>(), "n"_a)
    .def("resize", &dtwc::core::DenseDistanceMatrix::resize, "n"_a)
    .def("get", &dtwc::core::DenseDistanceMatrix::get, "i"_a, "j"_a)
    .def("set", &dtwc::core::DenseDistanceMatrix::set, "i"_a, "j"_a, "value"_a)
    .def("is_computed", &dtwc::core::DenseDistanceMatrix::is_computed, "i"_a, "j"_a)
    .def_prop_ro("size", &dtwc::core::DenseDistanceMatrix::size)
    .def("max", &dtwc::core::DenseDistanceMatrix::max)
    .def("to_numpy", [](const dtwc::core::DenseDistanceMatrix &dm) {
      // Expand packed triangular storage to a full N*N numpy array.
      // This is always a COPY — the C++ matrix stores only the upper triangle
      // (n*(n+1)/2 entries) so a true zero-copy view into a full N*N layout
      // is structurally impossible. Modifying the returned array does NOT
      // mutate the C++ matrix; use set(i, j, v) for that.
      // G4: the packed->dense expansion and the memcpy are O(N^2) and would
      // block every other Python thread. `dm` is const here, so releasing is safe.
      const size_t n = dm.size();
      std::vector<double> values(n * n);
      {
        nb::gil_scoped_release release;
        const Eigen::MatrixXd full = dtwc::io::to_full_matrix(dm);
        // Eigen is column-major; numpy expects row-major. The matrix is
        // symmetric, so the byte layout is identical and memcpy is correct.
        if (n > 0)
          std::memcpy(values.data(), full.data(), n * n * sizeof(double));
      }
      return adopt_as_ndarray(std::move(values), {n, n});
    }, "Return an independent copy of the full N*N distance matrix.\n\n"
       "The C++ matrix stores only the upper triangle, so this expands to a\n"
       "full symmetric N*N numpy array. Modifying the returned array does NOT\n"
       "affect the C++ matrix — use set(i, j, v) for that.")
    .def("write_csv", [](const dtwc::core::DenseDistanceMatrix &dm,
                          const std::filesystem::path &path) {
      dtwc::io::write_csv(dm, path);
    }, "path"_a)
    .def("read_csv", [](dtwc::core::DenseDistanceMatrix &dm,
                         const std::filesystem::path &path) {
      dtwc::io::read_csv(dm, path);
    }, "path"_a)
    .def("__repr__", [](const dtwc::core::DenseDistanceMatrix &dm) {
      return "DenseDistanceMatrix(n=" + std::to_string(dm.size()) + ")";
    });

  // =========================================================================
  // DTW distance functions
  // =========================================================================

  m.def("dtw_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                            nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                            int band, const std::string &metric) {
    const auto mt = dtwc::core::parse_metric_token(metric);
    nb::gil_scoped_release release;
    return dtwc::dtwBanded<double>(x.data(), x.size(), y.data(), y.size(), band, -1.0, mt);
  }, "x"_a, "y"_a, "band"_a = -1, "metric"_a = "l1",
     "Compute DTW distance (zero-copy from numpy).\n\n"
     "metric: 'l1' (default) or 'squared_euclidean'.\n"
     "band=-1 for full DTW, band>0 for Sakoe-Chiba banded DTW.");

  // Arg-type parity (api-contract-2.0.md §2.6): every distance fn takes a
  // zero-copy c-contiguous float64 ndarray (previously ddtw/wdtw/adtw/soft took
  // std::vector, forcing a copy). The C++ span/pointer overloads make this exact.
  m.def("ddtw_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                              nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                              int band) {
    nb::gil_scoped_release release;
    return dtwc::ddtwBanded<double>(x.data(), x.size(), y.data(), y.size(), band);
  }, "x"_a, "y"_a, "band"_a = -1,
     "Compute Derivative DTW distance (zero-copy from numpy).");

  m.def("wdtw_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                              nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                              int band, double g) {
    nb::gil_scoped_release release;
    return dtwc::wdtwBanded<double>(std::span<const double>(x.data(), x.size()),
                                    std::span<const double>(y.data(), y.size()), band, g);
  }, "x"_a, "y"_a, "band"_a = -1, "g"_a = 0.05,
     "Compute Weighted DTW distance with logistic weight steepness g (zero-copy).");

  m.def("adtw_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                              nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                              int band, double penalty) {
    nb::gil_scoped_release release;
    return dtwc::adtwBanded<double>(std::span<const double>(x.data(), x.size()),
                                    std::span<const double>(y.data(), y.size()), band, penalty);
  }, "x"_a, "y"_a, "band"_a = -1, "penalty"_a = 1.0,
     "Compute Amerced DTW distance with non-diagonal step penalty (zero-copy).");

  m.def("soft_dtw_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                                 nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                                 double gamma) {
    nb::gil_scoped_release release;
    return dtwc::soft_dtw<double>(std::span<const double>(x.data(), x.size()),
                                  std::span<const double>(y.data(), y.size()), gamma);
  }, "x"_a, "y"_a, "gamma"_a = 1.0,
     "Compute Soft-DTW distance (differentiable, zero-copy from numpy).");

  m.def("soft_dtw_gradient", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                                 nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                                 double gamma) {
    nb::gil_scoped_release release;
    return dtwc::soft_dtw_gradient<double>(std::span<const double>(x.data(), x.size()),
                                           std::span<const double>(y.data(), y.size()), gamma);
  }, "x"_a, "y"_a, "gamma"_a = 1.0,
     "Compute Soft-DTW gradient w.r.t. first series x (zero-copy from numpy).");

  m.def("dtw_distance_missing", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                                    nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                                    int band, const std::string &metric) {
    const auto mt = dtwc::core::parse_metric_token(metric);
    nb::gil_scoped_release release;
    return dtwc::dtwMissing_banded<double>(x.data(), x.size(), y.data(), y.size(), band, -1.0, mt);
  }, "x"_a, "y"_a, "band"_a = -1, "metric"_a = "l1",
     "DTW distance with missing data support (NaN = missing).\n\n"
     "NaN values in either series are treated as missing; pairs where\n"
     "one or both values are NaN contribute zero cost.\n"
     "metric: 'l1' (default) or 'squared_euclidean'.\n"
     "band=-1 for full DTW, band>0 for Sakoe-Chiba banded DTW.");

  m.def("dtw_arow_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                                  nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                                  int band, const std::string &metric) {
    const auto mt = dtwc::core::parse_metric_token(metric);
    nb::gil_scoped_release release;
    if (band >= 0)
      return dtwc::dtwAROW_banded<double>(x.data(), x.size(), y.data(), y.size(), band, mt);
    else
      return dtwc::dtwAROW_L<double>(x.data(), x.size(), y.data(), y.size(), mt);
  }, "x"_a, "y"_a, "band"_a = -1, "metric"_a = "l1",
     "DTW-AROW distance with diagonal-only alignment for missing values.\n\n"
     "When x[i] or y[j] is NaN, the warping path is restricted to the\n"
     "diagonal direction only (one-to-one alignment), preventing free\n"
     "stretching through missing regions.\n"
     "Reference: Yurtman et al. (ECML-PKDD 2023).\n"
     "metric: 'l1' (default) or 'squared_euclidean'.\n"
     "band=-1 for full DTW-AROW, band>0 for Sakoe-Chiba banded DTW-AROW.");

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
         "Raises RuntimeError if any series has incompatible size.");

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

  // Shared read accessor: fill (if needed) and expand packed triangular → NxN.
  // Named `distance_matrix()` in 2.0 (was `distance_matrix_numpy()`); always a
  // COPY because the C++ store keeps only the upper triangle, so a zero-copy view
  // into a full NxN layout is structurally impossible (§2.2 ‡).
  auto read_distance_matrix_np = [](dtwc::Problem &prob) {
    // Size is only known after the fill, so both happen inside one release.
    std::vector<double> values;
    size_t n = 0;
    {
      nb::gil_scoped_release release;
      prob.fill_distance_matrix();
      const auto &dm = prob.dense_distance_matrix();
      n = dm.size();
      const Eigen::MatrixXd full = dtwc::io::to_full_matrix(dm);
      values.resize(n * n);
      // Symmetric matrix: col-major == row-major, safe to memcpy.
      if (n > 0)
        std::memcpy(values.data(), full.data(), n * n * sizeof(double));
    }
    return adopt_as_ndarray(std::move(values), {n, n});
  };
  // Shared writer: load a precomputed NxN matrix (e.g. from a GPU compute).
  auto write_distance_matrix_np =
    [](dtwc::Problem &p, nb::ndarray<const double, nb::ndim<2>, nb::c_contig> dm) {
      const size_t n = dm.shape(0);
      if (dm.shape(1) != n)
        throw dtwc::InvalidInput("Expected square distance matrix");
      if (n != p.size())
        throw dtwc::InvalidInput("Matrix size doesn't match Problem data size");
      auto &mat = p.dense_distance_matrix();
      mat.resize(n);
      const double *data = dm.data();
      for (size_t i = 0; i < n; ++i)
        for (size_t j = i; j < n; ++j)
          mat.set(i, j, data[i * n + j]);
    };

  nb::class_<dtwc::Problem>(m, "Problem",
    "A clustering problem: data, configuration, distance matrix and results.\n\n"
    "Threading: a Problem instance must not be used concurrently from multiple\n"
    "Python threads; the GIL is released during C++ work so that other threads\n"
    "can run, but two threads calling methods on the same Problem race on its\n"
    "lazily-filled distance cache. Use one Problem per thread, or call\n"
    "fill_distance_matrix() first and only read afterwards.")
    .def(nb::init<>())
    .def("__init__", [](dtwc::Problem *p, const std::string &name) {
      new (p) dtwc::Problem(name);
    }, "name"_a)
    // ---- config properties (canonical names) ----
    .def_prop_rw("method", &dtwc::Problem::method, &dtwc::Problem::set_method)
    .def_prop_rw("max_iter", &dtwc::Problem::max_iter,
                 &dtwc::Problem::set_max_iter)
    .def_prop_rw("n_repetitions", &dtwc::Problem::n_repetitions,
                 &dtwc::Problem::set_n_repetitions,
                 "Repetitions for iterative methods.")
    .def_prop_rw("n_repetition",
                 [](const dtwc::Problem &p) {
                   warn_deprecated_alias("Problem.n_repetition",
                                         "Problem.n_repetitions");
                   return p.n_repetitions();
                 },
                 [](dtwc::Problem &p, int value) {
                   warn_deprecated_alias("Problem.n_repetition",
                                         "Problem.n_repetitions");
                   p.set_n_repetitions(value);
                 },
                 "Deprecated alias for n_repetitions (kept one cycle, §4).")
    .def_prop_rw("random_seed", &dtwc::Problem::random_seed,
                 &dtwc::Problem::set_random_seed,
                 "Invocation-local seed for Lloyd and MIP warm starts.")
    .def_prop_rw("band",
                 [](const dtwc::Problem &p) { return p.band; },
                 [](dtwc::Problem &p, int value) { p.set_band(value); })
    .def_prop_rw("variant_params",
                 [](const dtwc::Problem &p) -> const dtwc::core::DTWVariantParams & {
                   return p.variant_params;
                 },
                 [](dtwc::Problem &p, dtwc::core::DTWVariantParams value) {
                   p.set_variant(value);
                 })
    .def_prop_rw("missing_strategy",
                 [](const dtwc::Problem &p) { return p.missing_strategy; },
                 [](dtwc::Problem &p, dtwc::core::MissingStrategy value) {
                   p.set_missing_strategy(value);
                 },
                 "Strategy for handling NaN values (Error, ZeroCost, AROW, Interpolate).")
    .def_prop_rw("distance_strategy",
                 [](const dtwc::Problem &p) { return p.distance_strategy; },
                 [](dtwc::Problem &p, dtwc::DistanceMatrixStrategy value) {
                   p.set_distance_strategy(value);
                 },
                 "Distance matrix computation strategy (Auto, BruteForce, Pruned, CUDA, Metal).")
    .def_prop_rw("lb_strategy", &dtwc::Problem::lb_strategy,
                 &dtwc::Problem::set_lb_strategy,
                 "Lower-bound selection for the Pruned CPU path "
                 "(Auto/None/Kim/Keogh/KimKeogh/Enhanced/Webb).")
    .def_prop_rw("storage_policy", &dtwc::Problem::storage_policy,
                 &dtwc::Problem::set_storage_policy,
                 "How the next owning set_data call stores series "
                 "(Auto/Heap/Mmap); existing data is unchanged.")
    .def_prop_rw("cuda_settings",
                 [](const dtwc::Problem &p) -> const dtwc::CUDASettings & {
                   return p.cuda_settings;
                 },
                 [](dtwc::Problem &p, dtwc::CUDASettings value) {
                   p.set_cuda_settings(value);
                 },
                 "GPU compute options (used when distance_strategy == CUDA).")
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
    .def_rw("clusters_ind", &dtwc::Problem::clusters_ind)
    .def_rw("centroids_ind", &dtwc::Problem::centroids_ind)
    // ---- read accessors ----
    .def_prop_ro("size", &dtwc::Problem::size)
    .def("n_clusters", &dtwc::Problem::n_clusters, "Number of clusters (was cluster_size()).")
    .def_prop_ro("cluster_size", [](const dtwc::Problem &p) {
                   warn_deprecated_alias("Problem.cluster_size",
                                         "Problem.n_clusters");
                   return p.n_clusters();
                 },
                 "Deprecated alias for n_clusters() (kept one cycle, §4).")
    .def("labels", &dtwc::Problem::labels,
         "Cluster label of each series (reads clusters_ind; parity with Result.labels).")
    .def("medoids", &dtwc::Problem::medoids,
         "Medoid series indices (reads centroids_ind; parity with Result.medoids).")
    .def("series", [](const dtwc::Problem &p, size_t i) {
      auto s = p.series(i);
      return std::vector<double>(s.begin(), s.end());
    }, "i"_a, "Copy of series i as a list of doubles.")
    .def("series_name", [](const dtwc::Problem &p, size_t i) {
      return std::string(p.series_name(i));
    }, "i"_a, "Name of series i.")
    .def("centroid_of", &dtwc::Problem::centroid_of, "i"_a,
         "Medoid index of the cluster that series i belongs to.")
    .def("is_distance_matrix_filled", &dtwc::Problem::is_distance_matrix_filled)
    .def("max_distance", &dtwc::Problem::max_distance)
    .def("dist_by_ind", [](dtwc::Problem &p, int i, int j) {
      nb::gil_scoped_release release;
      return p.dist_by_ind(i, j);
    }, "i"_a, "j"_a,
       "Distance between series i and j, computing it on demand.\n\n"
       "The lazy compute path MUTATES this Problem, so it must not be called\n"
       "concurrently from several Python threads on the same object (see the\n"
       "Problem class docstring).")
    // ---- config setters ----
    .def("set_n_clusters", &dtwc::Problem::set_n_clusters, "n_clusters"_a)
    .def("set_number_of_clusters", [](dtwc::Problem &p, int n) {
           warn_deprecated_alias("Problem.set_number_of_clusters",
                                 "Problem.set_n_clusters");
           p.set_n_clusters(n);
         },
         "n_clusters"_a, "Deprecated alias for set_n_clusters (kept one cycle, §4).")
    .def("set_method", &dtwc::Problem::set_method, "method"_a)
    .def("set_band", &dtwc::Problem::set_band, "band"_a)
    .def("set_max_iter", &dtwc::Problem::set_max_iter, "max_iter"_a)
    .def("set_n_repetitions", &dtwc::Problem::set_n_repetitions, "n_repetitions"_a)
    .def("set_random_seed", &dtwc::Problem::set_random_seed, "random_seed"_a)
    .def("set_variant", nb::overload_cast<dtwc::core::DTWVariant>(&dtwc::Problem::set_variant), "variant"_a)
    .def("set_variant_params",
         nb::overload_cast<dtwc::core::DTWVariantParams>(&dtwc::Problem::set_variant), "params"_a,
         "Set the DTW variant + parameters and rebind the distance function.")
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
    .def("set_view_data", [](dtwc::Problem &p, std::vector<std::vector<double>> series,
                              std::vector<std::string> names, size_t ndim) {
      dtwc::Data d(std::move(series), std::move(names), ndim);
      p.set_view_data(std::move(d));
    }, "series"_a, "names"_a, "ndim"_a = 1,
       "Set data via the light/view path (sizes the distance matrix, skips the mmap\n"
       "cache). Python builds an owning Data (safe lifetime); the zero-copy span\n"
       "mode is a C++/CLARA-internal optimisation.")
    // ---- distance matrix ----
    .def("fill_distance_matrix", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.fill_distance_matrix();
    }, "Compute all pairwise DTW distances.")
    .def("distance_matrix", read_distance_matrix_np,
         "Fill (if needed) and return the full NxN distance matrix as a numpy\n"
         "array (independent copy; use set_distance_matrix to write).")
    .def("distance_matrix_numpy", [read_distance_matrix_np](dtwc::Problem &p) {
           warn_deprecated_alias("Problem.distance_matrix_numpy",
                                 "Problem.distance_matrix");
           return read_distance_matrix_np(p);
         },
         "Deprecated alias for distance_matrix() (kept one cycle, §4).")
    .def("set_distance_matrix", write_distance_matrix_np, "dm"_a,
         "Load a precomputed NxN distance matrix (e.g. from a GPU compute).")
    .def("set_distance_matrix_from_numpy",
         [write_distance_matrix_np](
           dtwc::Problem &p,
           nb::ndarray<const double, nb::ndim<2>, nb::c_contig> dm) {
           warn_deprecated_alias("Problem.set_distance_matrix_from_numpy",
                                 "Problem.set_distance_matrix");
           write_distance_matrix_np(p, dm);
         }, "dm"_a,
         "Deprecated alias for set_distance_matrix() (kept one cycle, §4).")
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
    }, "Total cost of the current cluster assignment.")
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
    }, "Write the cluster-assignment CSV.")
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
                                        int band, const std::string &metric,
                                        bool use_pruning) {
    const auto mt = dtwc::core::parse_metric_token(metric);

    // Task 3.6 (review H1): this high-level Python compute path never constructs
    // dtwc::env(), so its OpenMP warning would otherwise be silent under
    // OMP_NUM_THREADS=1. Warn once, deterministically, before either branch (the
    // pruned branch also warns via get_max_threads; this covers the unpruned one).
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

      if (use_pruning && (mt == dtwc::core::MetricType::L1 || mt == dtwc::core::MetricType::L2)) {
        // Legacy LB-guided exact-matrix route. LB_Kim, and LB_Keogh only for
        // band >= 0, can select a cutoff attempt. A cutoff result is recomputed
        // without early abandon because every matrix entry is required.
        dtwc::core::compute_distance_matrix_pruned(series, ptr, band, mt);
      } else {
        // Standard unpruned version (for non-L1 metrics or when pruning disabled).
        // Lock-free by design: each thread owns a disjoint set of rows (outer loop i).
        // Writes to ptr[i*n+j] and ptr[j*n+i] never collide across threads because
        // no two threads share the same i value.
        // num_threads pins the team to the number of slots sized above, so
        // omp_get_thread_num() can never index past `errors`.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 16) num_threads(n_error_slots)
        #endif
        for (int i = 0; i < static_cast<int>(n); ++i) {
#ifdef _OPENMP
            const size_t slot = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t slot = 0;
#endif
            // dtwBanded/dtwFull_L throw on NaN input. An exception escaping an
            // OpenMP region is undefined behaviour and terminates the process,
            // i.e. a hard interpreter crash instead of a Python InvalidInput.
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
      }
    }  // GIL re-acquired here

    for (const auto &error : errors)
      if (error) std::rethrow_exception(error);

    return adopt_as_ndarray(std::move(values), {n, n});
  }, "series"_a, "band"_a = -1, "metric"_a = "l1", "use_pruning"_a = true,
     "Compute pairwise DTW distance matrix entirely in C++.\n\n"
     "Returns NxN numpy array. Uses OpenMP parallelism when available.\n"
     "For L1 (and the core's equivalent scalar L2), use_pruning=True\n"
     "selects the legacy LB-guided exact-matrix path.\n"
     "band=-1 disables LB_Keogh. Every early-abandoned pair is\n"
     "recomputed without a cutoff; this option is not a speed guarantee.\n"
     "Squared Euclidean uses the direct exact path.\n"
     "This avoids a Python-level pair loop.");

  // =========================================================================
  // FastPAM
  // =========================================================================

  m.def("fast_pam", [](dtwc::Problem &prob, int n_clusters, int max_iter) {
    nb::gil_scoped_release release;
    return dtwc::fast_pam(prob, n_clusters, max_iter);
  }, "prob"_a, "n_clusters"_a, "max_iter"_a = 100,
     "Run FastPAM k-medoids clustering (Schubert & Rousseeuw 2021).\n\n"
     "The C++ core writes labels/medoids/k back into prob (since 1.6), so\n"
     "silhouette(prob) and davies_bouldin(prob) work after this call with no\n"
     "wrapper-side wiring (api-contract-2.0.md §2.5).");

  m.def("fast_pam_seeded",
        [](dtwc::Problem &prob, int n_clusters, std::uint64_t seed, int max_iter) {
    nb::gil_scoped_release release;
    return dtwc::fast_pam_seeded(prob, n_clusters, seed, max_iter);
  }, "prob"_a, "n_clusters"_a, "seed"_a, "max_iter"_a = 100,
     "Run FastPAM with an invocation-local deterministic BUILD seed.");

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

  m.def("fast_clara", [](dtwc::Problem &prob, int n_clusters, int sample_size,
                           int n_samples, int max_iter, unsigned seed) {
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

  m.def("one_batch_pam", [](dtwc::Problem &prob, int n_clusters, int batch_size,
                              int max_iter, std::uint64_t seed,
                              dtwc::algorithms::OneBatchWeighting weighting) {
    dtwc::algorithms::OneBatchPAMOptions options;
    options.n_clusters = n_clusters;
    options.batch_size = batch_size;
    options.max_iter = max_iter;
    options.random_seed = seed;
    options.weighting = weighting;
    nb::gil_scoped_release release;
    return dtwc::algorithms::one_batch_pam(prob, options);
  }, "prob"_a, "n_clusters"_a, "batch_size"_a = -1, "max_iter"_a = 100,
     "seed"_a = dtwc::settings::DEFAULT_RANDOM_SEED,
     "weighting"_a = dtwc::algorithms::OneBatchWeighting::NearestNeighbor,
     "Run OneBatchPAM using one fixed N-by-m distance table (AAAI 2025).");

  m.def("one_batch_pam_with_stats",
        [](dtwc::Problem &prob, const dtwc::algorithms::OneBatchPAMOptions &options) {
    dtwc::algorithms::OneBatchPAMStats stats;
    dtwc::core::ClusteringResult result;
    {
      nb::gil_scoped_release release;
      result = dtwc::algorithms::one_batch_pam(prob, options, &stats);
    }
    return nb::make_tuple(std::move(result), std::move(stats));
  }, "prob"_a, "options"_a,
     "Run OneBatchPAM and return (ClusteringResult, OneBatchPAMStats).");

  // =========================================================================
  // DTW barycenters
  // =========================================================================

  m.def("dtw_barycenter",
        [](const dtwc::Problem &prob, const std::vector<int> &indices,
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
            "runs consecutive row blocks of this size and publishes one\n"
            "generation after each block, the last included, so a completed\n"
            "fill leaves ceil(N / save_interval) generations. A value below 1\n"
            "with enabled raises InvalidInput.")
    .def_rw("enabled", &dtwc::CheckpointOptions::enabled,
            "Enable automatic mid-fill checkpointing. Requires dense distance\n"
            "storage (mmap storage raises InvalidInput) and a non-empty\n"
            "directory; both are checked before any distance is computed.\n"
            "DistanceMatrixStrategy.Pruned is downgraded to BruteForce.")
    .def("__repr__", [](const dtwc::CheckpointOptions &o) {
      return "CheckpointOptions(dir='" + o.directory
             + "', interval=" + std::to_string(o.save_interval)
             + ", enabled=" + (o.enabled ? "True" : "False") + ")";
    });

  m.def("save_checkpoint", [](const dtwc::Problem &prob,
                              const std::string &path,
                              dtwc::core::MetricType metric) {
        // N^2 CSV write; released for consistency with save_binary_checkpoint.
        // `prob` is const here and the writer only reads it.
        nb::gil_scoped_release release;
        dtwc::save_checkpoint(prob, path, metric);
      }, "prob"_a, "path"_a, "metric"_a = dtwc::core::MetricType::L1,
        "Save distance matrix checkpoint to directory.\n\n"
        "Creates distances.csv and metadata.txt in the given directory.\n"
        "The directory is created if it does not exist.\n\n"
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
        "Load distance matrix checkpoint from directory.\n\n"
        "Returns True if checkpoint was loaded successfully, False otherwise.\n"
        "Validates that matrix dimensions match the Problem's data size.\n"
        "Sets distance matrix filled flag if all pairs are computed.\n\n"
        "`metric` is the pointwise metric THIS run computes with: a checkpoint\n"
        "written under a different metric no longer matches the identity\n"
        "fingerprint and is rejected. Mirrors the CLI's --metric; defaults to\n"
        "L1 for backward compatibility.\n\n"
        "MUTATES `prob`: do not run it concurrently with any other method on\n"
        "the same Problem (see the Problem class docstring).");

  m.def("save_binary_checkpoint",
        [](const dtwc::core::ClusteringResult &result,
           const std::filesystem::path &path) {
    // A Python thread may mutate the bound result after the GIL is released.
    // Snapshot it first so the native writer always observes one coherent value.
    const dtwc::core::ClusteringResult snapshot = result;
    nb::gil_scoped_release release;
    dtwc::save_binary_checkpoint(snapshot, path);
  }, "result"_a, "path"_a,
     "Save a ClusteringResult to a binary version-1 checkpoint.");

  m.def("load_binary_checkpoint", [](const std::filesystem::path &path) {
    // Prepare all Python-facing text while the GIL is held. The release scope
    // contains only native state and filesystem work.
    const std::string path_text = utf8_path_text(path);
    dtwc::core::ClusteringResult result;
    bool loaded = false;
    {
      nb::gil_scoped_release release;
      loaded = dtwc::load_binary_checkpoint(result, path);
    }
    if (!loaded) {
      throw dtwc::IOError(
        "load_binary_checkpoint: cannot read a valid binary result "
        "checkpoint from '" + path_text + "'.");
    }
    return result;
  }, "path"_a,
     "Load a ClusteringResult from a binary version-1 checkpoint.\n\n"
     "Raises IOError if the checkpoint is absent, inaccessible, or invalid.");

  // =========================================================================
  // Scores
  // =========================================================================

  // Canonical 2.0 score names (api-contract-2.0.md §2.4): the `Index`/`Information`
  // noun is dropped. The old *_index / *_information spellings stay one cycle as
  // deprecated aliases (§4) — both forward to the same canonical C++ function.
  m.def("silhouette", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::silhouette(prob);
  }, "prob"_a, "Compute silhouette score for each data point.");

  m.def("davies_bouldin", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::davies_bouldin(prob);
  }, "prob"_a, "Compute Davies-Bouldin index (lower is better).");
  m.def("davies_bouldin_index", [](dtwc::Problem &prob) {
    warn_deprecated_alias("dtwcpp.davies_bouldin_index",
                          "dtwcpp.davies_bouldin");
    nb::gil_scoped_release release;
    return dtwc::scores::davies_bouldin(prob);
  }, "prob"_a, "Deprecated alias for davies_bouldin() (kept one cycle, §4).");

  m.def("dunn", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::dunn(prob);
  }, "prob"_a,
     "Compute Dunn index (min inter-cluster distance / max intra-cluster diameter).");
  m.def("dunn_index", [](dtwc::Problem &prob) {
    warn_deprecated_alias("dtwcpp.dunn_index", "dtwcpp.dunn");
    nb::gil_scoped_release release;
    return dtwc::scores::dunn(prob);
  }, "prob"_a, "Deprecated alias for dunn() (kept one cycle, §4).");

  m.def("inertia", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::inertia(prob);
  }, "prob"_a,
     "Compute inertia (total within-cluster distance sum to medoids).");

  m.def("calinski_harabasz", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::calinski_harabasz(prob);
  }, "prob"_a, "Compute Calinski-Harabasz index (medoid-adapted; higher is better).");
  m.def("calinski_harabasz_index", [](dtwc::Problem &prob) {
    warn_deprecated_alias("dtwcpp.calinski_harabasz_index",
                          "dtwcpp.calinski_harabasz");
    nb::gil_scoped_release release;
    return dtwc::scores::calinski_harabasz(prob);
  }, "prob"_a, "Deprecated alias for calinski_harabasz() (kept one cycle, §4).");

  m.def("adjusted_rand", [](const std::vector<int> &labels_true,
                            const std::vector<int> &labels_pred) {
    return dtwc::scores::adjusted_rand(labels_true, labels_pred);
  }, "labels_true"_a, "labels_pred"_a,
     "Adjusted Rand index between two label assignments (1.0 = perfect agreement).");
  m.def("adjusted_rand_index", [](const std::vector<int> &labels_true,
                                    const std::vector<int> &labels_pred) {
    warn_deprecated_alias("dtwcpp.adjusted_rand_index",
                          "dtwcpp.adjusted_rand");
    return dtwc::scores::adjusted_rand(labels_true, labels_pred);
  }, "labels_true"_a, "labels_pred"_a,
     "Deprecated alias for adjusted_rand() (kept one cycle, §4).");

  m.def("normalized_mutual_info", [](const std::vector<int> &labels_true,
                                      const std::vector<int> &labels_pred) {
    return dtwc::scores::normalized_mutual_info(labels_true, labels_pred);
  }, "labels_true"_a, "labels_pred"_a,
     "Normalized Mutual Information between two label assignments ([0,1]).");
  m.def("normalized_mutual_information", [](const std::vector<int> &labels_true,
                                              const std::vector<int> &labels_pred) {
    warn_deprecated_alias("dtwcpp.normalized_mutual_information",
                          "dtwcpp.normalized_mutual_info");
    return dtwc::scores::normalized_mutual_info(labels_true, labels_pred);
  }, "labels_true"_a, "labels_pred"_a,
     "Deprecated alias for normalized_mutual_info() (kept one cycle, §4).");

  // =========================================================================
  // Hierarchical clustering
  // =========================================================================

  m.def("build_dendrogram", [](dtwc::Problem &prob,
                                 const dtwc::algorithms::HierarchicalOptions &opts) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::build_dendrogram(prob, opts);
  }, "prob"_a, "opts"_a = dtwc::algorithms::HierarchicalOptions{},
     "Build a hierarchical dendrogram from a Problem.\n\n"
     "Requires distance matrix to be filled (call fill_distance_matrix() first).\n"
     "Returns a Dendrogram containing N-1 merge steps in merge order.\n"
     "Throws RuntimeError if N > opts.max_points (default 2000).");

  m.def("cut_dendrogram", [](const dtwc::algorithms::Dendrogram &dend,
                               dtwc::Problem &prob, int k) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::cut_dendrogram(dend, prob, k);
  }, "dendrogram"_a, "prob"_a, "k"_a,
     "Cut a dendrogram to produce k flat clusters.\n\n"
     "Returns a ClusteringResult with labels, medoid_indices, and total_cost.\n"
     "The C++ core also writes labels/medoids/k back into prob (since 1.6), so\n"
     "silhouette(prob) etc. work after this call with no wrapper wiring (§2.5).");

  // =========================================================================
  // CLARANS
  // =========================================================================

  m.def("clarans", [](dtwc::Problem &prob, const dtwc::algorithms::CLARANSOptions &opts) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::clarans(prob, opts);
  }, "prob"_a, "opts"_a,
     "Run CLARANS randomized k-medoids clustering.\n\n"
     "Experimental bounded mid-ground algorithm. Tests random\n"
     "(medoid_out, x_in) swaps, accepting only strictly improving ones.\n"
     "The C++ core writes labels/medoids/k back into prob (since 1.6) so\n"
     "scoring functions work after this call (§2.5).\n\n"
     "Reference: Ng & Han (2002), IEEE TKDE 14(5).");


  // =========================================================================
  // PDLP LP-relaxation arbiter (E2: was C++-only)
  // =========================================================================

  m.def("pdlp_gpu_available", &dtwc::mip::pdlp_gpu_available,
        "True if this build's HiGHS carries the cuPDLP GPU backend.\n\n"
        "The GPU-LP counterpart of HIGHS_AVAILABLE: the device is a property\n"
        "of the linked HiGHS build, not a per-call toggle.");

  nb::class_<dtwc::mip::PdlpParams>(m, "PdlpParams")
    .def(nb::init<>())
    .def_rw("variant", &dtwc::mip::PdlpParams::variant)
    .def_rw("tol", &dtwc::mip::PdlpParams::tol)
    .def_rw("iteration_limit", &dtwc::mip::PdlpParams::iteration_limit)
    .def_rw("use_gpu", &dtwc::mip::PdlpParams::use_gpu)
    .def_rw("verbose", &dtwc::mip::PdlpParams::verbose);

  nb::class_<dtwc::mip::PdlpResult>(m, "PdlpResult")
    .def_ro("lp_bound", &dtwc::mip::PdlpResult::lp_bound)
    .def_ro("solved", &dtwc::mip::PdlpResult::solved)
    .def_ro("iterations", &dtwc::mip::PdlpResult::iterations)
    .def_ro("gpu_used", &dtwc::mip::PdlpResult::gpu_used)
    .def("__repr__", [](const dtwc::mip::PdlpResult &r) {
      return "PdlpResult(lp_bound=" + std::to_string(r.lp_bound)
             + ", solved=" + (r.solved ? "True" : "False")
             + ", iterations=" + std::to_string(r.iterations)
             + ", gpu_used=" + (r.gpu_used ? "True" : "False") + ")";
    });

  m.def("pdlp_lp_bound",
        [](nb::ndarray<const double, nb::ndim<2>, nb::c_contig> D, int k,
           const dtwc::mip::PdlpParams &params) {
          if (D.shape(0) != D.shape(1))
            throw dtwc::InvalidInput("pdlp_lp_bound: D must be square.");
          if (D.shape(0) > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw dtwc::InvalidInput("pdlp_lp_bound: N exceeds INT_MAX.");
          const int n = static_cast<int>(D.shape(0));
          const double *data = D.data();
          nb::gil_scoped_release release;
          return dtwc::mip::pdlp_lp_bound(data, n, k, params);
        },
        "D"_a, "k"_a, "params"_a = dtwc::mip::PdlpParams{},
        "Solve the p-median LP relaxation with HiGHS PDLP.\n\n"
        "Returns the LP-relaxation optimum in RAW distance units - a valid\n"
        "lower bound on the integer k-medoids cost, NOT a clustering. It is\n"
        "the independent arbiter for the matrix-free LR-core bound.\n"
        "Requires a HiGHS build; raises SolverError otherwise.");

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
           int band, bool use_squared_l2, int device_id, bool verbose,
           bool use_lb_keogh, double lb_threshold) {
          dtwc::cuda::CUDADistMatOptions opts;
          opts.band = band;
          opts.use_squared_l2 = use_squared_l2;
          opts.device_id = device_id;
          opts.verbose = verbose;
          opts.use_lb_keogh = use_lb_keogh;
          opts.lb_threshold = lb_threshold;
          std::vector<double> matrix;
          size_t n = 0;
          {
            nb::gil_scoped_release release;
            auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
            n = result.n;
            matrix = checked_square_matrix(std::move(result.matrix), n, "CUDA");
          }
          return adopt_as_ndarray(std::move(matrix), {n, n});
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "device_id"_a = 0, "verbose"_a = false,
        "use_lb_keogh"_a = false, "lb_threshold"_a = -1.0,
        "Compute NxN DTW distance matrix on CUDA GPU.\n\n"
        "Returns NxN numpy array of DTW distances.\n"
        "When `use_lb_keogh=True`, `band >= 0`, and `lb_threshold > 0`, pairs\n"
        "whose LB_Keogh lower bound exceeds `lb_threshold` are pruned (finite\n"
        "double-max sentinel in result, not IEEE infinity).");

  m.def("compute_lb_keogh_cuda",
        [](const std::vector<std::vector<double>> &series,
           int band, int device_id) {
          std::vector<double> lb_values;
          {
            nb::gil_scoped_release release;
            auto result = dtwc::cuda::compute_lb_keogh_cuda(series, band, device_id);
            lb_values = std::move(result.lb_values);
          }
          const size_t np = lb_values.size();
          return adopt_as_ndarray(std::move(lb_values), {np});
        },
        "series"_a, "band"_a, "device_id"_a = 0,
        "Compute LB_Keogh lower bounds for all N*(N-1)/2 pairs on GPU.\n\n"
        "Returns flat array of symmetric LB_Keogh values (upper triangle).\n"
        "Requires band >= 0 (Sakoe-Chiba constraint).");

  m.attr("CUDA_AVAILABLE") = true;
#else
  m.def("cuda_available", []() { return false; },
        "Check if CUDA GPU is available.");

  m.def("cuda_device_info", [](int) { return std::string("CUDA not available (not compiled)"); },
        "device_id"_a = 0,
        "Get CUDA device info string.");

  m.def("compute_distance_matrix_cuda",
        [](const std::vector<std::vector<double>> &, int, bool, int, bool,
           bool, double) -> nb::object {
          throw std::runtime_error("CUDA support not compiled. Rebuild with -DDTWC_ENABLE_CUDA=ON");
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "device_id"_a = 0, "verbose"_a = false,
        "use_lb_keogh"_a = false, "lb_threshold"_a = -1.0,
        "Compute NxN DTW distance matrix on CUDA GPU (requires CUDA build).");

  m.def("compute_lb_keogh_cuda",
        [](const std::vector<std::vector<double>> &, int, int) -> nb::object {
          throw std::runtime_error("CUDA support not compiled. Rebuild with -DDTWC_ENABLE_CUDA=ON");
        },
        "series"_a, "band"_a, "device_id"_a = 0,
        "Compute LB_Keogh lower bounds on GPU (requires CUDA build).");

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
           int band, bool use_squared_l2, bool verbose,
           bool use_lb_keogh, double lb_threshold, int lb_envelope_band) {
          dtwc::metal::MetalDistMatOptions opts;
          opts.band = band;
          opts.use_squared_l2 = use_squared_l2;
          opts.verbose = verbose;
          opts.use_lb_keogh = use_lb_keogh;
          opts.lb_threshold = lb_threshold;
          opts.lb_envelope_band = lb_envelope_band;
          std::vector<double> matrix;
          size_t n = 0;
          {
            nb::gil_scoped_release release;
            auto result = dtwc::metal::compute_distance_matrix_metal(series, opts);
            n = result.n;
            matrix = checked_square_matrix(std::move(result.matrix), n, "Metal");
          }
          return adopt_as_ndarray(std::move(matrix), {n, n});
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "verbose"_a = false,
        "use_lb_keogh"_a = false, "lb_threshold"_a = 0.0,
        "lb_envelope_band"_a = -1,
        "Compute NxN DTW distance matrix on Apple GPU via Metal.\n\n"
        "Returns NxN numpy array of DTW distances. Pairs whose LB_Keogh lower\n"
        "bound exceeds `lb_threshold` are pruned (finite double-max result\n"
        "sentinel, not IEEE infinity) when\n"
        "`use_lb_keogh=True` on a wavefront dispatch path.");

  m.attr("METAL_AVAILABLE") = true;
#else
  m.def("metal_available", []() { return false; },
        "Check if Metal GPU is available.");
  m.def("metal_device_info", []() { return std::string("Metal not available (not compiled)"); },
        "Get Metal device info string.");
  m.def("compute_distance_matrix_metal",
        [](const std::vector<std::vector<double>> &, int, bool, bool,
           bool, double, int) -> nb::object {
          throw std::runtime_error("Metal support not compiled. Rebuild on macOS with -DDTWC_ENABLE_METAL=ON");
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "verbose"_a = false,
        "use_lb_keogh"_a = false, "lb_threshold"_a = 0.0,
        "lb_envelope_band"_a = -1,
        "Compute NxN DTW distance matrix on Apple GPU (requires Metal build).");
  m.attr("METAL_AVAILABLE") = false;
#endif

  // =========================================================================
  // Capability detection: OpenMP, MPI
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

#ifdef DTWC_HAS_MPI
  m.attr("MPI_AVAILABLE") = true;
#else
  m.attr("MPI_AVAILABLE") = false;
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
#ifdef DTWC_HAS_MPI
    info += "  MPI:    available\n";
#else
    info += "  MPI:    not compiled (rebuild with -DDTWC_ENABLE_MPI=ON)\n";
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
