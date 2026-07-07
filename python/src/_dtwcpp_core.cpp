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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <dtwc.hpp>
#include <env.hpp>
#include <error.hpp>
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
#include <algorithms/clarans.hpp>
#include <algorithms/hierarchical.hpp>
#include <scores.hpp>
#include <core/z_normalize.hpp>
#include <core/dtw_options.hpp>
#include <core/pruned_distance_matrix.hpp>
#include <core/matrix_io.hpp>
#include <test_api.hpp> // dtwc::test::parallelisation()/gpu() introspection (Task 3.3)

#include <Eigen/Core>

#include <cstring>
#include <vector>
#include <string>

namespace nb = nanobind;
using namespace nb::literals; // for _a arg names

NB_MODULE(_dtwcpp_core, m) {
  m.doc() = "DTWC++ — Fast Dynamic Time Warping and Clustering (C++ core)";

  // =========================================================================
  // Error taxonomy (api-contract-2.0.md §5)
  // =========================================================================
  // One base (DtwcError) + four leaves. Each leaf subclasses BOTH DtwcError AND
  // the closest built-in (ValueError / RuntimeError / OSError) so idiomatic
  // `except ValueError:` and `except dtwcpp.InvalidInput:` both catch it. The
  // types are function-local statics (module lifetime) referenced by the single
  // captureless translator below (registered so it runs before the default one).
  static PyObject *g_exc_base = PyErr_NewException("dtwcpp.DtwcError", PyExc_Exception, nullptr);
  static PyObject *g_exc_invalid = nullptr;
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
    g_exc_solver = make_leaf("dtwcpp.SolverError", PyExc_RuntimeError);
    g_exc_device = make_leaf("dtwcpp.DeviceError", PyExc_RuntimeError);
    g_exc_io = make_leaf("dtwcpp.IOError", PyExc_OSError);
  }
  m.attr("DtwcError") = nb::borrow(g_exc_base);
  m.attr("InvalidInput") = nb::borrow(g_exc_invalid);
  m.attr("SolverError") = nb::borrow(g_exc_solver);
  m.attr("DeviceError") = nb::borrow(g_exc_device);
  m.attr("IOError") = nb::borrow(g_exc_io);

  nb::register_exception_translator(
    [](const std::exception_ptr &p, void * /*payload*/) {
      try {
        std::rethrow_exception(p);
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

  // =========================================================================
  // Enums
  // =========================================================================

  nb::enum_<dtwc::Method>(m, "Method")
    .value("Kmedoids", dtwc::Method::Kmedoids)
    .value("MIP", dtwc::Method::MIP)
    .value("LRCore", dtwc::Method::LRCore);

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
    .value("SoftDTW", dtwc::core::DTWVariant::SoftDTW);

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
    .value("KimKeogh", dtwc::LowerBoundStrategy::KimKeogh);

  // =========================================================================
  // CUDASettings
  // =========================================================================

  nb::class_<dtwc::CUDASettings>(m, "CUDASettings")
    .def(nb::init<>())
    .def_rw("device_id", &dtwc::CUDASettings::device_id, "CUDA device index (default 0).")
    .def_rw("precision", &dtwc::CUDASettings::precision,
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

  // =========================================================================
  // DTWVariantParams
  // =========================================================================

  nb::class_<dtwc::core::DTWVariantParams>(m, "DTWVariantParams")
    .def(nb::init<>())
    .def_rw("variant", &dtwc::core::DTWVariantParams::variant)
    .def_rw("wdtw_g", &dtwc::core::DTWVariantParams::wdtw_g)
    .def_rw("adtw_penalty", &dtwc::core::DTWVariantParams::adtw_penalty)
    .def_rw("sdtw_gamma", &dtwc::core::DTWVariantParams::sdtw_gamma);

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
    .def("__repr__", [](const dtwc::MIPSettings &s) {
      return "MIPSettings(gap=" + std::to_string(s.mip_gap)
             + ", time_limit=" + std::to_string(s.time_limit_sec)
             + ", warm_start=" + (s.warm_start ? "True" : "False")
             + ", numeric_focus=" + std::to_string(s.numeric_focus)
             + ", mip_focus=" + std::to_string(s.mip_focus)
             + ", benders=" + s.benders
             + ", max_benders_iter=" + std::to_string(s.max_benders_iter)
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
      const Eigen::MatrixXd full = dtwc::io::to_full_matrix(dm);
      const size_t n = dm.size();
      double *ptr = new double[n * n];
      // Eigen is column-major; numpy expects row-major. The matrix is
      // symmetric, so the byte layout is identical and memcpy is correct.
      std::memcpy(ptr, full.data(), n * n * sizeof(double));
      nb::capsule owner(ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
      return nb::ndarray<nb::numpy, double>(ptr, {n, n}, owner);
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
    nb::gil_scoped_release release;
    auto mt = dtwc::core::MetricType::L1;
    if (metric == "squared_euclidean" || metric == "sqeuclidean")
        mt = dtwc::core::MetricType::SquaredL2;
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
    nb::gil_scoped_release release;
    auto mt = dtwc::core::MetricType::L1;
    if (metric == "squared_euclidean" || metric == "sqeuclidean")
        mt = dtwc::core::MetricType::SquaredL2;
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
    nb::gil_scoped_release release;
    auto mt = dtwc::core::MetricType::L1;
    if (metric == "squared_euclidean" || metric == "sqeuclidean")
        mt = dtwc::core::MetricType::SquaredL2;
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

  // =========================================================================
  // Problem class
  // =========================================================================

  // Shared read accessor: fill (if needed) and expand packed triangular → NxN.
  // Named `distance_matrix()` in 2.0 (was `distance_matrix_numpy()`); always a
  // COPY because the C++ store keeps only the upper triangle, so a zero-copy view
  // into a full NxN layout is structurally impossible (§2.2 ‡).
  auto read_distance_matrix_np = [](dtwc::Problem &prob) {
    {
      nb::gil_scoped_release release;
      prob.fill_distance_matrix();
    }
    const auto &dm = prob.dense_distance_matrix();
    const Eigen::MatrixXd full = dtwc::io::to_full_matrix(dm);
    const size_t n = dm.size();
    double *ptr = new double[n * n];
    // Symmetric matrix: col-major == row-major, safe to memcpy.
    std::memcpy(ptr, full.data(), n * n * sizeof(double));
    nb::capsule owner(ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return nb::ndarray<nb::numpy, double>(ptr, {n, n}, owner);
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

  nb::class_<dtwc::Problem>(m, "Problem")
    .def(nb::init<>())
    .def("__init__", [](dtwc::Problem *p, const std::string &name) {
      new (p) dtwc::Problem(name);
    }, "name"_a)
    // ---- config fields (canonical names) ----
    .def_rw("method", &dtwc::Problem::method)
    .def_rw("max_iter", &dtwc::Problem::maxIter)
    .def_rw("n_repetitions", &dtwc::Problem::N_repetition,
            "Repetitions for iterative methods.")
    .def_rw("n_repetition", &dtwc::Problem::N_repetition,
            "Deprecated alias for n_repetitions (kept one cycle, §4).")
    .def_rw("band", &dtwc::Problem::band)
    .def_rw("variant_params", &dtwc::Problem::variant_params)
    .def_rw("missing_strategy", &dtwc::Problem::missing_strategy,
            "Strategy for handling NaN values (Error, ZeroCost, AROW, Interpolate).")
    .def_rw("distance_strategy", &dtwc::Problem::distance_strategy,
            "Distance matrix computation strategy (Auto, BruteForce, Pruned, CUDA, Metal).")
    .def_rw("lb_strategy", &dtwc::Problem::lb_strategy,
            "Lower-bound selection for the Pruned CPU path (Auto/None/Kim/Keogh/KimKeogh).")
    .def_rw("storage_policy", &dtwc::Problem::storage_policy,
            "How series data is stored (Auto/Heap/Mmap).")
    .def_rw("cuda_settings", &dtwc::Problem::cuda_settings,
            "GPU compute options (used when distance_strategy == CUDA).")
    .def_rw("mip_settings", &dtwc::Problem::mip_settings,
            "MIP solver tuning parameters.")
    .def_rw("verbose", &dtwc::Problem::verbose,
            "Print progress messages for long-running operations.")
    .def_rw("name", &dtwc::Problem::name)
    .def_prop_rw("output_folder",
                 [](const dtwc::Problem &p) { return p.output_folder; },
                 [](dtwc::Problem &p, std::filesystem::path dir) { p.output_folder = std::move(dir); },
                 "Output folder for results written by the write_* methods.")
    .def_rw("clusters_ind", &dtwc::Problem::clusters_ind)
    .def_rw("centroids_ind", &dtwc::Problem::centroids_ind)
    // ---- read accessors ----
    .def_prop_ro("size", &dtwc::Problem::size)
    .def("n_clusters", &dtwc::Problem::n_clusters, "Number of clusters (was cluster_size()).")
    .def_prop_ro("cluster_size", [](const dtwc::Problem &p) { return p.n_clusters(); },
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
    .def("dist_by_ind", &dtwc::Problem::dist_by_ind, "i"_a, "j"_a)
    // ---- config setters ----
    .def("set_n_clusters", &dtwc::Problem::set_n_clusters, "n_clusters"_a)
    .def("set_number_of_clusters", [](dtwc::Problem &p, int n) { p.set_n_clusters(n); },
         "n_clusters"_a, "Deprecated alias for set_n_clusters (kept one cycle, §4).")
    .def("set_method", &dtwc::Problem::set_method, "method"_a)
    .def("set_band", &dtwc::Problem::set_band, "band"_a)
    .def("set_max_iter", &dtwc::Problem::set_max_iter, "max_iter"_a)
    .def("set_n_repetitions", &dtwc::Problem::set_n_repetitions, "n_repetitions"_a)
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
    .def("distance_matrix_numpy", read_distance_matrix_np,
         "Deprecated alias for distance_matrix() (kept one cycle, §4).")
    .def("set_distance_matrix", write_distance_matrix_np, "dm"_a,
         "Load a precomputed NxN distance matrix (e.g. from a GPU compute).")
    .def("set_distance_matrix_from_numpy", write_distance_matrix_np, "dm"_a,
         "Deprecated alias for set_distance_matrix() (kept one cycle, §4).")
    .def("refresh_distance_matrix", &dtwc::Problem::refresh_distance_matrix)
    .def("read_distance_matrix", &dtwc::Problem::read_distance_matrix, "path"_a,
         "Read a distance matrix from a CSV file.")
    .def("print_distance_matrix", &dtwc::Problem::print_distance_matrix)
    .def("use_mmap_distance_matrix", &dtwc::Problem::use_mmap_distance_matrix, "cache_path"_a,
         "Back the distance matrix with a memory-mapped cache file (big-N / resume).")
    // ---- clustering ----
    .def("cluster", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.cluster();
    }, "Run clustering (Lloyd k-medoids or MIP).")
    .def("find_total_cost", &dtwc::Problem::find_total_cost)
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
    .def("write_clusters", &dtwc::Problem::write_clusters)
    .def("write_medoid_members", &dtwc::Problem::write_medoid_members, "iter"_a, "rep"_a = 0)
    .def("write_distance_matrix", nb::overload_cast<>(&dtwc::Problem::write_distance_matrix, nb::const_))
    .def("write_silhouettes", &dtwc::Problem::write_silhouettes)
    .def("__repr__", [](const dtwc::Problem &p) {
      return "Problem(name='" + p.name + "', n=" + std::to_string(p.size())
             + ", k=" + std::to_string(p.n_clusters()) + ")";
    });

  // =========================================================================
  // Distance matrix convenience function
  // =========================================================================

  m.def("compute_distance_matrix", [](const std::vector<std::vector<double>> &series,
                                        int band, const std::string &metric,
                                        bool use_pruning) {
    // Task 3.6 (review H1): this high-level Python compute path never constructs
    // dtwc::env(), so its OpenMP warning would otherwise be silent under
    // OMP_NUM_THREADS=1. Warn once, deterministically, before either branch (the
    // pruned branch also warns via get_max_threads; this covers the unpruned one).
    dtwc::warn_if_single_threaded();

    auto mt = dtwc::core::MetricType::L1;
    if (metric == "squared_euclidean" || metric == "sqeuclidean")
        mt = dtwc::core::MetricType::SquaredL2;

    const size_t n = series.size();
    double* ptr = new double[n * n]();  // zero-init

    // Release GIL only for the compute-heavy section
    {
      nb::gil_scoped_release release;

      if (use_pruning && (mt == dtwc::core::MetricType::L1 || mt == dtwc::core::MetricType::L2)) {
        // LB-pruned version: precomputes envelopes + summaries,
        // uses early-abandon DTW guided by LB_Kim / LB_Keogh thresholds.
        dtwc::core::compute_distance_matrix_pruned(series, ptr, band, mt);
      } else {
        // Standard unpruned version (for non-L1 metrics or when pruning disabled).
        // Lock-free by design: each thread owns a disjoint set of rows (outer loop i).
        // Writes to ptr[i*n+j] and ptr[j*n+i] never collide across threads because
        // no two threads share the same i value.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
        #endif
        for (int i = 0; i < static_cast<int>(n); ++i) {
            for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
                double d = (band >= 0)
                    ? dtwc::dtwBanded<double>(series[i], series[j], band, -1.0, mt)
                    : dtwc::dtwFull_L<double>(series[i], series[j], -1.0, mt);
                ptr[i * n + j] = d;
                ptr[j * n + i] = d;
            }
        }
      }
    }  // GIL re-acquired here

    nb::capsule owner(ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double>(ptr, {n, n}, owner);
  }, "series"_a, "band"_a = -1, "metric"_a = "l1", "use_pruning"_a = true,
     "Compute pairwise DTW distance matrix entirely in C++.\n\n"
     "Returns NxN numpy array. Uses OpenMP parallelism when available.\n"
     "When use_pruning=True (default), uses LB_Kim and LB_Keogh lower\n"
     "bounds with early-abandon DTW for faster computation (L1 metric only).\n"
     "Much faster than calling dtw_distance in a Python loop.");

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
     "n_samples"_a = 5, "max_iter"_a = 100, "seed"_a = 42,
     "Run FastCLARA scalable k-medoids clustering.\n\n"
     "Runs FastPAM on random subsamples and assigns all points to the\n"
     "best medoids found. Avoids O(N^2) memory of full PAM.\n\n"
     "The C++ core writes labels/medoids/k back into prob (since 1.6), so\n"
     "silhouette(prob) and davies_bouldin(prob) work after this call (§2.5).\n\n"
     "Parameters:\n"
     "  prob: Problem with data loaded.\n"
     "  n_clusters: Number of clusters (k).\n"
     "  sample_size: Subsample size (-1 = auto: 40 + 2*k).\n"
     "  n_samples: Number of subsamples to try (default 5).\n"
     "  max_iter: Max PAM iterations per subsample (default 100).\n"
     "  seed: Random seed for reproducibility (default 42).");

  // =========================================================================
  // Checkpointing
  // =========================================================================

  nb::class_<dtwc::CheckpointOptions>(m, "CheckpointOptions")
    .def(nb::init<>())
    .def_rw("directory", &dtwc::CheckpointOptions::directory)
    .def_rw("save_interval", &dtwc::CheckpointOptions::save_interval)
    .def_rw("enabled", &dtwc::CheckpointOptions::enabled)
    .def("__repr__", [](const dtwc::CheckpointOptions &o) {
      return "CheckpointOptions(dir='" + o.directory
             + "', interval=" + std::to_string(o.save_interval)
             + ", enabled=" + (o.enabled ? "True" : "False") + ")";
    });

  m.def("save_checkpoint", &dtwc::save_checkpoint, "prob"_a, "path"_a,
        "Save distance matrix checkpoint to directory.\n\n"
        "Creates distances.csv and metadata.txt in the given directory.\n"
        "The directory is created if it does not exist.");

  m.def("load_checkpoint", &dtwc::load_checkpoint, "prob"_a, "path"_a,
        "Load distance matrix checkpoint from directory.\n\n"
        "Returns True if checkpoint was loaded successfully, False otherwise.\n"
        "Validates that matrix dimensions match the Problem's data size.\n"
        "Sets distance matrix filled flag if all pairs are computed.");

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
    nb::gil_scoped_release release;
    return dtwc::scores::davies_bouldin(prob);
  }, "prob"_a, "Deprecated alias for davies_bouldin() (kept one cycle, §4).");

  m.def("dunn", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::dunn(prob);
  }, "prob"_a,
     "Compute Dunn index (min inter-cluster distance / max intra-cluster diameter).");
  m.def("dunn_index", [](dtwc::Problem &prob) {
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
          nb::gil_scoped_release release;
          auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
          size_t n = result.n;
          double* data = new double[n * n];
          std::copy(result.matrix.begin(), result.matrix.end(), data);
          nb::gil_scoped_acquire acquire;
          nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
          return nb::ndarray<nb::numpy, double>(data, {n, n}, owner);
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "device_id"_a = 0, "verbose"_a = false,
        "use_lb_keogh"_a = false, "lb_threshold"_a = -1.0,
        "Compute NxN DTW distance matrix on CUDA GPU.\n\n"
        "Returns NxN numpy array of DTW distances.\n"
        "When `use_lb_keogh=True` and `lb_threshold > 0`, pairs whose LB_Keogh\n"
        "lower bound exceeds `lb_threshold` are pruned (+inf in result).");

  m.def("compute_lb_keogh_cuda",
        [](const std::vector<std::vector<double>> &series,
           int band, int device_id) {
          nb::gil_scoped_release release;
          auto result = dtwc::cuda::compute_lb_keogh_cuda(series, band, device_id);
          size_t np = result.lb_values.size();
          double* data = new double[np];
          std::copy(result.lb_values.begin(), result.lb_values.end(), data);
          nb::gil_scoped_acquire acquire;
          nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
          return nb::ndarray<nb::numpy, double>(data, {np}, owner);
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
          nb::gil_scoped_release release;
          auto result = dtwc::metal::compute_distance_matrix_metal(series, opts);
          size_t n = result.n;
          double *data = new double[n * n];
          std::copy(result.matrix.begin(), result.matrix.end(), data);
          nb::gil_scoped_acquire acquire;
          nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
          return nb::ndarray<nb::numpy, double>(data, {n, n}, owner);
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "verbose"_a = false,
        "use_lb_keogh"_a = false, "lb_threshold"_a = 0.0,
        "lb_envelope_band"_a = -1,
        "Compute NxN DTW distance matrix on Apple GPU via Metal.\n\n"
        "Returns NxN numpy array of DTW distances. Pairs whose LB_Keogh lower\n"
        "bound exceeds `lb_threshold` are pruned (result entry +inf) when\n"
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
