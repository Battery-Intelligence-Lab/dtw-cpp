/**
 * @file _dtwcpp_core.cpp
 * @brief nanobind Python bindings for DTWC++.
 *
 * @details Exposes DTW distance functions, clustering algorithms,
 *          distance matrix, and scoring to Python with zero-copy
 *          numpy integration where possible.
 *
 * @date 28 Mar 2026
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>

#include <dtwc.hpp>
#include <warping.hpp>
#include <warping_ddtw.hpp>
#include <warping_wdtw.hpp>
#include <warping_adtw.hpp>
#include <soft_dtw.hpp>
#include <algorithms/fast_pam.hpp>
#include <scores.hpp>
#include <core/z_normalize.hpp>
#include <core/dtw_options.hpp>

#include <vector>
#include <string>

namespace nb = nanobind;
using namespace nb::literals; // for _a arg names

NB_MODULE(_dtwcpp_core, m) {
  m.doc() = "DTWC++ — Fast Dynamic Time Warping and Clustering (C++ core)";

  // =========================================================================
  // Enums
  // =========================================================================

  nb::enum_<dtwc::Method>(m, "Method")
    .value("Kmedoids", dtwc::Method::Kmedoids)
    .value("MIP", dtwc::Method::MIP);

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
    .def("to_numpy", [](dtwc::core::DenseDistanceMatrix &dm) {
      size_t n = dm.size();
      size_t shape[2] = {n, n};
      return nb::ndarray<nb::numpy, double>(dm.raw(), 2, shape, nb::handle());
    }, nb::rv_policy::reference_internal,
       "Return a numpy view of the distance matrix (zero-copy).")
    .def("write_csv", &dtwc::core::DenseDistanceMatrix::write_csv, "path"_a)
    .def("read_csv", &dtwc::core::DenseDistanceMatrix::read_csv, "path"_a)
    .def("__repr__", [](const dtwc::core::DenseDistanceMatrix &dm) {
      return "DenseDistanceMatrix(n=" + std::to_string(dm.size()) + ")";
    });

  // =========================================================================
  // DTW distance functions
  // =========================================================================

  m.def("dtw_distance", [](const std::vector<double> &x,
                            const std::vector<double> &y,
                            int band) {
    nb::gil_scoped_release release;
    return dtwc::dtwBanded<double>(x, y, band);
  }, "x"_a, "y"_a, "band"_a = -1,
     "Compute DTW distance between two time series.\n\n"
     "band=-1 for full DTW, band>0 for Sakoe-Chiba banded DTW.");

  m.def("ddtw_distance", [](const std::vector<double> &x,
                              const std::vector<double> &y,
                              int band) {
    nb::gil_scoped_release release;
    return dtwc::ddtwBanded<double>(x, y, band);
  }, "x"_a, "y"_a, "band"_a = -1,
     "Compute Derivative DTW distance.");

  m.def("wdtw_distance", [](const std::vector<double> &x,
                              const std::vector<double> &y,
                              int band, double g) {
    nb::gil_scoped_release release;
    return dtwc::wdtwBanded<double>(x, y, band, g);
  }, "x"_a, "y"_a, "band"_a = -1, "g"_a = 0.05,
     "Compute Weighted DTW distance with logistic weight steepness g.");

  m.def("adtw_distance", [](const std::vector<double> &x,
                              const std::vector<double> &y,
                              int band, double penalty) {
    nb::gil_scoped_release release;
    return dtwc::adtwBanded<double>(x, y, penalty, band);
  }, "x"_a, "y"_a, "band"_a = -1, "penalty"_a = 1.0,
     "Compute Amerced DTW distance with non-diagonal step penalty.");

  m.def("soft_dtw_distance", [](const std::vector<double> &x,
                                 const std::vector<double> &y,
                                 double gamma) {
    nb::gil_scoped_release release;
    return dtwc::soft_dtw<double>(x, y, gamma);
  }, "x"_a, "y"_a, "gamma"_a = 1.0,
     "Compute Soft-DTW distance (differentiable).");

  m.def("soft_dtw_gradient", [](const std::vector<double> &x,
                                 const std::vector<double> &y,
                                 double gamma) {
    nb::gil_scoped_release release;
    return dtwc::soft_dtw_gradient<double>(x, y, gamma);
  }, "x"_a, "y"_a, "gamma"_a = 1.0,
     "Compute Soft-DTW gradient w.r.t. first series x.");

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
    .def_rw("p_vec", &dtwc::Data::p_vec)
    .def_rw("p_names", &dtwc::Data::p_names)
    .def_prop_ro("size", &dtwc::Data::size);

  // =========================================================================
  // Problem class
  // =========================================================================

  nb::class_<dtwc::Problem>(m, "Problem")
    .def(nb::init<>())
    .def("__init__", [](dtwc::Problem *p, const std::string &name) {
      new (p) dtwc::Problem(name);
    }, "name"_a)
    // Properties for public fields
    .def_rw("method", &dtwc::Problem::method)
    .def_rw("max_iter", &dtwc::Problem::maxIter)
    .def_rw("n_repetition", &dtwc::Problem::N_repetition)
    .def_rw("band", &dtwc::Problem::band)
    .def_rw("variant_params", &dtwc::Problem::variant_params)
    .def_rw("name", &dtwc::Problem::name)
    .def_rw("clusters_ind", &dtwc::Problem::clusters_ind)
    .def_rw("centroids_ind", &dtwc::Problem::centroids_ind)
    // Getters
    .def_prop_ro("size", &dtwc::Problem::size)
    .def_prop_ro("cluster_size", &dtwc::Problem::cluster_size)
    .def("is_distance_matrix_filled", &dtwc::Problem::isDistanceMatrixFilled)
    .def("max_distance", &dtwc::Problem::maxDistance)
    .def("dist_by_ind", &dtwc::Problem::distByInd, "i"_a, "j"_a)
    // Setters
    .def("set_number_of_clusters", &dtwc::Problem::set_numberOfClusters, "n_clusters"_a)
    .def("set_variant", nb::overload_cast<dtwc::core::DTWVariant>(&dtwc::Problem::set_variant), "variant"_a)
    .def("set_data", [](dtwc::Problem &p, std::vector<std::vector<double>> series,
                         std::vector<std::string> names) {
      dtwc::Data d(std::move(series), std::move(names));
      p.set_data(std::move(d));
    }, "series"_a, "names"_a, "Set time series data.")
    // Distance matrix
    .def("fill_distance_matrix", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.fillDistanceMatrix();
    }, "Compute all pairwise DTW distances.")
    .def("refresh_distance_matrix", &dtwc::Problem::refreshDistanceMatrix)
    // Clustering
    .def("cluster", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.cluster();
    }, "Run clustering (Lloyd k-medoids or MIP).")
    .def("find_total_cost", &dtwc::Problem::findTotalCost)
    .def("assign_clusters", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.assignClusters();
    })
    .def("calculate_medoids", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.calculateMedoids();
    })
    // I/O
    .def("print_clusters", &dtwc::Problem::printClusters)
    .def("write_clusters", &dtwc::Problem::writeClusters)
    .def("write_distance_matrix", nb::overload_cast<>(&dtwc::Problem::writeDistanceMatrix, nb::const_))
    .def("write_silhouettes", &dtwc::Problem::writeSilhouettes)
    .def("__repr__", [](const dtwc::Problem &p) {
      return "Problem(name='" + p.name + "', n=" + std::to_string(p.size())
             + ", k=" + std::to_string(p.cluster_size()) + ")";
    });

  // =========================================================================
  // FastPAM
  // =========================================================================

  m.def("fast_pam", [](dtwc::Problem &prob, int n_clusters, int max_iter) {
    nb::gil_scoped_release release;
    return dtwc::fast_pam(prob, n_clusters, max_iter);
  }, "prob"_a, "n_clusters"_a, "max_iter"_a = 100,
     "Run FastPAM k-medoids clustering (Schubert & Rousseeuw 2021).");

  // =========================================================================
  // Scores
  // =========================================================================

  m.def("silhouette", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::silhouette(prob);
  }, "prob"_a, "Compute silhouette scores for each data point.");

  m.def("davies_bouldin_index", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::daviesBouldinIndex(prob);
  }, "prob"_a, "Compute Davies-Bouldin Index.");
}
