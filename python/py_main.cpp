#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../dtwc/dtwc.hpp"

#include <vector>
#include <string>
#include <string_view>

namespace py = pybind11;

using namespace dtwc;

PYBIND11_MODULE(dtwcpp, m)
{
  m.doc() = "DTWC++ (Dynamic Time Warping Clustering++ Library)";

  py::class_<Problem>(m, "Problem")
    .def(py::init<>())
    .def(py::init<std::string_view>())
    .def(py::init<std::string_view, DataLoader &>())
    .def("size", &Problem::size)
    .def("cluster_size", &Problem::cluster_size)
    .def("get_name", (const std::string &(Problem::*)(size_t) const) & Problem::get_name, py::return_value_policy::reference)
    .def("p_vec", (const std::vector<data_t> &(Problem::*)(size_t) const) & Problem::p_vec, py::return_value_policy::reference)
    .def("refreshDistanceMatrix", &Problem::refreshDistanceMatrix)
    .def("resize", &Problem::resize)
    .def("centroid_of", &Problem::centroid_of)
    .def("readDistanceMatrix", &Problem::readDistanceMatrix)
    .def("set_numberOfClusters", &Problem::set_numberOfClusters)
    .def("set_clusters", &Problem::set_clusters)
    .def("set_solver", &Problem::set_solver)
    .def("set_data", &Problem::set_data)
    .def("maxDistance", &Problem::maxDistance)
    .def("distByInd", &Problem::distByInd)
    .def("isDistanceMatrixFilled", &Problem::isDistanceMatrixFilled)
    .def("fillDistanceMatrix", &Problem::fillDistanceMatrix)
    .def("printDistanceMatrix", &Problem::printDistanceMatrix)
    .def("writeDistanceMatrix", (void(Problem::*)(const std::string &) const) & Problem::writeDistanceMatrix)
    .def("writeClusters", &Problem::writeClusters)
    .def("writeMedoidMembers", &Problem::writeMedoidMembers)
    .def("writeSilhouettes", &Problem::writeSilhouettes)
    .def("init", &Problem::init)
    .def("cluster", &Problem::cluster)
    .def("cluster_and_process", &Problem::cluster_and_process)
    .def("findTotalCost", &Problem::findTotalCost)
    .def("assignClusters", &Problem::assignClusters)
    .def("calculateMedoids", &Problem::calculateMedoids);


  py::class_<DataLoader>(m, "DataLoader")
    .def(py::init<>())
    .def(py::init<const std::filesystem::path &>())
    .def(py::init<const std::filesystem::path &, int>());
  // .def("startColumn", (int(DataLoader::*)()) & DataLoader::startColumn)
  // .def("startRow", (int(DataLoader::*)()) & DataLoader::startRow)
  // .def("n_data", (int(DataLoader::*)()) & DataLoader::n_data)
  // .def("delimiter", (char(DataLoader::*)()) & DataLoader::delimiter)
  // .def("path", (std::filesystem::path(DataLoader::*)()) & DataLoader::path)
  // .def("verbosity", (int(DataLoader::*)()) & DataLoader::verbosity)
  // .def("startColumn", (DataLoader & (DataLoader::*)(int)) & DataLoader::startColumn)
  // .def("startRow", (DataLoader & (DataLoader::*)(int)) & DataLoader::startRow)
  // .def("n_data", &DataLoader::n_data)
  // .def("delimiter", &DataLoader::delimiter)
  // .def("path", &DataLoader::path)
  // .def("verbosity", &DataLoader::verbosity)
  // .def("load", &DataLoader::load);
}