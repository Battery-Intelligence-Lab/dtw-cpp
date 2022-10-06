// cppimport
#include <pybind11/pybind11.h>

#include "../dtwc/dataTypes.hpp"


namespace py = pybind11;

int square(int x)
{
  return x * x;
}

PYBIND11_MODULE(dtwcpy, m)
{
  m.def("square", &square);

  py::class_<VecMatrix<double>>(m, "VecMatrix")
    .def(py::init<int>());
}
/*
<%
setup_pybind11(cfg)
%>
*/