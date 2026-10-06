// z_normalize and lb_keogh as the wheel's binding TU (python/src/_dtwcpp_core.cpp, built -O3 ... -Os) compiles them.
#include "core/z_normalize.hpp"
#include "core/lower_bound_impl.hpp"
void zn(double *x, std::size_t n) { dtwc::core::z_normalize(x, n); }
double lbk(const double *q, std::size_t n, const double *u, const double *l) { return dtwc::core::lb_keogh(q, n, u, l); }
