// PF-5 vectorisation probe: exported wrappers force the header-only hot loops into existence so clang's
// loop-vectorize remarks can be read off them (internal-linkage instantiations would be dropped at -O3).
#include "core/distance_matrix.hpp"
#include "core/lower_bound_impl.hpp"
#include "core/mmap_distance_matrix.hpp"
#include "core/z_normalize.hpp"

#include <cstddef>

extern "C" {
double pf5_lb_keogh_f64(const double *q, std::size_t n, const double *u, const double *l) { return dtwc::core::lb_keogh(q, n, u, l); }
float pf5_lb_keogh_f32(const float *q, std::size_t n, const float *u, const float *l) { return dtwc::core::lb_keogh(q, n, u, l); }
void pf5_envelopes_f64(const double *s, std::size_t n, int band, double *u, double *l) { dtwc::core::compute_envelopes(s, n, band, u, l); }
void pf5_z_normalize_f64(double *s, std::size_t n) { dtwc::core::z_normalize(s, n); }
void pf5_z_normalize_f32(float *s, std::size_t n) { dtwc::core::z_normalize(s, n); }
double pf5_dense_max(const dtwc::core::DenseDistanceMatrix &m) { return m.max(); }
std::size_t pf5_dense_count(const dtwc::core::DenseDistanceMatrix &m) { return m.count_computed(); }
bool pf5_dense_all(const dtwc::core::DenseDistanceMatrix &m) { return m.all_computed(); }
double pf5_mmap_max(const dtwc::core::MmapDistanceMatrix &m) { return m.max(); }
std::size_t pf5_mmap_count(const dtwc::core::MmapDistanceMatrix &m) { return m.count_computed(); }
bool pf5_mmap_all(const dtwc::core::MmapDistanceMatrix &m) { return m.all_computed(); }
}
