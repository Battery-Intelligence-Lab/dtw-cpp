// What the fmin cell changes for input the checked paths refuse: NaN or +-inf in a series, through the
// unchecked per-pair layer (warping.hpp: "anything else comes back as NaN, as max() or as an
// ordinary-looking number") and the lanes kernel.
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"
#include "v_fmin.hpp"
#include <cmath>
#include <cstdio>
#include <limits>
namespace dc = dtwc::core;
struct L1 { double operator()(double a, double b) const { return std::abs(a - b); } };
int main() {
  const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
  struct Case { const char *name; double x[5]; double y[5]; };
  const Case cases[] = {
    { "NaN mid-x", { 1, 2, nan, 4, 5 }, { 1, 2, 3, 4, 5 } },
    { "NaN first", { nan, 2, 3, 4, 5 }, { 1, 2, 3, 4, 5 } },
    { "NaN last", { 1, 2, 3, 4, nan }, { 1, 2, 3, 4, 5 } },
    { "+inf mid", { 1, 2, inf, 4, 5 }, { 1, 2, 3, 4, 5 } },
    { "+inf both", { 1, inf, 3, 4, 5 }, { 1, inf, 3, 4, 5 } },
    { "finite", { 1, 3, 2, 5, 4 }, { 1, 2, 3, 4, 5 } },
  };
  for (const Case &c : cases) {
    const double a = dc::run_dtw<dc::SpanL1Cost>(c.x, 5, c.y, 5, -1, dc::StandardCell{}, -1.0);
    const double b = v_fmin::run_dtw<dc::SpanL1Cost>(c.x, 5, c.y, 5, -1, v_fmin::StandardCell{}, -1.0);
    const double *ys[8]; for (auto &p : ys) p = c.y;
    const double la = dc::dtw_kernel_lanes<double>(c.x, ys, 5, -1, L1{}, dc::StandardCell{})[0];
    const double lb = v_fmin::dtw_kernel_lanes<double>(c.x, ys, 5, -1, L1{}, v_fmin::StandardCell{})[0];
    std::printf("%-10s per-pair shipped %-8g fmin %-8g | lanes shipped %-8g fmin %-8g %s\n", c.name, a, b, la, lb,
                (std::isnan(a) == std::isnan(b) && (std::isnan(a) || a == b)) ? "" : "  <- differs");
  }
}
