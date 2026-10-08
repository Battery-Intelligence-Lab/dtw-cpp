// GCC on x86-64 keeps 64-byte lanes (9d6ccc48's successor): compiles only if W is 8 doubles and 16 floats.
#include "core/dtw_kernel.hpp"
static_assert(dtwc::core::dtw_lanes<double> == 8 && dtwc::core::dtw_lanes<float> == 16);
