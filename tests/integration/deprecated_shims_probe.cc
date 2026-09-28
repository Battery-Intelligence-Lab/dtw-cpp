// Compile probe for the 1.x [[deprecated]] shims that docs/api-contract-2.0.md
// section 3 keeps until 3.0 (test_deprecated_shims_warn in tests/CMakeLists.txt).
// Compiled, never linked or run: the test passes only when this file compiles
// AND the compiler names every shim used below in a deprecation diagnostic. One
// use of each shim kind stands in for all 24. One use per line, because MSVC
// reports only the first C4996 on a line, and uses rather than &names, because
// MSVC is silent on taking the address of a deprecated member. The .cc suffix
// keeps this file out of the *.cpp Catch2 glob.
#include <dtwc.hpp>

namespace {

[[maybe_unused]] void use_deprecated_shims(dtwc::Problem &problem,
                                           dtwc::DataLoader &loader)
{
  problem.set_numberOfClusters(2);                        // Problem method
  (void)problem.distByInd(0, 1);                          // Problem method, hot path
  (void)problem.maxIter;                                  // public field
  (void)loader.startColumn(1);                            // DataLoader setter
  problem.fillDistanceMatrix();                           // Problem compute method
}

} // namespace
