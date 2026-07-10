#include "warm_start.hpp"

#include "../Problem.hpp"
#include "../algorithms/fast_pam.hpp"
#include "../settings.hpp"

namespace dtwc::mip {

core::ClusteringResult make_warm_start(
  Problem &prob, std::uint64_t random_seed)
{
  return fast_pam_seeded(
    prob, prob.n_clusters(), random_seed, settings::DEFAULT_MAX_ITER);
}

} // namespace dtwc::mip
