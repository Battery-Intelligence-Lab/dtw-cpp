#include "dtwc/Problem.hpp"

#include <concepts>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename T>
concept raw_private_method_assignable = requires(T &problem) {
  problem.method = dtwc::Method::Kmedoids;
};

template <typename T>
concept raw_private_random_seed_assignable = requires(T &problem) {
  problem.random_seed = std::uint64_t{ 7 };
};

template <typename T>
concept raw_private_last_iterations_assignable = requires(T &problem) {
  problem.last_iterations = 7;
};

template <typename T>
concept raw_private_tadpole_dc_assignable = requires(T &problem) {
  problem.tadpole_dc = 7.0;
};

template <typename T>
concept raw_private_lb_strategy_assignable = requires(T &problem) {
  problem.lb_strategy = dtwc::LowerBoundStrategy::Auto;
};

template <typename T>
concept raw_private_storage_policy_assignable = requires(T &problem) {
  problem.storage_policy = dtwc::core::StoragePolicy::Auto;
};

template <typename T>
concept raw_private_verbose_assignable = requires(T &problem) {
  problem.verbose = true;
};

template <typename T>
concept raw_private_output_folder_assignable = requires(T &problem) {
  problem.output_folder = typename T::path_t{};
};

template <typename T>
concept raw_private_name_assignable = requires(T &problem) {
  problem.name = std::string{};
};

template <typename T>
concept raw_private_data_assignable = requires(T &problem) {
  problem.data = dtwc::Data{};
};

template <typename T>
concept public_resize_callable = requires(T &problem) {
  problem.resize();
};

#if !defined(DTWC_F19_SKIP_PRIVATE_ASSERTS)
static_assert(
  !raw_private_method_assignable<dtwc::Problem>,
  "F19_PRIVATE_method");
static_assert(
  !raw_private_random_seed_assignable<dtwc::Problem>,
  "F19_PRIVATE_random_seed");
static_assert(
  !raw_private_last_iterations_assignable<dtwc::Problem>,
  "F19_PRIVATE_last_iterations");
static_assert(
  !raw_private_tadpole_dc_assignable<dtwc::Problem>,
  "F19_PRIVATE_tadpole_dc");
static_assert(
  !raw_private_lb_strategy_assignable<dtwc::Problem>,
  "F19_PRIVATE_lb_strategy");
static_assert(
  !raw_private_storage_policy_assignable<dtwc::Problem>,
  "F19_PRIVATE_storage_policy");
static_assert(
  !raw_private_verbose_assignable<dtwc::Problem>,
  "F19_PRIVATE_verbose");
static_assert(
  !raw_private_output_folder_assignable<dtwc::Problem>,
  "F19_PRIVATE_output_folder");
static_assert(
  !raw_private_name_assignable<dtwc::Problem>,
  "F19_PRIVATE_name");
static_assert(
  !raw_private_data_assignable<dtwc::Problem>
    && !public_resize_callable<dtwc::Problem>,
  "F19_PRIVATE_data_and_resize");
#endif

template <typename T>
concept has_const_method_getter =
  requires(T &problem, const T &const_problem) {
    { problem.method() } -> std::same_as<dtwc::Method>;
    { const_problem.method() } -> std::same_as<dtwc::Method>;
  };

template <typename T>
concept has_const_random_seed_getter =
  requires(T &problem, const T &const_problem) {
    { problem.random_seed() } -> std::same_as<std::uint64_t>;
    { const_problem.random_seed() } -> std::same_as<std::uint64_t>;
  };

template <typename T>
concept has_const_last_iterations_getter =
  requires(T &problem, const T &const_problem) {
    { problem.last_iterations() } -> std::same_as<int>;
    { const_problem.last_iterations() } -> std::same_as<int>;
  };

template <typename T>
concept has_const_tadpole_dc_getter =
  requires(T &problem, const T &const_problem) {
    { problem.tadpole_dc() } -> std::same_as<double>;
    { const_problem.tadpole_dc() } -> std::same_as<double>;
  };

template <typename T>
concept has_const_lb_strategy_getter =
  requires(T &problem, const T &const_problem) {
    { problem.lb_strategy() } -> std::same_as<dtwc::LowerBoundStrategy>;
    { const_problem.lb_strategy() } -> std::same_as<dtwc::LowerBoundStrategy>;
  };

template <typename T>
concept has_const_storage_policy_getter =
  requires(T &problem, const T &const_problem) {
    { problem.storage_policy() } -> std::same_as<dtwc::core::StoragePolicy>;
    { const_problem.storage_policy() } -> std::same_as<dtwc::core::StoragePolicy>;
  };

template <typename T>
concept has_const_verbose_getter =
  requires(T &problem, const T &const_problem) {
    { problem.verbose() } -> std::same_as<bool>;
    { const_problem.verbose() } -> std::same_as<bool>;
  };

template <typename T>
concept has_const_output_folder_getter =
  requires(T &problem, const T &const_problem) {
    { problem.output_folder() } -> std::same_as<const typename T::path_t &>;
    { const_problem.output_folder() }
      -> std::same_as<const typename T::path_t &>;
  };

template <typename T>
concept has_const_name_getter =
  requires(T &problem, const T &const_problem) {
    { problem.name() } -> std::same_as<const std::string &>;
    { const_problem.name() } -> std::same_as<const std::string &>;
  };

template <typename T>
concept has_const_data_getter =
  requires(T &problem, const T &const_problem) {
    { problem.data() } -> std::same_as<const dtwc::Data &>;
    { const_problem.data() } -> std::same_as<const dtwc::Data &>;
  };

#if !defined(DTWC_F19_SKIP_GETTER_ASSERTS)
static_assert(
  has_const_method_getter<dtwc::Problem>,
  "F19_GETTER_method");
static_assert(
  has_const_random_seed_getter<dtwc::Problem>,
  "F19_GETTER_random_seed");
static_assert(
  has_const_last_iterations_getter<dtwc::Problem>,
  "F19_GETTER_last_iterations");
static_assert(
  has_const_tadpole_dc_getter<dtwc::Problem>,
  "F19_GETTER_tadpole_dc");
static_assert(
  has_const_lb_strategy_getter<dtwc::Problem>,
  "F19_GETTER_lb_strategy");
static_assert(
  has_const_storage_policy_getter<dtwc::Problem>,
  "F19_GETTER_storage_policy");
static_assert(
  has_const_verbose_getter<dtwc::Problem>,
  "F19_GETTER_verbose");
static_assert(
  has_const_output_folder_getter<dtwc::Problem>,
  "F19_GETTER_output_folder");
static_assert(
  has_const_name_getter<dtwc::Problem>,
  "F19_GETTER_name");
static_assert(
  has_const_data_getter<dtwc::Problem>,
  "F19_GETTER_data");
#endif

template <typename T>
concept raw_retained_max_iter_assignable = requires(T &problem) {
  problem.maxIter = 7;
};

template <typename T>
concept raw_retained_n_repetition_assignable = requires(T &problem) {
  problem.N_repetition = 7;
};

template <typename T>
concept raw_retained_band_assignable = requires(T &problem) {
  problem.band = 7;
};

template <typename T>
concept raw_retained_variant_params_assignable = requires(T &problem) {
  problem.variant_params = dtwc::core::DTWVariantParams{};
};

template <typename T>
concept raw_retained_missing_strategy_assignable = requires(T &problem) {
  problem.missing_strategy = dtwc::core::MissingStrategy::Error;
};

template <typename T>
concept raw_retained_distance_strategy_assignable = requires(T &problem) {
  problem.distance_strategy = dtwc::DistanceMatrixStrategy::Auto;
};

template <typename T>
concept raw_retained_cuda_settings_assignable = requires(T &problem) {
  problem.cuda_settings = dtwc::CUDASettings{};
};

template <typename T>
concept raw_retained_mip_settings_assignable = requires(T &problem) {
  problem.mip_settings = dtwc::MIPSettings{};
};

template <typename T>
concept raw_retained_init_fun_assignable = requires(T &problem) {
  problem.init_fun = std::function<void(dtwc::Problem &)>{};
};

template <typename T>
concept raw_retained_clusters_assignable = requires(T &problem) {
  problem.clusters_ind = std::vector<int>{};
};

template <typename T>
concept raw_retained_centroids_assignable = requires(T &problem) {
  problem.centroids_ind = std::vector<int>{};
};

#if !defined(DTWC_F19_SKIP_RETAINED_ASSERTS)
static_assert(
  raw_retained_max_iter_assignable<dtwc::Problem>,
  "F19_RETAINED_max_iter");
static_assert(
  raw_retained_n_repetition_assignable<dtwc::Problem>,
  "F19_RETAINED_n_repetition");
static_assert(
  raw_retained_band_assignable<dtwc::Problem>,
  "F19_RETAINED_band");
static_assert(
  raw_retained_variant_params_assignable<dtwc::Problem>,
  "F19_RETAINED_variant_params");
static_assert(
  raw_retained_missing_strategy_assignable<dtwc::Problem>,
  "F19_RETAINED_missing_strategy");
static_assert(
  raw_retained_distance_strategy_assignable<dtwc::Problem>,
  "F19_RETAINED_distance_strategy");
static_assert(
  raw_retained_cuda_settings_assignable<dtwc::Problem>,
  "F19_RETAINED_cuda_settings");
static_assert(
  raw_retained_mip_settings_assignable<dtwc::Problem>,
  "F19_RETAINED_mip_settings");
static_assert(
  raw_retained_init_fun_assignable<dtwc::Problem>,
  "F19_RETAINED_init_fun");
static_assert(
  raw_retained_clusters_assignable<dtwc::Problem>,
  "F19_RETAINED_clusters");
static_assert(
  raw_retained_centroids_assignable<dtwc::Problem>,
  "F19_RETAINED_centroids");
#endif

#if !defined(DTWC_F19_SKIP_SETTER_EXERCISE)
[[maybe_unused]] void exercise_private_field_setters(
  dtwc::Problem &problem,
  dtwc::Data replacement_data,
  dtwc::Problem::path_t replacement_output_folder)
{
  problem.set_method(dtwc::Method::Kmedoids);
  problem.set_random_seed(7);
  problem.set_tadpole_dc(7.0);
  problem.set_lb_strategy(dtwc::LowerBoundStrategy::Auto);
  problem.set_storage_policy(dtwc::core::StoragePolicy::Auto);
  problem.set_verbose(true);
  problem.set_output_folder(std::move(replacement_output_folder));
  problem.set_name(std::string{ "f19-encapsulation" });
  problem.set_data(std::move(replacement_data));
}
#endif

} // namespace
