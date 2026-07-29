#include <dtwc.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace {

using problem_set_int_t = void (dtwc::Problem::*)(int);
using problem_void_t = void (dtwc::Problem::*)();
using problem_const_void_t = void (dtwc::Problem::*)() const;
using problem_read_matrix_t =
  void (dtwc::Problem::*)(const std::filesystem::path &);
using problem_write_matrix_t =
  void (dtwc::Problem::*)(const std::string &) const;
using problem_const_double_t = dtwc::data_t (dtwc::Problem::*)() const;
using problem_index_distance_t =
  dtwc::data_t (dtwc::Problem::*)(int, int);
using problem_const_bool_t = bool (dtwc::Problem::*)() const;
using problem_medoid_writer_t =
  void (dtwc::Problem::*)(int, int) const;
using problem_double_t = double (dtwc::Problem::*)();
using problem_int_read_t = int (dtwc::Problem::*)() const;

using problem_score_t = double (*)(dtwc::Problem &);
using label_score_t = double (*)(
  const std::vector<int> &, const std::vector<int> &);
using loader_setter_t = dtwc::DataLoader &(dtwc::DataLoader::*)(int);
using path_setter_t =
  void (*)(const dtwc::settings::fs::path &);
using cstring_path_setter_t = void (*)(const char *);

// Canonical counterparts for all 31 retained function overload entities.
[[maybe_unused]] constexpr problem_set_int_t canonical_01 =
  static_cast<problem_set_int_t>(&dtwc::Problem::set_n_clusters);
[[maybe_unused]] constexpr problem_void_t canonical_02 =
  static_cast<problem_void_t>(&dtwc::Problem::refresh_distance_matrix);
[[maybe_unused]] constexpr problem_read_matrix_t canonical_03 =
  static_cast<problem_read_matrix_t>(&dtwc::Problem::read_distance_matrix);
[[maybe_unused]] constexpr problem_const_double_t canonical_04 =
  static_cast<problem_const_double_t>(&dtwc::Problem::max_distance);
[[maybe_unused]] constexpr problem_index_distance_t canonical_05 =
  static_cast<problem_index_distance_t>(&dtwc::Problem::dist_by_ind);
[[maybe_unused]] constexpr problem_const_bool_t canonical_06 =
  static_cast<problem_const_bool_t>(
    &dtwc::Problem::is_distance_matrix_filled);
[[maybe_unused]] constexpr problem_void_t canonical_07 =
  static_cast<problem_void_t>(&dtwc::Problem::fill_distance_matrix);
[[maybe_unused]] constexpr problem_const_void_t canonical_08 =
  static_cast<problem_const_void_t>(
    &dtwc::Problem::print_distance_matrix);
[[maybe_unused]] constexpr problem_write_matrix_t canonical_09 =
  static_cast<problem_write_matrix_t>(
    &dtwc::Problem::write_distance_matrix);
[[maybe_unused]] constexpr problem_const_void_t canonical_10 =
  static_cast<problem_const_void_t>(
    &dtwc::Problem::write_distance_matrix);
[[maybe_unused]] constexpr problem_const_void_t canonical_11 =
  static_cast<problem_const_void_t>(&dtwc::Problem::print_clusters);
[[maybe_unused]] constexpr problem_void_t canonical_12 =
  static_cast<problem_void_t>(&dtwc::Problem::write_clusters);
[[maybe_unused]] constexpr problem_medoid_writer_t canonical_13 =
  static_cast<problem_medoid_writer_t>(
    &dtwc::Problem::write_medoid_members);
[[maybe_unused]] constexpr problem_void_t canonical_14 =
  static_cast<problem_void_t>(&dtwc::Problem::write_silhouettes);
[[maybe_unused]] constexpr problem_double_t canonical_15 =
  static_cast<problem_double_t>(&dtwc::Problem::find_total_cost);
[[maybe_unused]] constexpr problem_void_t canonical_16 =
  static_cast<problem_void_t>(&dtwc::Problem::assign_clusters);
[[maybe_unused]] constexpr problem_void_t canonical_17 =
  static_cast<problem_void_t>(&dtwc::Problem::calculate_medoids);
[[maybe_unused]] constexpr problem_void_t canonical_18 =
  static_cast<problem_void_t>(&dtwc::Problem::cluster_by_mip);
[[maybe_unused]] constexpr problem_void_t canonical_19 =
  static_cast<problem_void_t>(
    &dtwc::Problem::cluster_by_kmedoids_lloyd);
[[maybe_unused]] constexpr problem_int_read_t canonical_20 =
  static_cast<problem_int_read_t>(&dtwc::Problem::n_clusters);

[[maybe_unused]] constexpr problem_score_t canonical_21 =
  static_cast<problem_score_t>(&dtwc::scores::davies_bouldin);
[[maybe_unused]] constexpr problem_score_t canonical_22 =
  static_cast<problem_score_t>(&dtwc::scores::dunn);
[[maybe_unused]] constexpr problem_score_t canonical_23 =
  static_cast<problem_score_t>(&dtwc::scores::calinski_harabasz);
[[maybe_unused]] constexpr label_score_t canonical_24 =
  static_cast<label_score_t>(&dtwc::scores::adjusted_rand);
[[maybe_unused]] constexpr label_score_t canonical_25 =
  static_cast<label_score_t>(&dtwc::scores::normalized_mutual_info);

[[maybe_unused]] constexpr loader_setter_t canonical_26 =
  static_cast<loader_setter_t>(&dtwc::DataLoader::start_column);
[[maybe_unused]] constexpr loader_setter_t canonical_27 =
  static_cast<loader_setter_t>(&dtwc::DataLoader::start_row);
[[maybe_unused]] constexpr path_setter_t canonical_28 =
  static_cast<path_setter_t>(&dtwc::settings::paths::set_data_path);
[[maybe_unused]] constexpr cstring_path_setter_t canonical_29 =
  static_cast<cstring_path_setter_t>(
    &dtwc::settings::paths::set_data_path);
[[maybe_unused]] constexpr path_setter_t canonical_30 =
  static_cast<path_setter_t>(
    &dtwc::settings::paths::set_results_path);
[[maybe_unused]] constexpr cstring_path_setter_t canonical_31 =
  static_cast<cstring_path_setter_t>(
    &dtwc::settings::paths::set_results_path);

// Both field entities map to a canonical write/read pair. Naming all four
// functions here proves the canonical field routes remain warning-silent.
[[maybe_unused]] constexpr problem_set_int_t canonical_32_write =
  static_cast<problem_set_int_t>(&dtwc::Problem::set_max_iter);
[[maybe_unused]] constexpr problem_int_read_t canonical_32_read =
  static_cast<problem_int_read_t>(&dtwc::Problem::max_iter);
[[maybe_unused]] constexpr problem_set_int_t canonical_33_write =
  static_cast<problem_set_int_t>(&dtwc::Problem::set_n_repetitions);
[[maybe_unused]] constexpr problem_int_read_t canonical_33_read =
  static_cast<problem_int_read_t>(&dtwc::Problem::n_repetitions);

} // namespace
