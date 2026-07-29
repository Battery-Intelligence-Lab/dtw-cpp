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
using problem_int_member_t = int dtwc::Problem::*;

using problem_score_t = double (*)(dtwc::Problem &);
using label_score_t = double (*)(
  const std::vector<int> &, const std::vector<int> &);
using loader_setter_t = dtwc::DataLoader &(dtwc::DataLoader::*)(int);
using path_setter_t =
  void (*)(const dtwc::settings::fs::path &);
using cstring_path_setter_t = void (*)(const char *);

// Problem function overloads: 20/20 retained legacy entities.
[[maybe_unused]] constexpr problem_set_int_t legacy_01 =
  static_cast<problem_set_int_t>(&dtwc::Problem::set_numberOfClusters);
[[maybe_unused]] constexpr problem_void_t legacy_02 =
  static_cast<problem_void_t>(&dtwc::Problem::refreshDistanceMatrix);
[[maybe_unused]] constexpr problem_read_matrix_t legacy_03 =
  static_cast<problem_read_matrix_t>(&dtwc::Problem::readDistanceMatrix);
[[maybe_unused]] constexpr problem_const_double_t legacy_04 =
  static_cast<problem_const_double_t>(&dtwc::Problem::maxDistance);
[[maybe_unused]] constexpr problem_index_distance_t legacy_05 =
  static_cast<problem_index_distance_t>(&dtwc::Problem::distByInd);
[[maybe_unused]] constexpr problem_const_bool_t legacy_06 =
  static_cast<problem_const_bool_t>(&dtwc::Problem::isDistanceMatrixFilled);
[[maybe_unused]] constexpr problem_void_t legacy_07 =
  static_cast<problem_void_t>(&dtwc::Problem::fillDistanceMatrix);
[[maybe_unused]] constexpr problem_const_void_t legacy_08 =
  static_cast<problem_const_void_t>(&dtwc::Problem::printDistanceMatrix);
[[maybe_unused]] constexpr problem_write_matrix_t legacy_09 =
  static_cast<problem_write_matrix_t>(&dtwc::Problem::writeDistanceMatrix);
[[maybe_unused]] constexpr problem_const_void_t legacy_10 =
  static_cast<problem_const_void_t>(&dtwc::Problem::writeDistanceMatrix);
[[maybe_unused]] constexpr problem_const_void_t legacy_11 =
  static_cast<problem_const_void_t>(&dtwc::Problem::printClusters);
[[maybe_unused]] constexpr problem_void_t legacy_12 =
  static_cast<problem_void_t>(&dtwc::Problem::writeClusters);
[[maybe_unused]] constexpr problem_medoid_writer_t legacy_13 =
  static_cast<problem_medoid_writer_t>(&dtwc::Problem::writeMedoidMembers);
[[maybe_unused]] constexpr problem_void_t legacy_14 =
  static_cast<problem_void_t>(&dtwc::Problem::writeSilhouettes);
[[maybe_unused]] constexpr problem_double_t legacy_15 =
  static_cast<problem_double_t>(&dtwc::Problem::findTotalCost);
[[maybe_unused]] constexpr problem_void_t legacy_16 =
  static_cast<problem_void_t>(&dtwc::Problem::assignClusters);
[[maybe_unused]] constexpr problem_void_t legacy_17 =
  static_cast<problem_void_t>(&dtwc::Problem::calculateMedoids);
[[maybe_unused]] constexpr problem_void_t legacy_18 =
  static_cast<problem_void_t>(&dtwc::Problem::cluster_by_MIP);
[[maybe_unused]] constexpr problem_void_t legacy_19 =
  static_cast<problem_void_t>(&dtwc::Problem::cluster_by_kMedoidsLloyd);
[[maybe_unused]] constexpr problem_int_read_t legacy_20 =
  static_cast<problem_int_read_t>(&dtwc::Problem::cluster_size);

// scores::* function aliases: 5/5.
[[maybe_unused]] constexpr problem_score_t legacy_21 =
  static_cast<problem_score_t>(&dtwc::scores::daviesBouldinIndex);
[[maybe_unused]] constexpr problem_score_t legacy_22 =
  static_cast<problem_score_t>(&dtwc::scores::dunnIndex);
[[maybe_unused]] constexpr problem_score_t legacy_23 =
  static_cast<problem_score_t>(&dtwc::scores::calinskiHarabaszIndex);
[[maybe_unused]] constexpr label_score_t legacy_24 =
  static_cast<label_score_t>(&dtwc::scores::adjustedRandIndex);
[[maybe_unused]] constexpr label_score_t legacy_25 =
  static_cast<label_score_t>(
    &dtwc::scores::normalizedMutualInformation);

// F21 loader/path aliases: 6/6 overload entities.
[[maybe_unused]] constexpr loader_setter_t legacy_26 =
  static_cast<loader_setter_t>(&dtwc::DataLoader::startColumn);
[[maybe_unused]] constexpr loader_setter_t legacy_27 =
  static_cast<loader_setter_t>(&dtwc::DataLoader::startRow);
[[maybe_unused]] constexpr path_setter_t legacy_28 =
  static_cast<path_setter_t>(&dtwc::settings::paths::setDataPath);
[[maybe_unused]] constexpr cstring_path_setter_t legacy_29 =
  static_cast<cstring_path_setter_t>(
    &dtwc::settings::paths::setDataPath);
[[maybe_unused]] constexpr path_setter_t legacy_30 =
  static_cast<path_setter_t>(
    &dtwc::settings::paths::setResultsPath);
[[maybe_unused]] constexpr cstring_path_setter_t legacy_31 =
  static_cast<cstring_path_setter_t>(
    &dtwc::settings::paths::setResultsPath);

// Retained public fields: 2/2 exact int data-member entities.
[[maybe_unused]] constexpr problem_int_member_t legacy_32 =
  &dtwc::Problem::maxIter;
[[maybe_unused]] constexpr problem_int_member_t legacy_33 =
  &dtwc::Problem::N_repetition;

} // namespace
