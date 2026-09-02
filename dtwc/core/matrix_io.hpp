/**
 * @file matrix_io.hpp
 * @brief CSV I/O and Eigen expansion for DenseDistanceMatrix.
 *
 * @details Free functions in dtwc::io operating on DenseDistanceMatrix
 *          by (const) reference. Separated from distance_matrix.hpp to satisfy
 *          SRP: the core data structure carries no I/O knowledge.
 *
 *          CSV format: full N×N, comma-separated, locale-independent
 *          general binary64 at max_digits10. File output is binary and uses
 *          exactly one LF per row on every host.
 *          Uncomputed entries are written as an empty field (not "nan"),
 *          so the file is clean and easy to inspect in spreadsheet tools.
 *          An empty field on read is treated as uncomputed.
 *
 *          operator<< lives in dtwc::core (not dtwc::io) so that
 *          ADL resolves it for DenseDistanceMatrix arguments.
 *
 * @author Volkan Kumtepeli
 * @date 04 Apr 2026
 */

#pragma once

#include "../error.hpp"
#include "distance_matrix.hpp"
#include "mmap_distance_matrix.hpp"

#include <Eigen/Core>

#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace dtwc::core::detail {

template <class Matrix>
inline void preflight_distance_matrix_csv(const Matrix &matrix)
{
  constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
  constexpr std::uint64_t fraction_mask = UINT64_C(0x000fffffffffffff);
  const size_t n = matrix.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const auto bits = std::bit_cast<std::uint64_t>(matrix.get(i, j));
      if ((bits & exponent_mask) == exponent_mask
          && (bits & fraction_mask) == 0) {
        throw InvalidInput(
          "distance-matrix CSV: computed non-finite value at row "
          + std::to_string(i) + ", column " + std::to_string(j) + ".");
      }
    }
  }
}

inline std::string_view distance_matrix_csv_token(
  double value, std::array<char, 64> &buffer)
{
  constexpr std::uint64_t sign_mask = UINT64_C(0x8000000000000000);
  constexpr std::uint64_t magnitude_mask = UINT64_C(0x7fffffffffffffff);
  constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
  constexpr std::uint64_t fraction_mask = UINT64_C(0x000fffffffffffff);
  const auto bits = std::bit_cast<std::uint64_t>(value);
  if ((bits & exponent_mask) == exponent_mask
      && (bits & fraction_mask) != 0)
    return {};
  if ((bits & magnitude_mask) == 0)
    return (bits & sign_mask) == 0 ? std::string_view{"0"}
                                   : std::string_view{"-0"};

  const auto formatted = std::to_chars(
    buffer.data(), buffer.data() + buffer.size(), value,
    std::chars_format::general, std::numeric_limits<double>::max_digits10);
  if (formatted.ec != std::errc{})
    throw std::runtime_error("Cannot format distance-matrix CSV value.");
  return {buffer.data(),
          static_cast<size_t>(formatted.ptr - buffer.data())};
}

} // namespace dtwc::core::detail

namespace dtwc::io {

/// Write the full N×N matrix to a CSV file.
/// Uncomputed entries are written as an empty field.
inline void write_csv(const core::DenseDistanceMatrix &dm, const std::filesystem::path &path)
{
  core::detail::preflight_distance_matrix_csv(dm);
  std::ofstream file(
    path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.good())
    throw std::runtime_error("Cannot open file for writing: " + path.string());
  const size_t n = dm.size();
  std::array<char, 64> number{};
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (j > 0) file.put(',');
      const auto token =
        core::detail::distance_matrix_csv_token(dm.get(i, j), number);
      if (!token.empty())
        file.write(token.data(), static_cast<std::streamsize>(token.size()));
      // uncomputed → empty field
    }
    file.put('\n');
  }
  file.close();
  if (!file.good())
    throw std::runtime_error("Write error on file: " + path.string());
}

/// Read a full N×N CSV file into the matrix.
/// Empty fields are treated as uncomputed; numeric fields are set.
inline void read_csv(core::DenseDistanceMatrix &dm, const std::filesystem::path &path)
{
  std::ifstream file(path);
  if (!file.good())
    throw std::runtime_error("Cannot open file for reading: " + path.string());

  struct Cell { double value; bool valid; };
  std::vector<std::vector<Cell>> rows;

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<Cell> row;
    // std::from_chars, not std::stod: the writer emits locale-independent
    // binary64 (see the file header), and std::stod honours LC_NUMERIC, so a
    // de_DE locale read "1.5" back as 1. A partial parse is an error, not a
    // truncation.
    // Loop shape matches the previous std::getline(ss, cell, ',') exactly,
    // including dropping a trailing empty field after a final comma.
    std::string_view rest{ line };
    while (!rest.empty()) {
      const auto comma = rest.find(',');
      const std::string_view cell =
        comma == std::string_view::npos ? rest : rest.substr(0, comma);
      if (cell.empty()) {
        row.push_back({ 0.0, false }); // empty field → uncomputed
      } else {
        double value{};
        const auto parsed = std::from_chars(
          cell.data(), cell.data() + cell.size(), value,
          std::chars_format::general);
        if (parsed.ec != std::errc{} || parsed.ptr != cell.data() + cell.size())
          throw std::runtime_error(
            "Invalid numeric field '" + std::string(cell) + "' in "
            + path.string());
        row.push_back({ value, true });
      }
      if (comma == std::string_view::npos)
        rest = {};
      else
        rest.remove_prefix(comma + 1);
    }
    rows.push_back(std::move(row));
  }

  if (!rows.empty()) {
    const size_t N = rows.size();
    dm.resize(N);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N && j < rows[i].size(); ++j)
        if (rows[i][j].valid)
          dm.set(i, j, rows[i][j].value);
  }
}

/// Expand packed triangular storage to a full N×N Eigen matrix.
/// Useful for numpy/MATLAB export from the binding layer.
inline Eigen::MatrixXd to_full_matrix(const core::DenseDistanceMatrix &dm)
{
  const size_t n = dm.size();
  Eigen::MatrixXd full = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j <= i; ++j) {
      const double v = dm.get(i, j);
      full(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = v;
      full(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = v;
    }
  return full;
}

} // namespace dtwc::io

namespace dtwc::core {

namespace detail {

/// Emit the full N x N CSV. Assumes preflight_distance_matrix_csv() has
/// ALREADY run on @p dm: it is the only part of the write that can throw, and
/// the F14 contract requires it to run before the destination is truncated.
/// Callers that must preflight before opening the file (Problem_IO's mmap CSV
/// copy) call this directly rather than paying a second O(N^2) value scan.
template <class Matrix>
inline std::ostream &write_distance_matrix_csv_preflighted(std::ostream &os,
                                                           const Matrix &dm)
{
  const size_t n = dm.size();
  std::array<char, 64> number{};
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (j > 0) os.put(',');
      const auto token = distance_matrix_csv_token(dm.get(i, j), number);
      if (!token.empty())
        os.write(token.data(), static_cast<std::streamsize>(token.size()));
      // uncomputed -> empty field
    }
    os.put('\n');
  }
  return os;
}

} // namespace detail

/// Stream output: prints CSV format (same as io::write_csv) to any ostream.
/// Defined in dtwc::core so ADL resolves it for DenseDistanceMatrix arguments.
inline std::ostream &operator<<(std::ostream &os, const DenseDistanceMatrix &dm)
{
  detail::preflight_distance_matrix_csv(dm);
  return detail::write_distance_matrix_csv_preflighted(os, dm);
}

/// Stream output for MmapDistanceMatrix (same CSV format as DenseDistanceMatrix).
inline std::ostream &operator<<(std::ostream &os, const MmapDistanceMatrix &dm)
{
  detail::preflight_distance_matrix_csv(dm);
  return detail::write_distance_matrix_csv_preflighted(os, dm);
}

} // namespace dtwc::core
