/**
 * @file matrix_io.hpp
 * @brief CSV I/O and full-matrix expansion for DistanceMatrix.
 *
 * @details Free functions in dtwc::io operating on DistanceMatrix
 *          by (const) reference. Separated from distance_matrix.hpp to satisfy
 *          SRP: the core data structure carries no I/O knowledge.
 *
 *          CSV format: full N×N, comma-separated, locale-independent
 *          general binary64 at max_digits10. File output is binary and uses
 *          exactly one LF per row on every host.
 *          Uncomputed entries are written as an empty field (not "nan"),
 *          so the file is clean and easy to inspect in spreadsheet tools.
 *          An empty field on read is treated as uncomputed; a file that is
 *          not square and symmetric is rejected with InvalidInput.
 *
 *          operator<< lives in dtwc::core (not dtwc::io) so that
 *          ADL resolves it for DistanceMatrix arguments.
 *
 * @author Volkan Kumtepeli
 * @date 04 Apr 2026
 */

#pragma once

#include "../base/error.hpp"
#include "../fileOperations.hpp"  // open_output, close_output
#include "../io/parse_number.hpp" // exact, locale-free floating-point parsing
#include "distance_matrix.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
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

inline void preflight_distance_matrix_csv(const DistanceMatrix &matrix)
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
  if (formatted.ec != std::errc{}) // programming error: the 64-byte buffer fits any double
    throw std::logic_error("Cannot format distance-matrix CSV value.");
  return {buffer.data(),
          static_cast<size_t>(formatted.ptr - buffer.data())};
}


/// Emit the full N x N CSV. Assumes preflight_distance_matrix_csv() has
/// ALREADY run on @p dm: it is the only part of the write that can throw, and
/// a failed write must leave the destination untouched, so it runs before the
/// destination is truncated.
inline std::ostream &write_distance_matrix_csv_preflighted(std::ostream &os,
                                                           const DistanceMatrix &dm)
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

} // namespace dtwc::core::detail

namespace dtwc::io {

/// Write the full N×N matrix to a CSV file; an uncomputed entry is an empty
/// field. A non-finite distance is rejected before the file is truncated.
inline void write_csv(const core::DistanceMatrix &dm, const std::filesystem::path &path)
{
  core::detail::preflight_distance_matrix_csv(dm);
  auto file = open_output(path, std::ios::out | std::ios::binary | std::ios::trunc);
  core::detail::write_distance_matrix_csv_preflighted(file, dm);
  close_output(file, path);
}

/// Read a full N×N CSV file into the matrix.
/// Empty fields are treated as uncomputed; numeric fields are set. The file
/// must be square and symmetric, or InvalidInput names the row: it used to
/// read a 2×3 file as 2×2, let the later of two different values of a pair
/// win, and leave a short row's missing cells silently uncomputed. One cell of
/// a pair may be empty (an upper- or lower-triangle file).
inline void read_csv(core::DistanceMatrix &dm, const std::filesystem::path &path)
{
  // Binary, so every platform reads the same bytes: a text-mode stream on
  // Windows ended the file at a 0x1A byte and dropped the rows after it
  // silently. The CR of a CRLF line end is dropped below.
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.good())
    throw IOError("Cannot open file for reading: " + path_to_utf8(path));

  struct Cell { double value; bool valid; };
  std::vector<std::vector<Cell>> rows;

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<Cell> row;
    // Locale-independent like the writer (std::stod honours LC_NUMERIC, so a
    // de_DE locale read "1.5" back as 1); a partial parse is an error. Every
    // comma separates two fields, so the writer's trailing uncomputed cells
    // ("0,,") count as cells.
    const std::string_view text{ line };
    std::size_t start = 0;
    while (true) {
      const auto comma = text.find(',', start);
      const auto cell = text.substr(
        start, comma == std::string_view::npos ? comma : comma - start);
      if (cell.empty()) {
        row.push_back({ 0.0, false }); // empty field → uncomputed
      } else {
        double value{};
        const auto parsed =
          parse_number(cell.data(), cell.data() + cell.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != cell.data() + cell.size())
          throw IOError(
            "Invalid numeric field '" + std::string(cell) + "' in "
            + path_to_utf8(path));
        row.push_back({ value, !std::isnan(value) }); // "nan" → uncomputed, as stored
      }
      if (comma == std::string_view::npos) break;
      start = comma + 1;
    }
    rows.push_back(std::move(row));
  }

  const size_t N = rows.size();
  const auto where = [&](size_t i, size_t j) {
    return "row " + std::to_string(i + 1) + ", column " + std::to_string(j + 1);
  };
  for (size_t i = 0; i < N; ++i) {
    // One trailing comma after N fields is tolerated, as before.
    if (rows[i].size() == N + 1 && !rows[i].back().valid) rows[i].pop_back();
    if (rows[i].size() != N)
      throw InvalidInput(
        "distance-matrix CSV '" + path_to_utf8(path) + "': row " + std::to_string(i + 1)
        + " has " + std::to_string(rows[i].size()) + " fields but the file has "
        + std::to_string(N) + " rows; a distance matrix is square (write an "
          "uncomputed entry as an empty field).");
  }
  std::array<char, 64> a_text{}, b_text{};
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < i; ++j) {
      const Cell a = rows[i][j], b = rows[j][i];
      if (a.valid && b.valid && a.value != b.value)
        throw InvalidInput(
          "distance-matrix CSV '" + path_to_utf8(path) + "': " + where(i, j) + " is "
          + std::string(core::detail::distance_matrix_csv_token(a.value, a_text))
          + " but " + where(j, i) + " is "
          + std::string(core::detail::distance_matrix_csv_token(b.value, b_text))
          + "; a distance matrix is symmetric.");
    }

  if (N == 0) return;
  dm.resize(N);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j <= i; ++j) {
      const Cell &cell = rows[i][j].valid ? rows[i][j] : rows[j][i];
      if (cell.valid) dm.set(i, j, cell.value);
    }
}

/// Expand packed triangular storage to a full N×N matrix, in row-major order.
/// Useful for numpy/MATLAB export from the binding layer.
///
/// Returned as a flat `std::vector<double>` rather than an Eigen matrix.
/// The matrix is symmetric, so row- and column-major layouts are byte-identical
/// and every caller already copied it straight into a `std::vector<double>` —
/// returning one removes an N×N copy instead of adding one.
inline std::vector<double> to_full_matrix(const core::DistanceMatrix &dm)
{
  const size_t n = dm.size();
  std::vector<double> full(n * n, 0.0);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j <= i; ++j) {
      const double v = dm.get(i, j);
      full[i * n + j] = v;
      full[j * n + i] = v;
    }
  return full;
}

} // namespace dtwc::io

namespace dtwc::core {

/// Stream output: prints CSV format (same as io::write_csv) to any ostream.
/// Defined in dtwc::core so ADL resolves it for DistanceMatrix arguments.
inline std::ostream &operator<<(std::ostream &os, const DistanceMatrix &dm)
{
  detail::preflight_distance_matrix_csv(dm);
  return detail::write_distance_matrix_csv_preflighted(os, dm);
}

} // namespace dtwc::core
