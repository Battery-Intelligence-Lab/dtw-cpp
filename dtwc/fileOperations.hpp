/**
 * @file fileOperations.hpp
 * @brief Functions for file operations
 *
 * @details This header file declares various functions for performing file operations such as
 * reading and writing data to/from files. It includes functions to handle comma-separated values (CSV) files,
 * read data into vectors or Armadillo matrices, and save matrices to files.
 * It provides the functionality to ignore Byte Order Marks (BOM) in text files,
 * read specific rows and columns from files, and handle data from directories or batch files.
 *
 * @date 21 Jan 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include "settings.hpp" // for resultsPath

#include <cassert>    // for assert
#include <charconv>   // for from_chars
#include <chrono>     // for filesystem
#include <cctype>     // for isspace, tolower
#include <cmath>      // for isfinite
#include <cstdlib>    // for size_t
#include <filesystem> // for operator<<, path, operator/, directory_iterator
#include <iostream>   // for operator<<, ifstream, basic_ostream, operator>>
#include <limits>
#include <optional>
#include <string>     // for string, getline, to_string
#include <utility>    // for pair
#include <vector>     // for vector
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept> // for std::runtime_error
#include <string_view>
#include <type_traits>

#include <rapidcsv.h>

namespace dtwc {

namespace fs = std::filesystem;

/**
 * @brief Ignores Byte Order Mark (BOM) in UTF-8 encoded files.
 *
 * @param in Reference to the input stream to process.
 */
inline void ignoreBOM(std::istream &in)
{
  const auto start = in.tellg();
  const char bom[] = { '\xEF', '\xBB', '\xBF' };
  char buf[3]{};
  if (in.read(buf, 3) && buf[0] == bom[0] && buf[1] == bom[1] && buf[2] == bom[2])
    return; // BOM consumed
  // No BOM (or partial match) — rewind to start
  in.clear();
  in.seekg(start);
}

namespace text_io_detail {

inline std::string_view trim_ascii(std::string_view token)
{
  while (!token.empty()
         && std::isspace(static_cast<unsigned char>(token.front())) != 0)
    token.remove_prefix(1);
  while (!token.empty()
         && std::isspace(static_cast<unsigned char>(token.back())) != 0)
    token.remove_suffix(1);
  return token;
}

inline std::vector<std::string_view> split_fields(std::string_view line,
                                                  char delimiter)
{
  std::vector<std::string_view> fields;
  if (delimiter == ' ') {
    std::size_t pos = 0;
    while (pos < line.size()) {
      while (pos < line.size()
             && std::isspace(static_cast<unsigned char>(line[pos])) != 0)
        ++pos;
      const std::size_t start = pos;
      while (pos < line.size()
             && std::isspace(static_cast<unsigned char>(line[pos])) == 0)
        ++pos;
      if (start != pos) fields.emplace_back(line.substr(start, pos - start));
    }
    return fields;
  }

  std::size_t start = 0;
  while (true) {
    const std::size_t end = line.find(delimiter, start);
    if (end == std::string_view::npos) {
      fields.emplace_back(line.substr(start));
      break;
    }
    fields.emplace_back(line.substr(start, end - start));
    start = end + 1;
  }
  return fields;
}

inline std::string lower_ascii(std::string_view token)
{
  std::string result;
  result.reserve(token.size());
  for (const char c : token)
    result.push_back(static_cast<char>(
      std::tolower(static_cast<unsigned char>(c))));
  return result;
}

[[noreturn]] inline void throw_numeric_field_error(
  const fs::path &path, std::size_t row, std::size_t column,
  std::string_view reason, std::string_view token)
{
  constexpr std::size_t max_token_chars = 64;
  std::string shown(token.substr(0, max_token_chars));
  if (token.size() > max_token_chars) shown += "...";
  throw std::runtime_error(
    "Error in delimited text file: '" + path.string() + "' row "
    + std::to_string(row) + ", column " + std::to_string(column)
    + ": " + std::string(reason) + " '" + shown + "'.");
}

template <typename T>
T parse_numeric_field(std::string_view raw_token, const fs::path &path,
                      std::size_t row, std::size_t column)
{
  const std::string_view token = trim_ascii(raw_token);
  if (token.empty())
    throw_numeric_field_error(path, row, column, "empty numeric field", token);

  const std::string lower = lower_ascii(token);
  if (lower == "nan") {
    if constexpr (std::is_floating_point_v<T>)
      return std::numeric_limits<T>::quiet_NaN();
    else
      throw_numeric_field_error(path, row, column,
                                "NaN is invalid for an integer field", token);
  }

  std::string_view parsed = token;
  if (parsed.size() > 1 && parsed.front() == '+') parsed.remove_prefix(1);
  T value{};
  std::from_chars_result result;
  if constexpr (std::is_floating_point_v<T>)
    result = std::from_chars(parsed.data(), parsed.data() + parsed.size(), value,
                             std::chars_format::general);
  else
    result = std::from_chars(parsed.data(), parsed.data() + parsed.size(), value);

  if (result.ec == std::errc::result_out_of_range)
    throw_numeric_field_error(path, row, column, "numeric field is out of range", token);
  if (result.ec != std::errc{} || result.ptr != parsed.data() + parsed.size())
    throw_numeric_field_error(path, row, column, "invalid numeric field", token);
  if constexpr (std::is_floating_point_v<T>) {
    if (!std::isfinite(value))
      throw_numeric_field_error(path, row, column,
                                "unapproved non-finite numeric field", token);
  }
  return value;
}

template <typename T, typename Consumer>
std::size_t parse_numeric_row(std::string_view line, const fs::path &path,
                              std::size_t row, int start_column,
                              char delimiter, Consumer &&consume)
{
  if (start_column < 0)
    throw std::runtime_error("Error in delimited text file: start_col must be non-negative.");
  // A physically empty line is the existing on-disk representation of an
  // empty series. Empty fields inside a delimited non-empty row remain errors.
  if (trim_ascii(line).empty()) return 0;
  const auto fields = split_fields(line, delimiter);
  const auto first = static_cast<std::size_t>(start_column);
  if (first > fields.size()) {
    throw std::runtime_error(
      "Error in delimited text file: '" + path.string() + "' row "
      + std::to_string(row) + " has only " + std::to_string(fields.size())
      + " fields, fewer than start_col=" + std::to_string(start_column) + ".");
  }

  std::size_t count = 0;
  for (std::size_t i = first; i < fields.size(); ++i) {
    consume(parse_numeric_field<T>(fields[i], path, row, i + 1));
    ++count;
  }
  return count;
}

template <typename T>
std::optional<T> parse_series_value_row(std::string_view line,
                                        const fs::path &path,
                                        std::size_t row, int start_column,
                                        char delimiter,
                                        bool allow_legacy_empty_header)
{
  if (start_column < 0)
    throw std::runtime_error("Error in delimited text file: start_col must be non-negative.");
  if (trim_ascii(line).empty()) return std::nullopt;
  const auto fields = split_fields(line, delimiter);
  const auto column = static_cast<std::size_t>(start_column);
  if (column >= fields.size()) {
    throw std::runtime_error(
      "Error in delimited text file: '" + path.string() + "' row "
      + std::to_string(row) + " has only " + std::to_string(fields.size())
      + " fields, fewer than required column " + std::to_string(column + 1)
      + ".");
  }
  // Preserve the repository's historical `,0` directory-file header only.
  // Textual first-row values are not guessed to be headers: callers with a
  // named header must opt in explicitly via start_row=1, so malformed data
  // cannot disappear merely because it occurs on the first row.
  if (allow_legacy_empty_header && trim_ascii(fields[column]).empty())
    return std::nullopt;
  return parse_numeric_field<T>(fields[column], path, row, column + 1);
}

} // namespace text_io_detail

/**
 * @brief Reads a file and returns the data as a vector of a specified type.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param name Path of the file to read.
 * @param start_row Starting row index for reading the data (default is 0).
 * @param start_col Starting column index for reading the data (default is 0).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::vector<data_t> A vector containing the read data.
 */
template <typename data_t>
auto readFile(const fs::path &name, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::ifstream in(name, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    throw std::runtime_error("Error in readFile: File " + name.string() + " could not be opened.");
  }

  ignoreBOM(in);

  std::string line{};

  for (int i = 0; i < start_row; i++) // Skip first start_row rows to start from start_row.
    std::getline(in, line);

  std::vector<data_t> p;
  p.reserve(10000);
  std::size_t row = static_cast<std::size_t>(start_row);
  bool first_data_line = true;
  while (std::getline(in, line)) {
    ++row;
    const auto value = text_io_detail::parse_series_value_row<data_t>(
      line, name, row, start_col, delimiter, first_data_line);
    first_data_line = false;
    if (value) p.push_back(*value);
  }

  p.shrink_to_fit();
  return p;
}

/**
 * @brief Options for load_folder / load_batch_file.
 *
 * @details Bundles the five auxiliary parameters (Ndata, verbose, start_row,
 * start_col, delimiter) that previously trailed the path argument as positional
 * args. C++20 designated initialisers make call sites self-documenting:
 *
 *   load_folder<double>(path, {.Ndata = 1000, .start_row = 1});
 *
 * The positional-arg overloads are retained for backwards compatibility; they
 * delegate to the struct-based form.
 */
struct LoadOptions {
  int Ndata = -1;       //!< Max number of series to read; -1 = all.
  int verbose = 1;      //!< Verbosity level for logging.
  int start_row = 0;    //!< First row to read (skip headers).
  int start_col = 0;    //!< First column to read (skip ID columns).
  char delimiter = ','; //!< Field delimiter character.
};

/**
 * @brief Loads all files from a given folder and returns their data as vectors along with file names.
 *
 * @tparam data_t The data type of the elements to be read.
 * @tparam Tpath Type of the folder path (auto-deduced).
 * @param folder_path Path of the folder containing the files.
 * @param Ndata Maximum number of data points to read from each file (default is -1, read all data).
 * @param verbose Verbosity level for logging output (default is 1).
 * @param start_row Starting row index for reading the data (default is 0).
 * @param start_col Starting column index for reading the data (default is 0).
 * @param delimiter Delimiter character used in the files (default is ',').
 * @return std::pair<std::vector<std::vector<data_t>>, std::vector<std::string>> A pair containing vectors of data and corresponding file names.
 */
template <typename data_t, typename Tpath>
auto load_folder(Tpath &folder_path, const LoadOptions &opts = {})
{
  std::cout << "Reading data:" << '\n';

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  int i_data = 0;
  for (const auto &entry : fs::directory_iterator(folder_path)) {

    auto p = readFile<data_t>(entry.path(), opts.start_row, opts.start_col, opts.delimiter);

    if (opts.verbose >= 2 || (opts.verbose == 1 && p.empty()))
      std::cout << entry.path() << "\tSize: " << p.size() << '\n';

    p_vec.push_back(std::move(p));
    p_names.push_back(entry.path().stem().string());

    i_data++;
    if (i_data == opts.Ndata) break;
  }

  std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}

/// Positional-arg overload retained for backwards compatibility; delegates to
/// the LoadOptions-based form.
template <typename data_t, typename Tpath>
auto load_folder(Tpath &folder_path, int Ndata, int verbose = 1,
                 int start_row = 0, int start_col = 0, char delimiter = ',')
{
  return load_folder<data_t>(folder_path,
    LoadOptions{Ndata, verbose, start_row, start_col, delimiter});
}

/**
 * @brief Loads batch data from a single file and returns the data as vectors.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param file_path Path of the file containing batch data.
 * @param Ndata Maximum number of data points to read (default is -1, read all data).
 * @param verbose Verbosity level for logging output (default is 1).
 * @param start_row Starting row index for reading the data (default is 0).
 * @param start_col Starting column index for reading the data (default is 0).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::pair<std::vector<std::vector<data_t>>, std::vectorstd::string> A pair containing vectors of data and corresponding identifiers.
 */
template <typename data_t>
auto load_batch_file(fs::path &file_path, const LoadOptions &opts = {})
{
  std::cout << "Reading data:" << '\n';

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  auto myAbsPath = fs::absolute(file_path);

  std::ifstream in(file_path, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    throw std::runtime_error("Error in load_batch_file: File " + file_path.string() + " could not be opened.");
  }

  ignoreBOM(in);

  std::string line;
  int line_no{ 0 };
  int n_rows{ 0 };
  while ((opts.Ndata == -1 || n_rows < opts.Ndata) && std::getline(in, line)) //!< Read file.
  {
    if (line_no++ < opts.start_row) // Skip first rows.
      continue;

    n_rows++;

    std::vector<data_t> p;
    text_io_detail::parse_numeric_row<data_t>(
      line, file_path, static_cast<std::size_t>(line_no), opts.start_col,
      opts.delimiter, [&](data_t value) { p.push_back(value); });

    p.shrink_to_fit();

    if (opts.verbose >= 2 || (opts.verbose == 1 && p.empty()))
      std::cout << file_path << '\t' << "data: " << n_rows << " Size: " << p.size() << '\n';

    p_vec.push_back(std::move(p));
    p_names.push_back(std::to_string(n_rows));
  }

  std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}

/// Positional-arg overload retained for backwards compatibility; delegates to
/// the LoadOptions-based form.
template <typename data_t>
auto load_batch_file(fs::path &file_path, int Ndata, int verbose = 1,
                     int start_row = 0, int start_col = 0, char delimiter = ',')
{
  return load_batch_file<data_t>(file_path,
    LoadOptions{Ndata, verbose, start_row, start_col, delimiter});
}

// ============================================================================
// RapidCSV-based functions for robust multi-column CSV parsing
// ============================================================================

/**
 * @brief Reads a multi-column CSV file and returns data as a 2D vector using rapidcsv.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param file_path Path of the CSV file to read.
 * @param has_header Whether the first row is a header (default is false).
 * @param has_row_names Whether the first column contains row names (default is false).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::vector<std::vector<data_t>> A 2D vector where each inner vector is a row.
 */
template <typename data_t>
auto readCSV(const fs::path &file_path, bool has_header = false, bool has_row_names = false, char delimiter = ',')
{
  if (!fs::exists(file_path)) {
    throw std::runtime_error("Error in readCSV: File " + file_path.string() + " does not exist.");
  }

  rapidcsv::LabelParams labels(
    has_header ? 0 : -1,    // Row header index (-1 = no header)
    has_row_names ? 0 : -1  // Column header index (-1 = no row names)
  );
  rapidcsv::SeparatorParams sep(delimiter);

  rapidcsv::Document doc(file_path.string(), labels, sep);

  std::vector<std::vector<data_t>> result;
  const size_t numRows = doc.GetRowCount();
  result.reserve(numRows);

  for (size_t i = 0; i < numRows; ++i) {
    result.push_back(doc.GetRow<data_t>(i));
  }

  return result;
}

/**
 * @brief Reads a multi-column CSV file where each row is a time series.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param file_path Path of the CSV file to read.
 * @param max_rows Maximum number of rows to read (-1 = all rows).
 * @param has_header Whether the first row is a header (default is false).
 * @param label_col Column index containing labels (-1 = no labels, use row numbers).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::pair<std::vector<std::vector<data_t>>, std::vector<std::string>> Data and names.
 */
template <typename data_t>
auto readTimeSeriesCSV(const fs::path &file_path, int max_rows = -1, bool has_header = false,
                       int label_col = -1, char delimiter = ',')
{
  if (!fs::exists(file_path)) {
    throw std::runtime_error("Error in readTimeSeriesCSV: File " + file_path.string() + " does not exist.");
  }

  rapidcsv::LabelParams labels(has_header ? 0 : -1, -1);
  rapidcsv::SeparatorParams sep(delimiter);

  rapidcsv::Document doc(file_path.string(), labels, sep);

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  const size_t numRows = doc.GetRowCount();
  const size_t rowsToRead = (max_rows < 0) ? numRows : std::min(static_cast<size_t>(max_rows), numRows);

  p_vec.reserve(rowsToRead);
  p_names.reserve(rowsToRead);

  for (size_t i = 0; i < rowsToRead; ++i) {
    auto row = doc.GetRow<std::string>(i);

    // Extract name from label column or use row number
    std::string name;
    if (label_col >= 0 && static_cast<size_t>(label_col) < row.size()) {
      name = row[label_col];
    } else {
      name = std::to_string(i + 1);
    }
    p_names.push_back(name);

    // Convert remaining columns to data
    std::vector<data_t> series;
    series.reserve(row.size());
    for (size_t j = 0; j < row.size(); ++j) {
      if (static_cast<int>(j) == label_col) continue; // Skip label column
      try {
        if constexpr (std::is_same_v<data_t, double>) {
          series.push_back(std::stod(row[j]));
        } else if constexpr (std::is_same_v<data_t, float>) {
          series.push_back(std::stof(row[j]));
        } else if constexpr (std::is_same_v<data_t, int>) {
          series.push_back(std::stoi(row[j]));
        } else {
          series.push_back(static_cast<data_t>(std::stod(row[j])));
        }
      } catch (const std::exception &) {
        // Skip non-numeric values
      }
    }
    p_vec.push_back(std::move(series));
  }

  return std::pair(std::move(p_vec), std::move(p_names));
}

/**
 * @brief Reads a single column from a CSV file.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param file_path Path of the CSV file to read.
 * @param column Column index to read (0-based).
 * @param has_header Whether the first row is a header (default is false).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::vector<data_t> A vector containing the column data.
 */
template <typename data_t>
auto readCSVColumn(const fs::path &file_path, size_t column, bool has_header = false, char delimiter = ',')
{
  if (!fs::exists(file_path)) {
    throw std::runtime_error("Error in readCSVColumn: File " + file_path.string() + " does not exist.");
  }

  rapidcsv::LabelParams labels(has_header ? 0 : -1, -1);
  rapidcsv::SeparatorParams sep(delimiter);

  rapidcsv::Document doc(file_path.string(), labels, sep);

  return doc.GetColumn<data_t>(column);
}

} // namespace dtwc
