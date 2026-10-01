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

#include "base/error.hpp"
#include "base/settings.hpp" // for resultsPath
#include "io/parse_number.hpp" // exact, locale-free floating-point parsing

#include <algorithm>  // for std::sort
#include <charconv>   // for from_chars (integers)
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
#include <system_error> // for std::system_error (utf8_to_path fallback)
#include <string_view>
#include <type_traits>

namespace dtwc {

namespace fs = std::filesystem;

/**
 * @brief The UTF-8 bytes of a path, for identifiers that cross a language boundary.
 *
 * @details `std::filesystem::path::string()` returns the NATIVE narrow encoding,
 * which on Windows is the active ANSI code page: a file named `cafe\u00e9.csv`
 * yields the byte 0xE9, which the Python binding cannot decode as UTF-8 and the
 * CLI writes verbatim into its CSVs. Series and dataset names are carried as
 * UTF-8 on every platform so C++, Python and MATLAB see identical bytes.
 */
inline std::string path_to_utf8(const fs::path &p)
{
  const auto encoded = p.u8string();
  return std::string(reinterpret_cast<const char *>(encoded.data()), encoded.size());
}

/**
 * @brief The inverse of path_to_utf8: a UTF-8 identifier as a path component.
 *
 * @details `fs::path` built from a plain `std::string` re-decodes it in the
 * native narrow encoding, so a UTF-8 name would reach the filesystem as mojibake
 * on Windows. Going through `std::u8string` keeps the round trip
 * `path_to_utf8(utf8_to_path(s)) == s`. On POSIX the bytes pass through.
 */
inline fs::path utf8_to_path(std::string_view name)
{
  try {
    return fs::path(std::u8string(reinterpret_cast<const char8_t *>(name.data()),
                                  name.size()));
  } catch (const std::system_error &) {
    // Not valid UTF-8. The only producer of such a name is a native-encoded
    // string that never went through a loader (a Windows ANSI `--name` from
    // argv), so interpret it natively rather than failing the write: MSVC's
    // char8_t conversion THROWS on an unmappable sequence.
    return fs::path(std::string(name));
  }
}

/**
 * @brief Open `path` for writing, creating its parent directory first.
 *
 * @details An unchecked ofstream silently produces no file when the directory is
 * missing or unwritable, so a failed open is an IOError naming the file. Pair
 * every call with close_output().
 */
inline std::ofstream open_output(const fs::path &path, std::ios::openmode mode = std::ios::out)
{
  const auto directory = path.parent_path();
  if (!directory.empty()) {
    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec && !fs::is_directory(directory))
      throw IOError("Cannot create the output directory '" + path_to_utf8(directory) + "': " + ec.message());
  }
  std::ofstream file(path, mode);
  if (!file.is_open())
    throw IOError("Cannot open '" + path_to_utf8(path) + "' for writing; check that its directory is writable.");
  return file;
}

/// Close an output file and check it again: a full disk or a file-size quota
/// fails the writes after a successful open, which an open-only check reports
/// as success over a truncated file.
inline void close_output(std::ofstream &file, const fs::path &path)
{
  file.close();
  if (!file)
    throw IOError("Write error on '" + path_to_utf8(path)
                  + "': the file is incomplete (disk full or file-size quota?). Free space or choose another "
                    "directory, then rerun.");
}

/**
 * @brief Ignores Byte Order Mark (BOM) in UTF-8 encoded files.
 *
 * @param in Reference to the input stream to process.
 * @param path The file `in` reads; the error names it.
 */
inline void ignoreBOM(std::istream &in, const fs::path &path = {})
{
  // peek/get/unget only, never tellg/seekg: a pipe or FIFO cannot seek, and the
  // failed seekg left the stream failed, so a BOM-less pipe read as zero rows.
  constexpr unsigned char bom[] = { 0xEF, 0xBB, 0xBF };
  std::size_t matched = 0;
  while (matched < 3 && in.peek() == bom[matched]) {
    in.get();
    ++matched;
  }
  if (matched == 0 || matched == 3) return;
  // A partial match is the start of a character such as U+FF54 (EF BD 94),
  // e.g. in a header row: hand the bytes back.
  while (matched-- > 0) in.unget();
  if (!in)
    throw IOError(
      "Error in delimited text file: '" + path.string() + "': cannot re-read the "
      "first bytes of a non-seekable stream after a partial UTF-8 byte-order mark.");
}

/**
 * @brief One series-count contract for every loader.
 *
 * @details Contract, shared by count() and both loaders (which used to disagree
 * for `Ndata == 0`): `Ndata == -1` means "all", otherwise stop at exactly
 * `Ndata`. Anything below -1 is rejected.
 */
inline void validate_ndata(index_t Ndata, const char *ctx)
{
  if (Ndata < -1)
    throw InvalidInput(
      std::string(ctx) + ": Ndata must be -1 (read all) or a non-negative "
      "count; got " + std::to_string(Ndata));
}

/// True while another series may still be produced.
inline bool ndata_wants_more(index_t Ndata, std::size_t produced)
{
  return Ndata < 0 || produced < static_cast<std::size_t>(Ndata);
}

namespace text_io_detail {

/// ASCII whitespace. Not std::isspace: that reads LC_CTYPE, so under a UTF-8
/// locale byte 0xA0 was a space and one file parsed differently per process.
constexpr bool is_ascii_space(char c) noexcept
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

/// ASCII lowercase; like is_ascii_space, independent of the C locale.
constexpr char to_lower_ascii(char c) noexcept
{
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline std::string_view trim_ascii(std::string_view token)
{
  while (!token.empty() && is_ascii_space(token.front())) token.remove_prefix(1);
  while (!token.empty() && is_ascii_space(token.back())) token.remove_suffix(1);
  return token;
}

inline std::vector<std::string_view> split_fields(std::string_view line,
                                                  char delimiter)
{
  std::vector<std::string_view> fields;
  if (delimiter == ' ') {
    std::size_t pos = 0;
    while (pos < line.size()) {
      while (pos < line.size() && is_ascii_space(line[pos])) ++pos;
      const std::size_t start = pos;
      while (pos < line.size() && !is_ascii_space(line[pos])) ++pos;
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
  for (const char c : token) result.push_back(to_lower_ascii(c));
  return result;
}

/// ASCII case-insensitive equality without allocating a lowered copy.
/// `lower` must already be lowercase; used on the per-field hot path.
inline bool equals_ascii_ci(std::string_view token, std::string_view lower)
{
  if (token.size() != lower.size()) return false;
  for (std::size_t i = 0; i < token.size(); ++i)
    if (to_lower_ascii(token[i]) != lower[i]) return false;
  return true;
}

[[noreturn]] inline void throw_numeric_field_error(
  const fs::path &path, std::size_t row, std::size_t column,
  std::string_view reason, std::string_view token)
{
  constexpr std::size_t max_token_chars = 64;
  std::string shown(token.substr(0, max_token_chars));
  if (token.size() > max_token_chars) shown += "...";
  throw IOError(
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

  if (equals_ascii_ci(token, "nan")) {
    if constexpr (std::is_floating_point_v<T>)
      return std::numeric_limits<T>::quiet_NaN();
    else
      throw_numeric_field_error(path, row, column,
                                "NaN is invalid for an integer field", token);
  }

  std::string_view parsed = token;
  // One leading '+' is accepted; "+-1" is not a number.
  if (parsed.size() > 1 && parsed.front() == '+' && parsed[1] != '-')
    parsed.remove_prefix(1);
  T value{};
  const char *const last = parsed.data() + parsed.size();
  std::from_chars_result result;
  if constexpr (std::is_floating_point_v<T>)
    result = io::parse_number(parsed.data(), last, value); // not std::from_chars: macOS 26+ only
  else
    result = std::from_chars(parsed.data(), last, value);

  if (result.ec == std::errc::result_out_of_range)
    throw_numeric_field_error(path, row, column, "numeric field is out of range", token);
  if (result.ec != std::errc{} || result.ptr != last)
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
                              std::size_t row, index_t start_column,
                              char delimiter, Consumer &&consume)
{
  if (start_column < 0)
    throw InvalidInput("Error in delimited text file: start_col must be non-negative.");
  // Blank lines never get here (for_each_data_line); an empty field is an error.
  const auto fields = split_fields(line, delimiter);
  const auto first = static_cast<std::size_t>(start_column);
  // Too wide a start_col is the request's mistake, as it is for an in-memory
  // source (contract §1.2), not a failed read: InvalidInput.
  if (first > fields.size()) {
    throw InvalidInput(
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
                                        std::size_t row, index_t start_column,
                                        char delimiter,
                                        bool allow_legacy_empty_header)
{
  if (start_column < 0)
    throw InvalidInput("Error in delimited text file: start_col must be non-negative.");
  const auto fields = split_fields(line, delimiter);
  const auto column = static_cast<std::size_t>(start_column);
  if (column >= fields.size()) { // too wide a start_col: InvalidInput, as above
    throw InvalidInput(
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
  // One value per line: the fields after it used to be dropped silently, so a
  // two-column `index,value` file read without start_col=1 clustered the index.
  if (fields.size() > column + 1) {
    throw IOError(
      "Error in delimited text file: '" + path.string() + "' row "
      + std::to_string(row) + " has " + std::to_string(fields.size())
      + " fields; a file in a one-series-per-file folder holds one value per "
        "line, here in column " + std::to_string(column + 1)
      + " (skip leading columns with --skip-cols / start_col).");
  }
  return parse_numeric_field<T>(fields[column], path, row, column + 1);
}

/// The line loop of every text reader. Skips `start_row` lines, then calls
/// `on_line(line, row)` (row 1-based, counting skipped lines) for each data
/// line until `Ndata` lines were produced (-1 = all); returns that count.
/// Blank lines after the last data line are ignored. A blank line followed by
/// data is an error: it used to become an empty series in a batch file and to
/// vanish, shifting every later value, in a folder file.
template <typename OnLine>
std::size_t for_each_data_line(std::istream &in, const fs::path &path,
                               index_t start_row, index_t Ndata, OnLine &&on_line)
{
  std::string line;
  std::size_t row = 0, produced = 0, blank_row = 0;
  while (ndata_wants_more(Ndata, produced) && std::getline(in, line)) {
    ++row;
    if (static_cast<index_t>(row) <= start_row) continue; // a header row
    if (trim_ascii(line).empty()) {
      if (blank_row == 0) blank_row = row;
      continue;
    }
    if (blank_row != 0)
      throw IOError(
        "Error in delimited text file: '" + path.string() + "' row "
        + std::to_string(blank_row) + " is empty; an empty line is neither a "
          "series nor a value (write a missing value as nan).");
    on_line(std::string_view(line), row);
    ++produced;
  }
  return produced;
}

/// Open a text file for reading, positioned after any UTF-8 byte-order mark.
inline std::ifstream open_text_file(const fs::path &path, std::string_view reader)
{
  std::ifstream in(path, std::ios_base::in);
  if (!in.good())
    throw IOError("Error in " + std::string(reader) + ": File "
                  + path.string() + " could not be opened.");
  ignoreBOM(in, path);
  return in;
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
auto readFile(const fs::path &name, index_t start_row = 0, index_t start_col = 0, char delimiter = ',')
{
  auto in = text_io_detail::open_text_file(name, "readFile");

  std::vector<data_t> p;
  p.reserve(10000);
  // The legacy `,0` header is a header only when no rows were skipped: after
  // start_row it is the first value, and an empty one is an error.
  bool legacy_header = start_row == 0;
  text_io_detail::for_each_data_line(in, name, start_row, -1,
    [&](std::string_view line, std::size_t row) {
      const auto value = text_io_detail::parse_series_value_row<data_t>(
        line, name, row, start_col, delimiter, legacy_header);
      legacy_header = false;
      if (value) p.push_back(*value);
    });

  p.shrink_to_fit();
  return p;
}

/**
 * @brief Directory entries in one deterministic, filesystem-independent order.
 *
 * @details fs::directory_iterator order is filesystem-defined, so without this
 * every name, label, medoid and distance-matrix index depends on the machine
 * that ran the job. Non-regular entries are dropped rather than passed to
 * readFile(), and so are dot-files (.DS_Store, .gitkeep, AppleDouble `._a.csv`):
 * an empty .gitkeep used to load as an empty series. One O(n log n) pass per
 * load, before any parallel work.
 */
inline std::vector<fs::path> sorted_directory_files(const fs::path &folder_path)
{
  std::vector<fs::path> files;
  for (const auto &entry : fs::directory_iterator(folder_path)) {
    const fs::path leaf = entry.path().filename(); // owned: filename() is a temporary
    if (entry.is_regular_file() && !leaf.empty() && leaf.native().front() != '.')
      files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  return files;
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
  index_t Ndata = -1;     //!< Max number of series to read; -1 = all.
  int verbose = 1;        //!< Verbosity level for logging.
  index_t start_row = 0;  //!< First row to read (skip headers).
  index_t start_col = 0;  //!< First column to read (skip ID columns).
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
  validate_ndata(opts.Ndata, "load_folder");
  if (opts.verbose > 0) std::cout << "Reading data:" << '\n';

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  for (const auto &file : sorted_directory_files(folder_path)) {
    if (!ndata_wants_more(opts.Ndata, p_vec.size())) break;

    auto p = readFile<data_t>(file, opts.start_row, opts.start_col, opts.delimiter);

    if (opts.verbose >= 2 || (opts.verbose == 1 && p.empty()))
      std::cout << file << "\tSize: " << p.size() << '\n';

    p_vec.push_back(std::move(p));
    p_names.push_back(path_to_utf8(file.stem()));
  }

  if (opts.verbose > 0)
    std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}

/// Positional-arg overload retained for backwards compatibility; delegates to
/// the LoadOptions-based form.
template <typename data_t, typename Tpath>
auto load_folder(Tpath &folder_path, index_t Ndata, int verbose = 1,
                 index_t start_row = 0, index_t start_col = 0, char delimiter = ',')
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
  validate_ndata(opts.Ndata, "load_batch_file");
  if (opts.verbose > 0) std::cout << "Reading data:" << '\n';

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  auto in = text_io_detail::open_text_file(file_path, "load_batch_file");
  text_io_detail::for_each_data_line(in, file_path, opts.start_row, opts.Ndata,
    [&](std::string_view line, std::size_t row) {
      std::vector<data_t> p;
      text_io_detail::parse_numeric_row<data_t>(
        line, file_path, row, opts.start_col, opts.delimiter,
        [&](data_t value) { p.push_back(value); });
      p.shrink_to_fit();

      const auto n_rows = p_vec.size() + 1;
      if (opts.verbose >= 2 || (opts.verbose == 1 && p.empty()))
        std::cout << file_path << '\t' << "data: " << n_rows << " Size: " << p.size() << '\n';

      p_vec.push_back(std::move(p));
      p_names.push_back(std::to_string(n_rows));
    });

  if (opts.verbose > 0)
    std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}

/// Positional-arg overload retained for backwards compatibility; delegates to
/// the LoadOptions-based form.
template <typename data_t>
auto load_batch_file(fs::path &file_path, index_t Ndata, int verbose = 1,
                     index_t start_row = 0, index_t start_col = 0, char delimiter = ',')
{
  return load_batch_file<data_t>(file_path,
    LoadOptions{Ndata, verbose, start_row, start_col, delimiter});
}

} // namespace dtwc
