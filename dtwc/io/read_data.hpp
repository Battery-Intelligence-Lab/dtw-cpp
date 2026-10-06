/**
 * @file read_data.hpp
 * @brief dtwc::read_data: the text reader, the one way from a CSV/TSV path to Data, for dtwc_cl, dtwc::load and
 *        Python's and MATLAB's load(). Parquet and Arrow IPC are dtwc::io::read_arrow's (io/read_arrow.hpp, dtwc_io).
 */

#pragma once

#include "../Data.hpp"
#include "../base/settings.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dtwc {

/// The input formats: read_data reads Text; dtwc::io::read_arrow (dtwc_io) Parquet and Arrow IPC.
enum class InputFormat { Text, Parquet, ArrowIPC };

/**
 * @brief The reader `path` needs, judged from the file system alone, before any file is opened.
 *
 * A .parquet/.pq file, or a folder holding one, is Parquet; an .arrow/.ipc/.feather file is Arrow IPC; anything else is
 * CSV/TSV text, one file or a folder of them. This reader has no Arrow: Parquet and Arrow IPC are an IOError naming
 * the build option, never read as text (dtwc_cl asks dtwc::io::arrow_format first in a build with Arrow).
 */
InputFormat input_format(const std::filesystem::path &path);

/// A Parquet input's files in read order: the file itself, or a folder's .parquet/.pq files in the order the text
/// reader reads a folder (sorted, hidden files skipped).
std::vector<std::filesystem::path> parquet_files(const std::filesystem::path &path);

/// Whether `path` is an Arrow IPC file: an .arrow/.ipc/.feather name that is not a folder.
bool is_arrow_ipc(const std::filesystem::path &path);

/// Refuse a reader option the input cannot honour, as InvalidInput: skip_cols, skip_rows and delimiter parse CSV/TSV
/// text, column selects a Parquet column. `format` is empty for series passed in memory, which no reader parses.
void require_reader_options(std::optional<InputFormat> format, index_t skip_cols, index_t skip_rows, char delimiter,
                            std::string_view column);

/**
 * @brief Read every series a CSV/TSV `path` names; Parquet and Arrow IPC are input_format()'s IOError.
 *
 * - Text: skip_cols leading fields and skip_rows leading lines are dropped; delimiter '\0' is inferred from the
 *   extension. A file holds one series per row, named by its 1-based row number; a folder one series per file, named
 *   by its stem.
 * - Parquet (dtwc::io::read_arrow): `column` (empty: the first Float32/Float64 or list column). A scalar column is one
 *   series named by its file's stem; a list column is one series per row, named series_<index> and numbered across a
 *   folder's files.
 * - Arrow IPC (dtwc::io::read_arrow): the series are the rows of the `data` column across every record batch, named by
 *   a Utf8/LargeUtf8 `name` column (else series_<index>); the schema metadata `ndim` gives the features per time step.
 *
 * A null is InvalidInput, an option the format cannot honour InvalidInput, and a read failure an IOError naming the file.
 */
Data read_data(const std::filesystem::path &path, index_t skip_cols = 0, index_t skip_rows = 0, char delimiter = '\0',
               const std::string &column = {});

namespace detail {

/// The name of a run, or of a dataset, that was not given one: the input's file
/// name without its extension, or its folder's name; "dataset" for series in
/// memory (an empty path) and for a path that names neither.
std::string default_name(const std::filesystem::path &input);

/// Called in a reader's catch (...): rethrows the exception being handled, an
/// IOError or a std::exception that is no dtwc::Error as an IOError naming
/// `path` ("load: failed to read '<path>': ..."), so every reader says it alike.
[[noreturn]] void rethrow_naming_the_file(const std::filesystem::path &path);

} // namespace detail

} // namespace dtwc
