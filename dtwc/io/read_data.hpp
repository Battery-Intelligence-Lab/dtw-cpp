/**
 * @file read_data.hpp
 * @brief dtwc::read_data: the one way from a path to Data, for dtwc_cl, dtwc::load and Python's load().
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

/// The readers behind read_data.
enum class InputFormat { Text, Parquet, ArrowIPC };

/**
 * @brief The reader `path` needs, judged from the file system alone, before any file is opened.
 *
 * A .parquet/.pq file, or a folder holding one, is Parquet; an .arrow/.ipc/.feather file is Arrow IPC; anything else is
 * CSV/TSV text, one file or a folder of them. A format whose reader this build lacks is an IOError naming the build
 * option: it is never read as text.
 */
InputFormat input_format(const std::filesystem::path &path);

/// A Parquet input's files in read order: the file itself, or a folder's .parquet/.pq files in the order the text
/// reader reads a folder (sorted, hidden files skipped).
std::vector<std::filesystem::path> parquet_files(const std::filesystem::path &path);

/// Refuse a reader option the input cannot honour, as InvalidInput: skip_cols, skip_rows and delimiter parse CSV/TSV
/// text, column selects a Parquet column. `format` is empty for series passed in memory, which no reader parses.
void require_reader_options(std::optional<InputFormat> format, index_t skip_cols, index_t skip_rows, char delimiter,
                            std::string_view column);

/**
 * @brief Read every series `path` names, with the reader input_format() picks.
 *
 * - Text: skip_cols leading fields and skip_rows leading lines are dropped; delimiter '\0' is inferred from the
 *   extension. A file holds one series per row, named by its 1-based row number; a folder one series per file, named
 *   by its stem.
 * - Parquet: `column` (empty: the first Float32/Float64 or list column). A scalar column is one series named by its
 *   file's stem; a list column is one series per row, named series_<index> and numbered across a folder's files.
 * - Arrow IPC: the series are the rows of the `data` column across every record batch, named by a Utf8/LargeUtf8
 *   `name` column (else series_<index>); the schema metadata `ndim` gives the features per time step.
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

} // namespace detail

} // namespace dtwc
