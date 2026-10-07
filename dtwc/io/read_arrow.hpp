/**
 * @file read_arrow.hpp
 * @brief dtwc::io::read_arrow: Parquet and Arrow IPC, read with Arrow C++ (dtwc_io: a build with Arrow only).
 */

#pragma once

#include "read_data.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace dtwc::io {

/**
 * @brief The format of `path` this build's Arrow reader takes, judged from the file system alone, before any file is
 *        opened: Parquet (a .parquet/.pq file or a folder holding one, in a build with Parquet) or Arrow IPC (an
 *        .arrow/.ipc/.feather file). Empty for anything else, which dtwc::read_data reads or refuses.
 */
std::optional<InputFormat> arrow_format(const std::filesystem::path &path);

/**
 * @brief Every series a Parquet or Arrow IPC `path` names, laid out as read_data.hpp describes; any other path is
 *        dtwc::read_data's, with no text options.
 *
 * `column` selects a Parquet column (empty: the layout rule decides), `skip_cols` and `skip_rows` drop a Parquet file's
 * leading columns and rows; Arrow IPC takes none of them. A null is InvalidInput, an option the format cannot honour
 * InvalidInput, and a read failure an IOError naming the file.
 */
Data read_arrow(const std::filesystem::path &path, const std::string &column = {}, index_t skip_cols = 0,
                index_t skip_rows = 0);

} // namespace dtwc::io
