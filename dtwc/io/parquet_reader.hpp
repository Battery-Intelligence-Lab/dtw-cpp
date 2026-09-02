/// @file parquet_reader.hpp — Read time series from Parquet files.
///
/// Reads a single numeric column from a Parquet file, treating each row as one
/// sample. Multiple Parquet files (one per series) or a single file with a
/// List<Float64> column are both supported.
///
/// Requires DTWC_HAS_PARQUET (Apache Arrow + Parquet, Apache-2.0 license).
///
/// @author Volkan Kumtepeli
/// @author Claude 4.6
/// @date 08 Apr 2026

#pragma once

#ifdef DTWC_HAS_PARQUET

#include "../Data.hpp"
#include "../settings.hpp"
#include "parquet_schema.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::io {

namespace detail {

inline void check_arrow(const arrow::Status &s, const char *ctx)
{
  if (!s.ok())
    throw std::runtime_error(std::string(ctx) + ": " + s.ToString());
}

} // namespace detail


/// Load a single Parquet file as one time series per row (columnar layout)
/// or one series per cell (list-column layout).
///
/// @param path       Parquet file path.
/// @param col_name   Column to extract (empty = auto-detect first numeric).
/// @return Data with one series per row (or per list element).
inline Data load_parquet_file(const std::filesystem::path &path,
                              const std::string &col_name = "")
{
  // Open file via mmap for efficient large-file handling
  auto mmap_result = arrow::io::MemoryMappedFile::Open(path.string(), arrow::io::FileMode::READ);
  detail::check_arrow(mmap_result.status(), "load_parquet_file mmap");

  auto builder = parquet::arrow::FileReaderBuilder();
  detail::check_arrow(builder.Open(*mmap_result), "load_parquet_file Open");
  std::unique_ptr<parquet::arrow::FileReader> reader;
  detail::check_arrow(builder.Build(&reader), "load_parquet_file Build");

  std::shared_ptr<arrow::Schema> arrow_schema;
  detail::check_arrow(reader->GetSchema(&arrow_schema), "GetSchema");

  const auto selected = detail::find_parquet_series_column(
    arrow_schema, col_name);
  const int col_idx = selected.parquet_leaf_index;
  const auto col_type = selected.type;

  // Read just the selected column
  std::shared_ptr<arrow::Table> table;
  detail::check_arrow(reader->ReadTable({col_idx}, &table), "ReadTable");

  auto col = table->column(0);
  const int64_t N = table->num_rows();

  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;

  if (col_type->id() == arrow::Type::LIST || col_type->id() == arrow::Type::LARGE_LIST) {
    // List column: each cell is a variable-length series
    vecs.reserve(static_cast<size_t>(N));
    names.reserve(static_cast<size_t>(N));

    // Every list cell is one series. Value type, list offsets and null counts
    // are all validated by the shared helpers in parquet_schema.hpp, so the
    // eager and streaming readers cannot drift apart again (audit A3 / C).
    auto append_list_series = [&](auto list) {
      // Two null_count() reads per chunk, never one per element.
      detail::require_no_nulls(*list, "list column cell");
      detail::require_no_nulls(*list->values(), "list column value");
      const auto &values = *list->values();
      const int64_t len = list->length();
      for (int64_t i = 0; i < len; ++i) {
        const int64_t start = list->value_offset(i);
        const int64_t end = list->value_offset(i + 1);
        detail::require_list_range(start, end, values.length());
        const auto sz = static_cast<size_t>(end - start);
        std::vector<data_t> series(sz);
        detail::copy_arrow_numeric<data_t>(values, start, sz, series.data());
        vecs.push_back(std::move(series));
        names.push_back("series_" + std::to_string(vecs.size() - 1));
      }
    };

    for (int c = 0; c < col->num_chunks(); ++c) {
      auto chunk = col->chunk(c);
      if (col_type->id() == arrow::Type::LIST)
        append_list_series(std::static_pointer_cast<arrow::ListArray>(chunk));
      else
        append_list_series(std::static_pointer_cast<arrow::LargeListArray>(chunk));
    }
  } else {
    // Scalar column: the entire column is one series (one file = one series).
    std::vector<data_t> series;
    series.reserve(static_cast<size_t>(N));

    for (int c = 0; c < col->num_chunks(); ++c) {
      const auto chunk = col->chunk(c);
      detail::require_no_nulls(*chunk, "scalar column value");
      const auto n = static_cast<size_t>(chunk->length());
      const size_t offset = series.size();
      series.resize(offset + n);
      detail::copy_arrow_numeric<data_t>(*chunk, 0, n, series.data() + offset);
    }

    std::string name = path.stem().string();
    vecs.push_back(std::move(series));
    names.push_back(std::move(name));
  }

  return Data(std::move(vecs), std::move(names));
}


/// Load multiple Parquet files from a directory (one file = one series).
///
/// @param dir        Directory containing .parquet files.
/// @param col_name   Column to extract from each file.
/// @return Data with one series per file.
inline Data load_parquet_directory(const std::filesystem::path &dir,
                                   const std::string &col_name = "")
{
  namespace fs = std::filesystem;

  std::vector<fs::path> paths;
  for (const auto &entry : fs::directory_iterator(dir)) {
    auto ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".parquet" || ext == ".pq")
      paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());

  if (paths.empty())
    throw std::runtime_error("No .parquet files found in " + dir.string());

  std::vector<std::vector<data_t>> all_vecs;
  std::vector<std::string> all_names;

  for (const auto &p : paths) {
    auto d = load_parquet_file(p, col_name);
    for (size_t i = 0; i < d.size(); ++i) {
      all_vecs.push_back(std::move(d.p_vec[i]));
      all_names.push_back(std::move(d.p_names[i]));
    }
  }

  return Data(std::move(all_vecs), std::move(all_names));
}

} // namespace dtwc::io

#endif // DTWC_HAS_PARQUET
