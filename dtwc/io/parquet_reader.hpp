/// @file parquet_reader.hpp — Read time series from Parquet files.
///
/// Reads a single numeric column from a Parquet file, treating each row as one
/// sample. Multiple Parquet files (one per series) or a single file with a
/// List<Float64> column are both supported.
///
/// Requires DTWC_HAS_PARQUET (Apache Arrow + Parquet, Apache-2.0 license).
///
/// @author Volkan Kumtepeli
/// @author Claude (generated)
/// @date 08 Apr 2026

#pragma once

#ifdef DTWC_HAS_PARQUET

#include "../Data.hpp"
#include "../settings.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <span>
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

/// Find the first numeric (float/double) column, or a named column.
inline int find_column(const std::shared_ptr<arrow::Schema> &schema,
                       const std::string &col_name)
{
  if (!col_name.empty()) {
    int idx = schema->GetFieldIndex(col_name);
    if (idx < 0)
      throw std::runtime_error("Column '" + col_name + "' not found in Parquet schema");
    return idx;
  }
  // Auto-detect first float64 or float32 column
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto tid = schema->field(i)->type()->id();
    if (tid == arrow::Type::DOUBLE || tid == arrow::Type::FLOAT)
      return i;
  }
  throw std::runtime_error("No numeric column found in Parquet schema. Use --column to specify.");
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

  int col_idx = detail::find_column(arrow_schema, col_name);
  auto col_type = arrow_schema->field(col_idx)->type();

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

    for (int c = 0; c < col->num_chunks(); ++c) {
      auto chunk = col->chunk(c);
      int64_t len = chunk->length();

      if (col_type->id() == arrow::Type::LIST) {
        auto list = std::static_pointer_cast<arrow::ListArray>(chunk);
        auto values = std::static_pointer_cast<arrow::DoubleArray>(list->values());
        for (int64_t i = 0; i < len; ++i) {
          auto start = list->value_offset(i);
          auto end = list->value_offset(i + 1);
          auto sz = static_cast<size_t>(end - start);
          vecs.emplace_back(values->raw_values() + start, values->raw_values() + start + sz);
          names.push_back("series_" + std::to_string(vecs.size() - 1));
        }
      } else {
        auto list = std::static_pointer_cast<arrow::LargeListArray>(chunk);
        auto values = std::static_pointer_cast<arrow::DoubleArray>(list->values());
        for (int64_t i = 0; i < len; ++i) {
          auto start = list->value_offset(i);
          auto end = list->value_offset(i + 1);
          auto sz = static_cast<size_t>(end - start);
          vecs.emplace_back(values->raw_values() + start, values->raw_values() + start + sz);
          names.push_back("series_" + std::to_string(vecs.size() - 1));
        }
      }
    }
  } else {
    // Scalar column: entire column is one series (one file = one series)
    std::vector<data_t> series;
    series.reserve(static_cast<size_t>(N));

    for (int c = 0; c < col->num_chunks(); ++c) {
      auto chunk = std::static_pointer_cast<arrow::DoubleArray>(col->chunk(c));
      const double *raw = chunk->raw_values();
      for (int64_t i = 0; i < chunk->length(); ++i)
        series.push_back(raw[i]);
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
