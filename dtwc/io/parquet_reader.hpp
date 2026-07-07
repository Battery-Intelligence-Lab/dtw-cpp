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

    // Append every list element as one series, converting Float32 -> data_t and
    // validating that offsets stay within the values buffer. find_column accepts
    // Float32 columns, so the values array may be Float32 or Float64. (audit
    // io-security: no Float64 check + no bounds on list offsets)
    auto append_list_series = [&](auto list) {
      auto values = list->values();
      const int64_t vlen = values->length();
      const bool is_double = values->type_id() == arrow::Type::DOUBLE;
      const bool is_float = values->type_id() == arrow::Type::FLOAT;
      if (!is_double && !is_float)
        throw std::runtime_error(
          "load_parquet_file: list value type must be Float64 or Float32, got " +
          values->type()->ToString());
      const double *draw = is_double
        ? std::static_pointer_cast<arrow::DoubleArray>(values)->raw_values() : nullptr;
      const float *fraw = is_float
        ? std::static_pointer_cast<arrow::FloatArray>(values)->raw_values() : nullptr;

      const int64_t len = list->length();
      for (int64_t i = 0; i < len; ++i) {
        const int64_t start = list->value_offset(i);
        const int64_t end = list->value_offset(i + 1);
        if (start < 0 || end < start || end > vlen)
          throw std::runtime_error(
            "load_parquet_file: list offset [" + std::to_string(start) + ", " +
            std::to_string(end) + ") out of bounds [0, " + std::to_string(vlen) + "]");
        const auto sz = static_cast<size_t>(end - start);
        std::vector<data_t> series(sz);
        if (is_double)
          std::copy_n(draw + start, sz, series.begin());
        else
          for (size_t j = 0; j < sz; ++j)
            series[j] = static_cast<data_t>(fraw[start + static_cast<int64_t>(j)]);
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
    // Scalar column: entire column is one series (one file = one series).
    // find_column accepts Float32 columns, so handle both value types rather than
    // blindly casting to DoubleArray. (audit io-security: reader path lacked the
    // FLOAT branch the chunk reader has -> latent garbage / OOB on f32 files)
    std::vector<data_t> series;
    series.reserve(static_cast<size_t>(N));

    for (int c = 0; c < col->num_chunks(); ++c) {
      auto arr = col->chunk(c);
      if (arr->type_id() == arrow::Type::DOUBLE) {
        auto dbl = std::static_pointer_cast<arrow::DoubleArray>(arr);
        const double *raw = dbl->raw_values();
        for (int64_t i = 0; i < dbl->length(); ++i)
          series.push_back(raw[i]);
      } else if (arr->type_id() == arrow::Type::FLOAT) {
        auto flt = std::static_pointer_cast<arrow::FloatArray>(arr);
        const float *raw = flt->raw_values();
        for (int64_t i = 0; i < flt->length(); ++i)
          series.push_back(static_cast<data_t>(raw[i]));
      } else {
        throw std::runtime_error(
          "load_parquet_file: scalar column type must be Float64 or Float32, got " +
          arr->type()->ToString());
      }
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
