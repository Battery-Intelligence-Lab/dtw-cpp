/// @file parquet_chunk_reader.hpp — Row-group streaming reader for Parquet files.
///
/// Provides chunked access to Parquet data for RAM-aware processing.
/// Each row group can be read independently, enabling streaming CLARA
/// assignment without loading the entire dataset into memory.
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
#include <parquet/file_reader.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::io {

namespace detail {

inline void check_arrow_chunk(const arrow::Status &s, const char *ctx)
{
  if (!s.ok())
    throw std::runtime_error(std::string(ctx) + ": " + s.ToString());
}

/// Find column index by name or auto-detect first numeric column.
inline int find_column_chunk(const std::shared_ptr<arrow::Schema> &schema,
                             const std::string &col_name)
{
  if (!col_name.empty()) {
    int idx = schema->GetFieldIndex(col_name);
    if (idx < 0)
      throw std::runtime_error("Column '" + col_name + "' not found in Parquet schema");
    return idx;
  }
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto tid = schema->field(i)->type()->id();
    if (tid == arrow::Type::DOUBLE || tid == arrow::Type::FLOAT ||
        tid == arrow::Type::LIST || tid == arrow::Type::LARGE_LIST)
      return i;
  }
  throw std::runtime_error("No numeric or list column found in Parquet schema. Use --column to specify.");
}

/// Extract values from a list array element into a data_t vector.
/// Handles both Float and Double value arrays.
template<typename ListArrayT>
inline void extract_list_element(const std::shared_ptr<ListArrayT> &list,
                                 int64_t i,
                                 std::vector<data_t> &out)
{
  auto start = list->value_offset(i);
  auto end = list->value_offset(i + 1);
  auto sz = static_cast<size_t>(end - start);

  auto values = list->values();
  if (values->type_id() == arrow::Type::DOUBLE) {
    auto dbl = std::static_pointer_cast<arrow::DoubleArray>(values);
    out.assign(dbl->raw_values() + start, dbl->raw_values() + start + sz);
  } else if (values->type_id() == arrow::Type::FLOAT) {
    auto flt = std::static_pointer_cast<arrow::FloatArray>(values);
    out.resize(sz);
    for (size_t j = 0; j < sz; ++j)
      out[j] = static_cast<data_t>(flt->raw_values()[start + static_cast<int64_t>(j)]);
  } else {
    throw std::runtime_error("Unsupported list value type: " + values->type()->ToString());
  }
}

/// Extract series from an Arrow table column into vectors.
/// Handles scalar (one file = one series), list, and large-list columns.
/// Supports both Float and Double value types.
inline void extract_series_from_column(
  const std::shared_ptr<arrow::ChunkedArray> &col,
  const std::shared_ptr<arrow::DataType> &col_type,
  std::vector<std::vector<data_t>> &vecs,
  std::vector<std::string> &names,
  int64_t name_offset = 0)
{
  if (col_type->id() == arrow::Type::LIST || col_type->id() == arrow::Type::LARGE_LIST) {
    for (int c = 0; c < col->num_chunks(); ++c) {
      auto chunk = col->chunk(c);
      int64_t len = chunk->length();

      if (col_type->id() == arrow::Type::LIST) {
        auto list = std::static_pointer_cast<arrow::ListArray>(chunk);
        for (int64_t i = 0; i < len; ++i) {
          std::vector<data_t> series;
          extract_list_element(list, i, series);
          vecs.push_back(std::move(series));
          names.push_back("series_" + std::to_string(name_offset + static_cast<int64_t>(vecs.size()) - 1));
        }
      } else {
        auto list = std::static_pointer_cast<arrow::LargeListArray>(chunk);
        for (int64_t i = 0; i < len; ++i) {
          std::vector<data_t> series;
          extract_list_element(list, i, series);
          vecs.push_back(std::move(series));
          names.push_back("series_" + std::to_string(name_offset + static_cast<int64_t>(vecs.size()) - 1));
        }
      }
    }
  } else {
    // Scalar column: entire column is one series (handles Float and Double)
    std::vector<data_t> series;
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
        throw std::runtime_error("Unsupported scalar column type: " + arr->type()->ToString());
      }
    }
    vecs.push_back(std::move(series));
    names.push_back("series_" + std::to_string(name_offset));
  }
}

/// Float32 variant: extract series as float vectors (no widening to double).
/// For Float Parquet columns, this avoids the 2x memory overhead of widening.
inline void extract_series_from_column_f32(
  const std::shared_ptr<arrow::ChunkedArray> &col,
  const std::shared_ptr<arrow::DataType> &col_type,
  std::vector<std::vector<float>> &vecs,
  std::vector<std::string> &names,
  int64_t name_offset = 0)
{
  if (col_type->id() == arrow::Type::LIST || col_type->id() == arrow::Type::LARGE_LIST) {
    for (int c = 0; c < col->num_chunks(); ++c) {
      auto chunk = col->chunk(c);
      int64_t len = chunk->length();

      auto extract_list = [&](auto list) {
        auto values = list->values();
        for (int64_t i = 0; i < len; ++i) {
          auto start = list->value_offset(i);
          auto end = list->value_offset(i + 1);
          auto sz = static_cast<size_t>(end - start);
          std::vector<float> series(sz);

          if (values->type_id() == arrow::Type::FLOAT) {
            auto flt = std::static_pointer_cast<arrow::FloatArray>(values);
            std::copy_n(flt->raw_values() + start, sz, series.begin());
          } else if (values->type_id() == arrow::Type::DOUBLE) {
            auto dbl = std::static_pointer_cast<arrow::DoubleArray>(values);
            for (size_t j = 0; j < sz; ++j)
              series[j] = static_cast<float>(dbl->raw_values()[start + static_cast<int64_t>(j)]);
          } else {
            throw std::runtime_error("Unsupported list value type: " + values->type()->ToString());
          }
          vecs.push_back(std::move(series));
          names.push_back("series_" + std::to_string(name_offset + static_cast<int64_t>(vecs.size()) - 1));
        }
      };

      if (col_type->id() == arrow::Type::LIST)
        extract_list(std::static_pointer_cast<arrow::ListArray>(chunk));
      else
        extract_list(std::static_pointer_cast<arrow::LargeListArray>(chunk));
    }
  } else {
    std::vector<float> series;
    for (int c = 0; c < col->num_chunks(); ++c) {
      auto arr = col->chunk(c);
      if (arr->type_id() == arrow::Type::FLOAT) {
        auto flt = std::static_pointer_cast<arrow::FloatArray>(arr);
        const float *raw = flt->raw_values();
        for (int64_t i = 0; i < flt->length(); ++i)
          series.push_back(raw[i]);
      } else if (arr->type_id() == arrow::Type::DOUBLE) {
        auto dbl = std::static_pointer_cast<arrow::DoubleArray>(arr);
        for (int64_t i = 0; i < dbl->length(); ++i)
          series.push_back(static_cast<float>(dbl->raw_values()[i]));
      } else {
        throw std::runtime_error("Unsupported scalar column type: " + arr->type()->ToString());
      }
    }
    vecs.push_back(std::move(series));
    names.push_back("series_" + std::to_string(name_offset));
  }
}

} // namespace detail


/// Row-group streaming reader for Parquet files.
///
/// Opens the file once (via mmap), reads metadata eagerly, and provides
/// methods to read individual row groups or sparse row subsets on demand.
///
/// @note NOT thread-safe. The underlying parquet::arrow::FileReader does not
///       support concurrent ReadRowGroups calls. Use one reader per thread or
///       serialize access externally.
class ParquetChunkReader
{
public:
  /// Open a Parquet file for chunked reading.
  ///
  /// @param path       Parquet file path.
  /// @param col_name   Column to extract (empty = auto-detect first numeric/list).
  explicit ParquetChunkReader(const std::filesystem::path &path,
                              const std::string &col_name = "")
  {
    auto mmap_result = arrow::io::MemoryMappedFile::Open(path.string(), arrow::io::FileMode::READ);
    detail::check_arrow_chunk(mmap_result.status(), "ParquetChunkReader mmap");
    file_ = *mmap_result;

    auto builder = parquet::arrow::FileReaderBuilder();
    detail::check_arrow_chunk(builder.Open(file_), "ParquetChunkReader Open");
    detail::check_arrow_chunk(builder.Build(&reader_), "ParquetChunkReader Build");

    detail::check_arrow_chunk(reader_->GetSchema(&schema_), "GetSchema");
    col_idx_ = detail::find_column_chunk(schema_, col_name);
    col_type_ = schema_->field(col_idx_)->type();

    // Cache row-group metadata
    auto *pq_meta = reader_->parquet_reader()->metadata().get();
    num_row_groups_ = pq_meta->num_row_groups();
    total_rows_ = pq_meta->num_rows();

    rg_row_counts_.resize(num_row_groups_);
    rg_row_offsets_.resize(num_row_groups_);
    int64_t offset = 0;
    for (int rg = 0; rg < num_row_groups_; ++rg) {
      rg_row_counts_[rg] = pq_meta->RowGroup(rg)->num_rows();
      rg_row_offsets_[rg] = offset;
      offset += rg_row_counts_[rg];
    }

    // Estimate bytes per series from file metadata
    if (total_rows_ > 0) {
      int64_t col_bytes = 0;
      for (int rg = 0; rg < num_row_groups_; ++rg)
        col_bytes += pq_meta->RowGroup(rg)->ColumnChunk(col_idx_)->total_uncompressed_size();
      estimated_bytes_per_series_ = static_cast<size_t>(col_bytes / total_rows_);
    }
  }

  /// Number of row groups in the file.
  int num_row_groups() const { return num_row_groups_; }

  /// Total number of rows across all row groups.
  int64_t total_rows() const { return total_rows_; }

  /// Number of rows in a specific row group.
  int64_t row_group_rows(int rg) const { return rg_row_counts_[rg]; }

  /// Estimated uncompressed bytes per series (from metadata, no data read).
  size_t estimated_bytes_per_series() const { return estimated_bytes_per_series_; }

  /// Total estimated data size in bytes.
  size_t estimated_total_bytes() const { return estimated_bytes_per_series_ * static_cast<size_t>(total_rows_); }

  /// Read a single row group into a Data object.
  ///
  /// @param rg  Row group index [0, num_row_groups()).
  /// @return Owning Data with series from that row group.
  Data read_row_group(int rg) const
  {
    return read_row_groups(rg, 1);
  }

  /// Read a contiguous batch of row groups [rg_start, rg_start+count).
  ///
  /// @param rg_start  First row group index.
  /// @param count     Number of row groups to read.
  /// @return Owning Data with all series from the batch.
  Data read_row_groups(int rg_start, int count) const
  {
    if (rg_start < 0 || rg_start + count > num_row_groups_)
      throw std::runtime_error("ParquetChunkReader::read_row_groups: range out of bounds");

    std::vector<int> rg_indices(count);
    std::iota(rg_indices.begin(), rg_indices.end(), rg_start);

    std::shared_ptr<arrow::Table> table;
    detail::check_arrow_chunk(
      reader_->ReadRowGroups(rg_indices, {col_idx_}, &table),
      "read_row_groups");

    auto col = table->column(0);
    std::vector<std::vector<data_t>> vecs;
    std::vector<std::string> names;
    vecs.reserve(static_cast<size_t>(table->num_rows()));
    names.reserve(static_cast<size_t>(table->num_rows()));

    detail::extract_series_from_column(col, col_type_, vecs, names, rg_row_offsets_[rg_start]);
    return Data(std::move(vecs), std::move(names));
  }

  /// Read a contiguous batch of row groups as float32 Data (2x memory saving).
  Data read_row_groups_f32(int rg_start, int count) const
  {
    if (rg_start < 0 || rg_start + count > num_row_groups_)
      throw std::runtime_error("ParquetChunkReader::read_row_groups_f32: range out of bounds");

    std::vector<int> rg_indices(count);
    std::iota(rg_indices.begin(), rg_indices.end(), rg_start);

    std::shared_ptr<arrow::Table> table;
    detail::check_arrow_chunk(
      reader_->ReadRowGroups(rg_indices, {col_idx_}, &table),
      "read_row_groups_f32");

    auto col = table->column(0);
    std::vector<std::vector<float>> vecs;
    std::vector<std::string> names;
    vecs.reserve(static_cast<size_t>(table->num_rows()));
    names.reserve(static_cast<size_t>(table->num_rows()));

    detail::extract_series_from_column_f32(col, col_type_, vecs, names, rg_row_offsets_[rg_start]);
    return Data(std::move(vecs), std::move(names));
  }

  /// Read specific rows by global index (sparse access for subsampling).
  ///
  /// Groups the requested indices by row group, reads only the needed
  /// row groups, and filters to the requested rows.
  ///
  /// @param indices  Sorted global row indices to read.
  /// @return Data with series in the same order as indices.
  Data read_rows(std::vector<int64_t> indices) const
  {
    if (indices.empty())
      return Data{};

    // Validate all indices are within bounds
    for (auto idx : indices) {
      if (idx < 0 || idx >= total_rows_)
        throw std::runtime_error("ParquetChunkReader::read_rows: index " +
          std::to_string(idx) + " out of range [0, " + std::to_string(total_rows_) + ")");
    }

    // Sort indices and remember original positions for reordering
    std::vector<size_t> order(indices.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return indices[a] < indices[b];
    });

    std::vector<int64_t> sorted_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
      sorted_indices[i] = indices[order[i]];

    // Group by row group
    std::vector<std::vector<data_t>> result_vecs(indices.size());
    std::vector<std::string> result_names(indices.size());

    size_t idx_pos = 0;
    for (int rg = 0; rg < num_row_groups_ && idx_pos < sorted_indices.size(); ++rg) {
      int64_t rg_start = rg_row_offsets_[rg];
      int64_t rg_end = rg_start + rg_row_counts_[rg];

      // Collect indices that fall in this row group
      std::vector<int64_t> local_indices;
      std::vector<size_t> result_positions; // position in sorted order
      while (idx_pos < sorted_indices.size() && sorted_indices[idx_pos] < rg_end) {
        local_indices.push_back(sorted_indices[idx_pos] - rg_start);
        result_positions.push_back(idx_pos);
        ++idx_pos;
      }

      if (local_indices.empty()) continue;

      // Read this row group
      std::shared_ptr<arrow::Table> table;
      detail::check_arrow_chunk(
        reader_->ReadRowGroups({rg}, {col_idx_}, &table),
        "read_rows ReadRowGroups");

      auto col = table->column(0);
      std::vector<std::vector<data_t>> rg_vecs;
      std::vector<std::string> rg_names;
      rg_vecs.reserve(static_cast<size_t>(table->num_rows()));
      rg_names.reserve(static_cast<size_t>(table->num_rows()));
      detail::extract_series_from_column(col, col_type_, rg_vecs, rg_names, rg_start);

      // Pick only the requested rows (copy, not move, to handle duplicate indices safely)
      for (size_t k = 0; k < local_indices.size(); ++k) {
        size_t local_idx = static_cast<size_t>(local_indices[k]);
        size_t sorted_pos = result_positions[k];
        size_t orig_pos = order[sorted_pos];

        result_vecs[orig_pos] = rg_vecs[local_idx];
        result_names[orig_pos] = "series_" + std::to_string(indices[orig_pos]);
      }
    }

    return Data(std::move(result_vecs), std::move(result_names));
  }

  /// Compute how many row groups fit in a RAM budget.
  ///
  /// @param ram_budget  Available bytes for chunk data.
  /// @return Number of row groups per batch (at least 1).
  int row_groups_per_batch(size_t ram_budget) const
  {
    if (num_row_groups_ == 0) return 0;
    size_t avg_rg_bytes = estimated_bytes_per_series_ *
      static_cast<size_t>(total_rows_ / num_row_groups_);
    if (avg_rg_bytes == 0) return 1; // Conservative: 1 row group at a time when metadata is empty
    return std::max(1, static_cast<int>(ram_budget / avg_rg_bytes));
  }

private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  int col_idx_ = 0;
  std::shared_ptr<arrow::DataType> col_type_;

  int num_row_groups_ = 0;
  int64_t total_rows_ = 0;
  std::vector<int64_t> rg_row_counts_;
  std::vector<int64_t> rg_row_offsets_;
  size_t estimated_bytes_per_series_ = 0;
};

} // namespace dtwc::io

#endif // DTWC_HAS_PARQUET
