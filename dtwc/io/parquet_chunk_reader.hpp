/// @file parquet_chunk_reader.hpp — The Parquet reader.
///
/// read_arrow() reads a file whole; streaming CLARA reads row groups and
/// sparse rows, each row group independently, without loading the entire
/// dataset into memory.
///
/// Requires DTWC_HAS_PARQUET (Apache Arrow + Parquet, Apache-2.0 license).
///
/// @author Volkan Kumtepeli
/// @author Claude 4.6
/// @date 08 Apr 2026

#pragma once

#ifdef DTWC_HAS_PARQUET

#include "../Data.hpp"
#include "../base/error.hpp"
#include "../base/settings.hpp"
#include "../fileOperations.hpp"
#include "parquet_schema.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dtwc::io {

namespace detail {

inline void check_arrow_chunk(const arrow::Status &s, const char *ctx)
{
  if (!s.ok())
    throw dtwc::IOError(std::string(ctx) + ": " + s.ToString());
}

/// Extract one list cell into `out`, converting Float32/Float64 to `T`.
/// The null check is one `null_count()` read per array, hoisted by the caller.
template <typename T, typename ListArrayT>
inline void extract_list_element(const std::shared_ptr<ListArrayT> &list,
                                 int64_t i,
                                 std::vector<T> &out)
{
  const auto start = list->value_offset(i);
  const auto end = list->value_offset(i + 1);
  const auto &values = *list->values();
  require_list_range(start, end, values.length());
  const auto sz = static_cast<size_t>(end - start);
  out.resize(sz);
  copy_arrow_numeric<T>(values, start, sz, out.data());
}

/// Append the series of an Arrow table column to `vecs` as `T`: a scalar
/// column is one series, a list or large-list column one series per cell, for
/// Float32 and Float64 source values alike.
template <typename T>
inline void extract_series_from_column(
  const std::shared_ptr<arrow::ChunkedArray> &col,
  const std::shared_ptr<arrow::DataType> &col_type,
  std::vector<std::vector<T>> &vecs)
{
  const bool is_list = col_type->id() == arrow::Type::LIST
    || col_type->id() == arrow::Type::LARGE_LIST;

  if (is_list) {
    auto append_cells = [&](auto list) {
      // Two null_count() reads per chunk: the list cells and their values.
      require_no_nulls(*list, "list column cell");
      require_no_nulls(*list->values(), "list column value");
      const int64_t len = list->length();
      for (int64_t i = 0; i < len; ++i) {
        std::vector<T> series;
        extract_list_element(list, i, series);
        vecs.push_back(std::move(series));
      }
    };

    for (int c = 0; c < col->num_chunks(); ++c) {
      auto chunk = col->chunk(c);
      if (col_type->id() == arrow::Type::LIST)
        append_cells(std::static_pointer_cast<arrow::ListArray>(chunk));
      else
        append_cells(std::static_pointer_cast<arrow::LargeListArray>(chunk));
    }
    return;
  }

  // Scalar column: the whole column is one series.
  std::vector<T> series;
  for (int c = 0; c < col->num_chunks(); ++c) {
    const auto chunk = col->chunk(c);
    require_no_nulls(*chunk, "scalar column value");
    const auto n = static_cast<size_t>(chunk->length());
    const size_t offset = series.size();
    series.resize(offset + n);
    copy_arrow_numeric<T>(*chunk, 0, n, series.data() + offset);
  }
  vecs.push_back(std::move(series));
}

} // namespace detail


/// The Parquet reader: the whole file, row groups, or sparse rows.
///
/// Opens the file once (via mmap), reads metadata eagerly, and reads the
/// selected column on demand. A scalar column is one series named by the
/// file's stem; a list column is one series per row, named series_<row>.
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
    : name_(path_to_utf8(path.stem()))
  {
    // Arrow takes a UTF-8 path on every platform; path::string() is the ANSI
    // code page on Windows, where a non-ASCII name failed to open.
    auto mmap_result = arrow::io::MemoryMappedFile::Open(path_to_utf8(path), arrow::io::FileMode::READ);
    detail::check_arrow_chunk(mmap_result.status(), "ParquetChunkReader mmap");
    file_ = *mmap_result;

    auto builder = parquet::arrow::FileReaderBuilder();
    detail::check_arrow_chunk(builder.Open(file_), "ParquetChunkReader Open");
    detail::check_arrow_chunk(builder.Build(&reader_), "ParquetChunkReader Build");

    detail::check_arrow_chunk(reader_->GetSchema(&schema_), "GetSchema");
    const auto selected = detail::find_parquet_series_column(schema_, col_name);
    col_idx_ = selected.parquet_leaf_index;
    col_type_ = selected.type;
    list_layout_ = col_type_->id() == arrow::Type::LIST
                || col_type_->id() == arrow::Type::LARGE_LIST;

    const auto value_type = detail::parquet_series_value_type(col_type_);
    if (value_type->id() == arrow::Type::FLOAT)
      source_value_bytes_ = sizeof(float);
    else if (value_type->id() == arrow::Type::DOUBLE)
      source_value_bytes_ = sizeof(double);
    else
      throw dtwc::IOError(
        "ParquetChunkReader: selected column values must be Float32 or Float64, got "
        + value_type->ToString());

    // Cache row-group metadata
    auto *pq_meta = reader_->parquet_reader()->metadata().get();
    num_row_groups_ = pq_meta->num_row_groups();
    total_rows_ = pq_meta->num_rows();
    if (num_row_groups_ < 0 || total_rows_ < 0)
      throw dtwc::IOError("ParquetChunkReader: invalid negative file metadata");

    rg_row_counts_.resize(num_row_groups_);
    rg_row_offsets_.resize(num_row_groups_);
    rg_encoded_bytes_.resize(num_row_groups_);
    rg_value_counts_.resize(num_row_groups_);
    int64_t offset = 0;
    for (int rg = 0; rg < num_row_groups_; ++rg) {
      rg_row_counts_[rg] = pq_meta->RowGroup(rg)->num_rows();
      const auto column = pq_meta->RowGroup(rg)->ColumnChunk(col_idx_);
      const auto encoded_bytes = column->total_uncompressed_size();
      const auto value_count = column->num_values();
      if (rg_row_counts_[rg] < 0 || encoded_bytes < 0 || value_count < 0
          || rg_row_counts_[rg] > std::numeric_limits<int64_t>::max() - offset)
        throw dtwc::IOError("ParquetChunkReader: invalid row-group metadata");
      rg_row_offsets_[rg] = offset;
      rg_encoded_bytes_[rg] = static_cast<size_t>(encoded_bytes);
      rg_value_counts_[rg] = static_cast<size_t>(value_count);
      offset += rg_row_counts_[rg];
      column_encoded_bytes_ += rg_encoded_bytes_[rg];
      total_value_count_ += rg_value_counts_[rg];
    }
    if (offset != total_rows_)
      throw dtwc::IOError(
        "ParquetChunkReader: row-group counts do not match file metadata");

  }

  /// Number of row groups in the file.
  int num_row_groups() const { return num_row_groups_; }

  /// Total number of rows across all row groups.
  int64_t total_rows() const { return total_rows_; }

  /// Whether each physical row is one list-encoded time series.
  bool is_list_layout() const { return list_layout_; }

  /// Number of time series under the loader contract. A scalar column is one
  /// series regardless of its physical row count; a list column is one per row.
  int64_t logical_series_count() const
  {
    return list_layout_ ? total_rows_ : int64_t{1};
  }

  /// Number of rows in a specific row group.
  int64_t row_group_rows(int rg) const { return rg_row_counts_[rg]; }

  /// Conservative estimate of resident `Data` bytes for the selected output
  /// precision. The selected Parquet column bytes are never scaled down; a
  /// Float32 column is scaled up for Float64 materialization, and per-series
  /// vector/name objects are included.
  size_t estimated_resident_bytes(bool use_float32) const
  {
    const size_t payload = estimated_payload_bytes(use_float32);

    const size_t object_bytes = use_float32
      ? sizeof(std::vector<float>) + sizeof(std::string)
      : sizeof(std::vector<data_t>) + sizeof(std::string);
    return payload + static_cast<size_t>(logical_series_count()) * object_bytes;
  }

  /// Conservative peak while the CLI materializes this file. The current
  /// loader first owns an Arrow decode table and Float64 `Data`; Float32 output
  /// then allocates a second `Data` before releasing the Float64 vectors.
  size_t estimated_materialization_peak_bytes(bool use_float32) const
  {
    const size_t f64 = estimated_resident_bytes(false);
    const size_t decode_peak = source_payload_bytes() + f64;
    if (!use_float32) return decode_peak;
    const size_t conversion_peak = f64 + estimated_resident_bytes(true);
    return std::max(decode_peak, conversion_peak);
  }

  /// Append every series of the file to `series` and `names`. A list column's
  /// names number on from series.size(), so a folder's files never repeat one.
  void read_all(std::vector<std::vector<data_t>> &series, std::vector<std::string> &names) const
  {
    append_row_groups(0, num_row_groups_, series, names, series.size());
  }

  /// Read a contiguous batch of row groups [rg_start, rg_start+count) as `T`:
  /// data_t, or float at half the resident footprint.
  ///
  /// @param rg_start  First row group index.
  /// @param count     Number of row groups to read.
  /// @return Owning Data with all series from the batch.
  template <typename T = data_t>
  Data read_row_groups(int rg_start, int count) const
  {
    // fast_clara walks [0, num_row_groups()) in batches.
    assert(rg_start >= 0 && count >= 0 && count <= num_row_groups_ && rg_start <= num_row_groups_ - count);

    const auto rows = static_cast<size_t>(std::accumulate(
      rg_row_counts_.begin() + rg_start, rg_row_counts_.begin() + rg_start + count, int64_t{ 0 }));
    std::vector<std::vector<T>> vecs;
    std::vector<std::string> names;
    vecs.reserve(rows);
    names.reserve(rows);
    append_row_groups(rg_start, count, vecs, names, static_cast<size_t>(rg_row_offsets_[rg_start]));
    return Data(std::move(vecs), std::move(names));
  }

  /// Compute a conservative fixed batch count that fits every row-group batch
  /// in a RAM budget. Throws when even one row group cannot fit; silently
  /// exceeding the requested cap is never a valid fallback.
  ///
  /// @param ram_budget  Available bytes for chunk data.
  /// @param use_float32  Whether chunks materialize as Float32.
  /// @return Number of row groups per batch (at least 1 when non-empty).
  int row_groups_per_batch(size_t ram_budget, bool use_float32 = false) const
  {
    if (num_row_groups_ == 0) return 0;

    size_t largest_group = 0;
    const size_t object_bytes = use_float32
      ? sizeof(std::vector<float>) + sizeof(std::string)
      : sizeof(std::vector<data_t>) + sizeof(std::string);
    for (int rg = 0; rg < num_row_groups_; ++rg) {
      largest_group = std::max(
        largest_group,
        row_group_materialization_peak_bytes(rg, use_float32, object_bytes));
    }

    if (largest_group > ram_budget) {
      throw dtwc::InvalidInput(
        "ParquetChunkReader: --ram-limit leaves " +
        std::to_string(ram_budget) +
        " bytes for chunks, but one row group needs approximately " +
        std::to_string(largest_group) +
        " bytes; rewrite with smaller row groups or raise the limit");
    }
    if (largest_group == 0) return 1;
    const size_t capacity = std::min<size_t>(
      static_cast<size_t>(num_row_groups_), ram_budget / largest_group);
    return static_cast<int>(std::max<size_t>(1, capacity));
  }

  /// Read specific rows by global index (sparse access for subsampling) as
  /// `T`: data_t, or float to keep FastCLARA's Float32 route in Float32.
  ///
  /// Groups the requested indices by row group, reads only the needed
  /// row groups, and filters to the requested rows.
  ///
  /// @param indices  Global row indices to read.
  /// @return Data with series in the same order as indices.
  template <typename T = data_t>
  Data read_rows(
    std::vector<int64_t> indices,
    size_t ram_budget = std::numeric_limits<size_t>::max()) const
  {
    static_assert(std::is_same_v<T, data_t> || std::is_same_v<T, float>);
    if (indices.empty()) return Data{};
    // fast_clara rejects a scalar column before streaming and samples indices
    // from [0, rows).
    assert(list_layout_);
    for ([[maybe_unused]] const auto index : indices) assert(index >= 0 && index < total_rows_);

    const size_t result_object_bytes = sizeof(std::vector<T>) + sizeof(std::string);
    size_t retained_bytes = indices.size() * result_object_bytes;
    if (retained_bytes > ram_budget)
      throw dtwc::InvalidInput(
        "ParquetChunkReader::read_rows: --ram-limit is too small for sparse "
        "sample metadata");

    std::vector<size_t> order(indices.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
      return indices[lhs] < indices[rhs];
    });
    std::vector<int64_t> sorted_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
      sorted_indices[i] = indices[order[i]];

    std::vector<std::vector<T>> result_vecs(indices.size());
    std::vector<std::string> result_names(indices.size());
    size_t index_position = 0;
    for (int rg = 0;
         rg < num_row_groups_ && index_position < sorted_indices.size(); ++rg) {
      const int64_t row_group_start = rg_row_offsets_[rg];
      const int64_t row_group_end = row_group_start + rg_row_counts_[rg];
      std::vector<int64_t> local_indices;
      std::vector<size_t> result_positions;
      while (index_position < sorted_indices.size()
             && sorted_indices[index_position] < row_group_end) {
        local_indices.push_back(
          sorted_indices[index_position] - row_group_start);
        result_positions.push_back(index_position++);
      }
      if (local_indices.empty()) continue;

      const size_t group_peak = row_group_materialization_peak_bytes(
        rg, std::is_same_v<T, float>, result_object_bytes);
      if (group_peak > ram_budget - retained_bytes)
        throw dtwc::InvalidInput(
          "ParquetChunkReader::read_rows: selected row group needs " +
          std::to_string(group_peak) + " bytes in addition to " +
          std::to_string(retained_bytes) +
          " retained sample bytes, exceeding --ram-limit=" +
          std::to_string(ram_budget) +
          "; rewrite with smaller row groups or raise the limit");

      std::shared_ptr<arrow::Table> table;
      detail::check_arrow_chunk(
        reader_->ReadRowGroups({rg}, {col_idx_}, &table),
        "read_rows ReadRowGroups");
      std::vector<std::vector<T>> row_group_series;
      row_group_series.reserve(static_cast<size_t>(table->num_rows()));
      detail::extract_series_from_column(table->column(0), col_type_, row_group_series);

      size_t selected_payload = 0;
      for (const auto local_index : local_indices) {
        selected_payload += row_group_series[static_cast<size_t>(local_index)].size() * sizeof(T);
      }
      const size_t copy_budget = ram_budget - retained_bytes - group_peak;
      if (selected_payload > copy_budget)
        throw dtwc::InvalidInput(
          "ParquetChunkReader::read_rows: copying selected series needs " +
          std::to_string(selected_payload) +
          " additional bytes, exceeding --ram-limit=" +
          std::to_string(ram_budget));

      // Copy rather than move because duplicate requested indices are valid.
      for (size_t i = 0; i < local_indices.size(); ++i) {
        const size_t original = order[result_positions[i]];
        result_vecs[original] =
          row_group_series[static_cast<size_t>(local_indices[i])];
        result_names[original] =
          "series_" + std::to_string(indices[original]);
      }
      retained_bytes += selected_payload;
    }
    return Data(std::move(result_vecs), std::move(result_names));
  }

private:
  /// Append the series of row groups [rg_start, rg_start+count) with their
  /// names: the file's stem for a scalar column, series_<first_name + i> for
  /// the i-th row of a list column.
  template <typename T>
  void append_row_groups(int rg_start, int count, std::vector<std::vector<T>> &vecs,
                         std::vector<std::string> &names, size_t first_name) const
  {
    std::vector<int> rg_indices(count);
    std::iota(rg_indices.begin(), rg_indices.end(), rg_start);

    std::shared_ptr<arrow::Table> table;
    detail::check_arrow_chunk(
      reader_->ReadRowGroups(rg_indices, {col_idx_}, &table),
      "read_row_groups");

    const size_t first = vecs.size();
    detail::extract_series_from_column(table->column(0), col_type_, vecs);
    if (!list_layout_) {
      names.push_back(name_);
      return;
    }
    for (size_t i = first; i < vecs.size(); ++i)
      names.push_back("series_" + std::to_string(first_name + (i - first)));
  }

  size_t estimated_payload_bytes(bool use_float32) const
  {
    const size_t target_width = use_float32 ? sizeof(float) : sizeof(data_t);
    return std::max(
      column_encoded_bytes_,
      total_value_count_ * target_width);
  }

  size_t source_payload_bytes() const
  {
    return std::max(
      column_encoded_bytes_,
      total_value_count_ * source_value_bytes_);
  }

  size_t row_group_materialization_peak_bytes(
    int row_group, bool use_float32, size_t object_bytes) const
  {
    const size_t source = std::max(
      rg_encoded_bytes_[row_group], rg_value_counts_[row_group] * source_value_bytes_);
    const size_t target_width = use_float32 ? sizeof(float) : sizeof(data_t);
    const size_t target = rg_value_counts_[row_group] * target_width;
    const size_t objects = static_cast<size_t>(rg_row_counts_[row_group]) * object_bytes;
    return source + target + objects;
  }

  std::string name_; ///< the file's stem in UTF-8: a scalar column's series name
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  int col_idx_ = 0;
  std::shared_ptr<arrow::DataType> col_type_;
  bool list_layout_ = false;
  size_t source_value_bytes_ = 0;

  int num_row_groups_ = 0;
  int64_t total_rows_ = 0;
  std::vector<int64_t> rg_row_counts_;
  std::vector<int64_t> rg_row_offsets_;
  std::vector<size_t> rg_encoded_bytes_;
  std::vector<size_t> rg_value_counts_;
  size_t column_encoded_bytes_ = 0;
  size_t total_value_count_ = 0;
};

} // namespace dtwc::io

#endif // DTWC_HAS_PARQUET
