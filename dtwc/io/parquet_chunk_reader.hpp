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
#include <arrow/util/config.h>
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
#include <optional>
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

/// Append one series per row of `table` to `vecs` as `T`: the row's values in
/// the Float32/Float64 columns at `positions`, in that order (the row layout).
template <typename T>
inline void extract_row_series(const arrow::Table &table, const std::vector<int> &positions,
                               std::vector<std::vector<T>> &vecs)
{
  const size_t first = vecs.size();
  vecs.resize(first + static_cast<size_t>(table.num_rows()), std::vector<T>(positions.size()));
  for (size_t j = 0; j < positions.size(); ++j) {
    size_t row = first;
    for (const auto &chunk : table.column(positions[j])->chunks()) {
      require_no_nulls(*chunk, ("column '" + table.field(positions[j])->name() + "'").c_str());
      const auto n = static_cast<size_t>(chunk->length());
      const auto scatter = [&](const auto *raw) {
        for (size_t i = 0; i < n; ++i) vecs[row + i][j] = static_cast<T>(raw[i]);
      };
      if (chunk->type_id() == arrow::Type::DOUBLE)
        scatter(static_cast<const arrow::DoubleArray &>(*chunk).raw_values());
      else
        scatter(static_cast<const arrow::FloatArray &>(*chunk).raw_values());
      row += n;
    }
  }
}

/// Append each row's name in a Utf8/LargeUtf8 column to `names`; a null is no name.
inline void extract_row_names(const arrow::ChunkedArray &column, std::vector<std::optional<std::string>> &names)
{
  for (const auto &chunk : column.chunks()) {
    const auto append = [&](const auto &strings) {
      for (int64_t i = 0; i < strings.length(); ++i)
        names.push_back(strings.IsNull(i) ? std::nullopt : std::optional<std::string>(strings.GetString(i)));
    };
    if (chunk->type_id() == arrow::Type::STRING)
      append(static_cast<const arrow::StringArray &>(*chunk));
    else
      append(static_cast<const arrow::LargeStringArray &>(*chunk));
  }
}

} // namespace detail


/// The Parquet reader: the whole file, row groups, or sparse rows.
///
/// Opens the file once (via mmap), reads metadata eagerly, and reads the
/// columns detail::resolve_parquet_layout() selects on demand: a scalar column
/// is one series named by the file's stem; a list column, or a row of the
/// file's Float32/Float64 columns, is one series per row, named by the file's
/// first string column (a null: series_<row>), else series_<row>. `skip_cols`
/// drops the file's leading columns and `skip_rows` its leading rows (of a
/// scalar column, the series' leading values); the rows are numbered after them.
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
  /// @param col_name   Column to extract (empty: the layout rule decides).
  /// @param skip_cols  Leading columns dropped, as a CSV row's leading fields.
  /// @param skip_rows  Leading rows dropped, as a CSV file's leading lines.
  explicit ParquetChunkReader(const std::filesystem::path &path,
                              const std::string &col_name = "",
                              std::int64_t skip_cols = 0, std::int64_t skip_rows = 0)
    : name_(path_to_utf8(path.stem())), row_offset_(skip_rows)
  {
    if (skip_cols < 0 || skip_rows < 0)
      throw dtwc::InvalidInput("--skip-cols and --skip-rows (skip_cols, skip_rows) must be non-negative.");
    // Arrow takes a UTF-8 path on every platform; path::string() is the ANSI
    // code page on Windows, where a non-ASCII name failed to open.
    auto mmap_result = arrow::io::MemoryMappedFile::Open(path_to_utf8(path), arrow::io::FileMode::READ);
    detail::check_arrow_chunk(mmap_result.status(), "ParquetChunkReader mmap");
    file_ = *mmap_result;

    auto builder = parquet::arrow::FileReaderBuilder();
    detail::check_arrow_chunk(builder.Open(file_), "ParquetChunkReader Open");
    detail::check_arrow_chunk(builder.Build(&reader_), "ParquetChunkReader Build");

    detail::check_arrow_chunk(reader_->GetSchema(&schema_), "GetSchema");
    layout_ = detail::resolve_parquet_layout(schema_, col_name, skip_cols);
    // The leaves a batch reads, ascending: a read table's columns follow them.
    for (const auto &column : layout_.samples) leaves_.push_back(column.parquet_leaf_index);
    if (layout_.names) leaves_.push_back(layout_.names->parquet_leaf_index);
    std::sort(leaves_.begin(), leaves_.end());
    const auto position = [&](const detail::ParquetColumn &column) {
      return static_cast<int>(
        std::lower_bound(leaves_.begin(), leaves_.end(), column.parquet_leaf_index) - leaves_.begin());
    };
    for (const auto &column : layout_.samples) {
      sample_positions_.push_back(position(column));
      const bool f32 = detail::parquet_series_value_type(column.type)->id() == arrow::Type::FLOAT;
      source_value_bytes_ = std::max(source_value_bytes_, f32 ? sizeof(float) : sizeof(double));
    }
    if (layout_.names) name_position_ = position(*layout_.names);

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
    rg_name_bytes_.resize(num_row_groups_);
    int64_t offset = 0;
    for (int rg = 0; rg < num_row_groups_; ++rg) {
      const auto group = pq_meta->RowGroup(rg);
      rg_row_counts_[rg] = group->num_rows();
      size_t encoded_bytes = 0; // the sample columns
      size_t value_count = 0;   // their values
      size_t name_bytes = 0;    // the name column: decoded, then copied into the series' names
      for (const int leaf : leaves_) {
        const auto column = group->ColumnChunk(leaf);
        if (column->total_uncompressed_size() < 0 || column->num_values() < 0)
          throw dtwc::IOError("ParquetChunkReader: invalid row-group metadata");
        if (layout_.names && leaf == layout_.names->parquet_leaf_index) {
          name_bytes += static_cast<size_t>(column->total_uncompressed_size());
          continue;
        }
        encoded_bytes += static_cast<size_t>(column->total_uncompressed_size());
        value_count += static_cast<size_t>(column->num_values());
      }
      if (rg_row_counts_[rg] < 0 || rg_row_counts_[rg] > std::numeric_limits<int64_t>::max() - offset)
        throw dtwc::IOError("ParquetChunkReader: invalid row-group metadata");
      rg_row_offsets_[rg] = offset;
      rg_encoded_bytes_[rg] = encoded_bytes;
      rg_value_counts_[rg] = value_count;
      rg_name_bytes_[rg] = name_bytes;
      offset += rg_row_counts_[rg];
      column_encoded_bytes_ += rg_encoded_bytes_[rg];
      total_value_count_ += rg_value_counts_[rg];
      name_bytes_ += name_bytes;
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
  bool is_list_layout() const { return layout_.kind == detail::ParquetSeriesLayout::Kind::List; }

  /// Whether each row is one series: a list column, or a row of Float32/Float64
  /// columns. Otherwise the file is one series, its one scalar column.
  bool rows_are_series() const { return layout_.kind != detail::ParquetSeriesLayout::Kind::OneSeries; }

  /// Number of time series under the loader contract: one per row after
  /// skip_rows, or one for a scalar column whatever its row count.
  int64_t logical_series_count() const
  {
    return rows_are_series() ? std::max<int64_t>(total_rows_ - row_offset_, 0) : int64_t{1};
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

  /// Append every series of the file to `series` and `names`. A row's
  /// series_<i> numbers on from series.size(), so a folder's files never repeat one.
  void read_all(std::vector<std::vector<data_t>> &series, std::vector<std::string> &names) const
  {
    append_row_groups(0, num_row_groups_, series, names, series.size());
  }

  /// Every series' name, as read_all() names them, reading the name column
  /// alone, a row group at a time: what a RAM-limited run writes beside its labels.
  std::vector<std::string> series_names() const
  {
    if (!rows_are_series()) return { name_ };
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(logical_series_count()));
    for (int rg = 0; rg < num_row_groups_; ++rg) {
      const auto rows = static_cast<size_t>(rg_row_counts_[rg]);
      const auto dropped = static_cast<size_t>(
        std::clamp<int64_t>(row_offset_ - first_row_of(rg), 0, rg_row_counts_[rg]));
      std::vector<std::optional<std::string>> row_names;
      if (layout_.names && dropped < rows)
        detail::extract_row_names(
          *read_groups({ rg }, { layout_.names->parquet_leaf_index }, "series_names")->column(0), row_names);
      for (size_t row = dropped; row < rows; ++row)
        names.push_back(layout_.names && row_names[row] ? std::move(*row_names[row])
                                                        : "series_" + std::to_string(names.size()));
    }
    return names;
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
    append_row_groups(rg_start, count, vecs, names,
                      static_cast<size_t>(std::max<int64_t>(first_row_of(rg_start) - row_offset_, 0)));
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
  /// `T`: data_t, or float to keep FastCLARA's Float32 route in Float32. An
  /// index counts the series, the rows after skip_rows.
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
    // from [0, logical_series_count()).
    assert(rows_are_series());
    for ([[maybe_unused]] const auto index : indices)
      assert(index >= 0 && index < logical_series_count());

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
    std::vector<int64_t> sorted_indices(indices.size()); // the file's rows
    for (size_t i = 0; i < indices.size(); ++i)
      sorted_indices[i] = indices[order[i]] + row_offset_;

    std::vector<std::vector<T>> result_vecs(indices.size());
    std::vector<std::string> result_names(indices.size());
    size_t index_position = 0;
    for (int rg = 0;
         rg < num_row_groups_ && index_position < sorted_indices.size(); ++rg) {
      const int64_t row_group_start = rg_row_offsets_[rg];
      const int64_t row_group_end = row_group_start + rg_row_counts_[rg];
      // As read_row_groups: the group's rows before row_offset_ are not read, so
      // a local index counts from the first row after them.
      const int64_t dropped = std::clamp<int64_t>(row_offset_ - row_group_start, 0, rg_row_counts_[rg]);
      std::vector<int64_t> local_indices;
      std::vector<size_t> result_positions;
      while (index_position < sorted_indices.size()
             && sorted_indices[index_position] < row_group_end) {
        local_indices.push_back(
          sorted_indices[index_position] - row_group_start - dropped);
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

      auto table = read_groups({ rg }, leaves_, "read_rows ReadRowGroups");
      if (dropped > 0) table = table->Slice(dropped);
      std::vector<std::vector<T>> row_group_series;
      row_group_series.reserve(static_cast<size_t>(table->num_rows()));
      extract_rows(*table, row_group_series);
      std::vector<std::optional<std::string>> row_group_names;
      if (layout_.names) detail::extract_row_names(*table->column(name_position_), row_group_names);

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
        const auto local = static_cast<size_t>(local_indices[i]);
        result_vecs[original] = row_group_series[local];
        result_names[original] = layout_.names && row_group_names[local]
          ? *row_group_names[local] : "series_" + std::to_string(indices[original]);
      }
      retained_bytes += selected_payload;
    }
    return Data(std::move(result_vecs), std::move(result_names));
  }

private:
  /// The `leaves` of `groups`: Arrow 24 deprecated the Status form for the Result form.
  std::shared_ptr<arrow::Table> read_groups(const std::vector<int> &groups, const std::vector<int> &leaves,
                                            const char *what) const
  {
#if ARROW_VERSION_MAJOR >= 24
    auto table = reader_->ReadRowGroups(groups, leaves);
    detail::check_arrow_chunk(table.status(), what);
    return *std::move(table);
#else
    std::shared_ptr<arrow::Table> table;
    detail::check_arrow_chunk(reader_->ReadRowGroups(groups, leaves, &table), what);
    return table;
#endif
  }

  /// The file row a group starts at; the row count past the last group.
  int64_t first_row_of(int rg) const { return rg < num_row_groups_ ? rg_row_offsets_[rg] : total_rows_; }

  /// Append one series per row of a read table: a list column's cells, or the row's samples.
  template <typename T>
  void extract_rows(const arrow::Table &table, std::vector<std::vector<T>> &vecs) const
  {
    if (layout_.kind == detail::ParquetSeriesLayout::Kind::Rows)
      detail::extract_row_series(table, sample_positions_, vecs);
    else
      detail::extract_series_from_column(table.column(sample_positions_.front()), layout_.samples.front().type,
                                         vecs);
  }

  /// Append the series of row groups [rg_start, rg_start+count), less the rows
  /// before row_offset_, with their names: the file's stem for a scalar column;
  /// for a row, its name in the string column, else series_<first_name + i>
  /// for the i-th row kept.
  template <typename T>
  void append_row_groups(int rg_start, int count, std::vector<std::vector<T>> &vecs,
                         std::vector<std::string> &names, size_t first_name) const
  {
    std::vector<int> rg_indices(count);
    std::iota(rg_indices.begin(), rg_indices.end(), rg_start);
    // The rows before row_offset_ are not read, as a CSV file's skipped lines: a null there is no error.
    auto table = read_groups(rg_indices, leaves_, "read_row_groups");
    const auto dropped = static_cast<size_t>(
      std::clamp<int64_t>(row_offset_ - first_row_of(rg_start), 0, table->num_rows()));
    if (dropped > 0) table = table->Slice(static_cast<int64_t>(dropped));

    if (!rows_are_series()) { // the column is one series: skip_rows drops its leading values
      detail::extract_series_from_column(table->column(sample_positions_.front()), layout_.samples.front().type,
                                         vecs);
      names.push_back(name_);
      return;
    }
    std::vector<std::vector<T>> rows;
    rows.reserve(static_cast<size_t>(table->num_rows()));
    extract_rows(*table, rows);
    std::vector<std::optional<std::string>> row_names;
    if (layout_.names) detail::extract_row_names(*table->column(name_position_), row_names);
    for (size_t row = 0; row < rows.size(); ++row) {
      vecs.push_back(std::move(rows[row]));
      names.push_back(layout_.names && row_names[row] ? std::move(*row_names[row])
                                                      : "series_" + std::to_string(first_name + row));
    }
  }

  size_t estimated_payload_bytes(bool use_float32) const
  {
    const size_t target_width = use_float32 ? sizeof(float) : sizeof(data_t);
    return std::max(
      column_encoded_bytes_,
      total_value_count_ * target_width) + name_bytes_;
  }

  size_t source_payload_bytes() const
  {
    return std::max(
      column_encoded_bytes_,
      total_value_count_ * source_value_bytes_) + name_bytes_;
  }

  size_t row_group_materialization_peak_bytes(
    int row_group, bool use_float32, size_t object_bytes) const
  {
    const size_t source = std::max(
      rg_encoded_bytes_[row_group], rg_value_counts_[row_group] * source_value_bytes_);
    const size_t target_width = use_float32 ? sizeof(float) : sizeof(data_t);
    const size_t target = rg_value_counts_[row_group] * target_width;
    const size_t objects = static_cast<size_t>(rg_row_counts_[row_group]) * object_bytes;
    return source + target + objects + 2 * rg_name_bytes_[row_group];
  }

  std::string name_; ///< the file's stem in UTF-8: a scalar column's series name
  int64_t row_offset_ = 0; ///< skip_rows: the file rows before the first series
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  detail::ParquetSeriesLayout layout_;
  std::vector<int> leaves_;           ///< the Parquet leaves read, ascending
  std::vector<int> sample_positions_; ///< each sample column's column in a read table
  int name_position_ = -1;            ///< the name column's column in a read table
  size_t source_value_bytes_ = 0;

  int num_row_groups_ = 0;
  int64_t total_rows_ = 0;
  std::vector<int64_t> rg_row_counts_;
  std::vector<int64_t> rg_row_offsets_;
  std::vector<size_t> rg_encoded_bytes_;
  std::vector<size_t> rg_value_counts_;
  std::vector<size_t> rg_name_bytes_;
  size_t column_encoded_bytes_ = 0;
  size_t total_value_count_ = 0;
  size_t name_bytes_ = 0;
};

} // namespace dtwc::io

#endif // DTWC_HAS_PARQUET
