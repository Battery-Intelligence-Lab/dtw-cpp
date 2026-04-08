/// @file arrow_ipc_reader.hpp — Zero-copy Arrow IPC file reader for time series data.
///
/// Reads Arrow IPC files (Feather v2) via memory-mapped I/O.
/// The file must contain a "data" column of type List<Float64> or
/// LargeList<Float64>, and optionally a "name" column of type Utf8.
/// Schema metadata key "ndim" specifies features per timestep (default 1).
///
/// Requires DTWC_HAS_ARROW (Apache Arrow, Apache-2.0 license).

#pragma once

#ifdef DTWC_HAS_ARROW

#include "../settings.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::io {

/// Memory-mapped Arrow IPC file reader providing zero-copy span access.
class ArrowIPCDataSource {
  std::shared_ptr<arrow::io::MemoryMappedFile> mmap_;
  std::shared_ptr<arrow::Table> table_;

  // Cached pointers for O(1) series access — exactly one of these is non-null.
  const arrow::ListArray *list_{ nullptr };           // List<Float64> (int32 offsets)
  const arrow::LargeListArray *large_list_{ nullptr }; // LargeList<Float64> (int64 offsets)
  const double *flat_values_{ nullptr };               // raw pointer into mmap'd values buffer
  const arrow::StringArray *names_{ nullptr };
  size_t n_{ 0 };
  size_t ndim_{ 1 };

  ArrowIPCDataSource() = default;

  static void check_status(const arrow::Status &s, const char *context)
  {
    if (!s.ok())
      throw std::runtime_error(std::string(context) + ": " + s.ToString());
  }

  /// Get offset range for series i (works for both List and LargeList).
  std::pair<int64_t, int64_t> offset_range(size_t i) const
  {
    const auto ii = static_cast<int64_t>(i);
    if (list_)
      return { list_->value_offset(ii), list_->value_offset(ii + 1) };
    return { large_list_->value_offset(ii), large_list_->value_offset(ii + 1) };
  }

public:
  /// Open an Arrow IPC file via memory-mapped I/O.
  static ArrowIPCDataSource open(const std::filesystem::path &path)
  {
    ArrowIPCDataSource src;

    // Memory-map the file
    auto mmap_result = arrow::io::MemoryMappedFile::Open(path.string(), arrow::io::FileMode::READ);
    check_status(mmap_result.status(), "ArrowIPCDataSource::open mmap");
    src.mmap_ = *mmap_result;

    // Open IPC file reader
    auto reader_result = arrow::ipc::RecordBatchFileReader::Open(src.mmap_);
    check_status(reader_result.status(), "ArrowIPCDataSource::open reader");
    auto reader = *reader_result;

    // Read all record batches into a table
    auto table_result = reader->ToTable();
    check_status(table_result.status(), "ArrowIPCDataSource::open ToTable");
    src.table_ = *table_result;

    // Find the data column
    auto data_col = src.table_->GetColumnByName("data");
    if (!data_col)
      throw std::runtime_error("ArrowIPCDataSource: missing 'data' column");

    if (data_col->num_chunks() != 1)
      throw std::runtime_error(
        "ArrowIPCDataSource: 'data' column has " +
        std::to_string(data_col->num_chunks()) + " chunks (expected 1). "
        "Re-convert the file to produce a single record batch.");

    auto chunk = data_col->chunk(0);
    const arrow::DoubleArray *values = nullptr;

    // Dispatch on List vs LargeList (type-safe, no reinterpret_cast)
    if (chunk->type_id() == arrow::Type::LIST) {
      src.list_ = static_cast<const arrow::ListArray *>(chunk.get());
      values = static_cast<const arrow::DoubleArray *>(src.list_->values().get());
      src.n_ = static_cast<size_t>(src.list_->length());
    } else if (chunk->type_id() == arrow::Type::LARGE_LIST) {
      src.large_list_ = static_cast<const arrow::LargeListArray *>(chunk.get());
      values = static_cast<const arrow::DoubleArray *>(src.large_list_->values().get());
      src.n_ = static_cast<size_t>(src.large_list_->length());
    } else {
      throw std::runtime_error(
        "ArrowIPCDataSource: 'data' column must be List<Float64> or LargeList<Float64>, got " +
        chunk->type()->ToString());
    }

    src.flat_values_ = values->raw_values();

    // Find optional name column
    auto name_col = src.table_->GetColumnByName("name");
    if (name_col && name_col->num_chunks() == 1) {
      src.names_ = static_cast<const arrow::StringArray *>(name_col->chunk(0).get());
    }

    // Read ndim from schema metadata
    auto metadata = src.table_->schema()->metadata();
    if (metadata) {
      auto idx = metadata->FindKey("ndim");
      if (idx >= 0)
        src.ndim_ = static_cast<size_t>(std::stoul(metadata->value(idx)));
    }

    return src;
  }

  size_t size() const { return n_; }
  size_t ndim() const { return ndim_; }

  /// Zero-copy span of series i's flat data (timesteps * ndim doubles).
  std::span<const data_t> series(size_t i) const
  {
    assert(i < n_);
    auto [start, end] = offset_range(i);
    return { flat_values_ + start, static_cast<size_t>(end - start) };
  }

  size_t series_flat_size(size_t i) const
  {
    assert(i < n_);
    auto [start, end] = offset_range(i);
    return static_cast<size_t>(end - start);
  }

  size_t series_length(size_t i) const { return series_flat_size(i) / ndim_; }

  /// Name of series i. Returns "series_N" if no name column exists.
  std::string name(size_t i) const
  {
    if (names_) return names_->GetString(static_cast<int64_t>(i));
    return "series_" + std::to_string(i);
  }

  std::vector<std::string> all_names() const
  {
    std::vector<std::string> result(n_);
    for (size_t i = 0; i < n_; ++i)
      result[i] = name(i);
    return result;
  }
};

} // namespace dtwc::io

#endif // DTWC_HAS_ARROW
