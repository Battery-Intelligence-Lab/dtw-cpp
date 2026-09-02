/**
 * @file parquet_schema.hpp
 * @brief Shared Parquet time-series column selection.
 */

#pragma once

#ifdef DTWC_HAS_PARQUET

#include <arrow/api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>

namespace dtwc::io::detail {

struct ParquetSeriesColumn
{
  int arrow_field_index;
  int parquet_leaf_index;
  std::shared_ptr<arrow::DataType> type;
};

inline int parquet_leaf_count(const std::shared_ptr<arrow::DataType> &type)
{
  const auto &fields = type->fields();
  if (fields.empty()) return 1;
  int count = 0;
  for (const auto &field : fields) {
    const int child_count = parquet_leaf_count(field->type());
    if (child_count > std::numeric_limits<int>::max() - count)
      throw std::runtime_error("Parquet schema has too many physical leaf columns");
    count += child_count;
  }
  return count;
}

inline std::shared_ptr<arrow::DataType> parquet_series_value_type(
  const std::shared_ptr<arrow::DataType> &type)
{
  if (type->id() == arrow::Type::LIST)
    return std::static_pointer_cast<arrow::ListType>(type)->value_type();
  if (type->id() == arrow::Type::LARGE_LIST)
    return std::static_pointer_cast<arrow::LargeListType>(type)->value_type();
  return type;
}

inline bool is_parquet_series_column(
  const std::shared_ptr<arrow::DataType> &type)
{
  const auto value_type = parquet_series_value_type(type);
  return value_type->id() == arrow::Type::FLOAT
         || value_type->id() == arrow::Type::DOUBLE;
}

/** Select a scalar/list Float32/Float64 column with identical eager/streaming rules. */
inline ParquetSeriesColumn find_parquet_series_column(
  const std::shared_ptr<arrow::Schema> &schema,
  const std::string &column_name)
{
  int leaf_index = 0;
  if (!column_name.empty()) {
    const int index = schema->GetFieldIndex(column_name);
    if (index < 0)
      throw std::runtime_error(
        "Column '" + column_name + "' not found in Parquet schema");
    if (!is_parquet_series_column(schema->field(index)->type()))
      throw std::runtime_error(
        "Parquet column '" + column_name +
        "' must be Float32, Float64, List<Float32/Float64>, or "
        "LargeList<Float32/Float64>");
    for (int preceding = 0; preceding < index; ++preceding) {
      const int leaves = parquet_leaf_count(schema->field(preceding)->type());
      if (leaves > std::numeric_limits<int>::max() - leaf_index)
        throw std::runtime_error("Parquet schema has too many physical leaf columns");
      leaf_index += leaves;
    }
    return { index, leaf_index, schema->field(index)->type() };
  }

  for (int index = 0; index < schema->num_fields(); ++index) {
    if (is_parquet_series_column(schema->field(index)->type()))
      return { index, leaf_index, schema->field(index)->type() };
    const int leaves = parquet_leaf_count(schema->field(index)->type());
    if (leaves > std::numeric_limits<int>::max() - leaf_index)
      throw std::runtime_error("Parquet schema has too many physical leaf columns");
    leaf_index += leaves;
  }
  throw std::runtime_error(
    "No scalar/list Float32 or Float64 column found in Parquet schema. "
    "Use --column to specify one.");
}

/// Reject an Arrow array that carries nulls.
///
/// @details Audit 2026-09-02 A3: neither Parquet reader inspected nulls at all
/// (`null_count`/`IsNull` appeared zero times in both). A null list cell
/// produced a wrong series and a null element produced whatever bytes the
/// values buffer happened to hold, straight into the DTW distances.
/// `arrow_c_data.cpp` already rejects both; this matches its policy and its
/// remedy wording. Exactly ONE `null_count()` read per array — i.e. per chunk,
/// per values buffer — never a per-element probe inside a copy loop.
inline void require_no_nulls(const arrow::Array &array, const char *what)
{
  const std::int64_t nulls = array.null_count();
  if (nulls != 0)
    throw std::runtime_error(
      std::string("Parquet reader: ") + what + " contains "
      + std::to_string(nulls)
      + " null(s) (drop or fill nulls before clustering).");
}

/// Validate one list cell's [start, end) against the values buffer length.
/// Offsets come from the file and are used to index a mapped buffer directly.
inline void require_list_range(std::int64_t start, std::int64_t end,
                               std::int64_t values_length)
{
  if (start < 0 || end < start || end > values_length)
    throw std::runtime_error(
      "Parquet reader: list offset [" + std::to_string(start) + ", "
      + std::to_string(end) + ") is outside the values buffer [0, "
      + std::to_string(values_length) + ")");
}

/// Copy `count` values starting at `start` from a Float64/Float32 Arrow array
/// into `dst`, converting to the destination element type.
///
/// @details One type dispatch per array, never per element. This single
/// definition replaces the six near-identical extractor bodies that each
/// re-implemented the Float32/Float64 branch (audit 2026-09-02, section C).
/// The caller has already validated the range and the null count.
template <typename T>
inline void copy_arrow_numeric(const arrow::Array &values, std::int64_t start,
                               std::size_t count, T *dst)
{
  if (values.type_id() == arrow::Type::DOUBLE) {
    const double *raw =
      static_cast<const arrow::DoubleArray &>(values).raw_values() + start;
    for (std::size_t j = 0; j < count; ++j) dst[j] = static_cast<T>(raw[j]);
  } else if (values.type_id() == arrow::Type::FLOAT) {
    const float *raw =
      static_cast<const arrow::FloatArray &>(values).raw_values() + start;
    for (std::size_t j = 0; j < count; ++j) dst[j] = static_cast<T>(raw[j]);
  } else {
    throw std::runtime_error(
      "Parquet reader: value type must be Float64 or Float32, got "
      + values.type()->ToString());
  }
}

} // namespace dtwc::io::detail

#endif // DTWC_HAS_PARQUET
