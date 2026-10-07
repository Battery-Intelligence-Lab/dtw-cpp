/**
 * @file parquet_schema.hpp
 * @brief Where a Parquet file's series are: the one column rule of every reader (read_data.hpp states it).
 */

#pragma once

#ifdef DTWC_HAS_PARQUET

#include "../base/error.hpp"

#include <arrow/api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::io::detail {

/// A top-level column: its Arrow field, its first Parquet leaf, its type.
struct ParquetColumn
{
  int arrow_field_index;
  int parquet_leaf_index;
  std::shared_ptr<arrow::DataType> type;
};

/// Where a file's series are: one scalar column as one series (OneSeries), a list column as one series per row
/// (List), or each row's values in its Float32/Float64 columns as one series (Rows). The first string column names
/// the rows of the last two.
struct ParquetSeriesLayout
{
  enum class Kind { OneSeries, List, Rows };
  Kind kind = Kind::OneSeries;
  std::vector<ParquetColumn> samples; ///< the column read, or (Rows) each row's sample columns in file order
  std::optional<ParquetColumn> names; ///< the first string column (List and Rows)
};

inline int parquet_leaf_count(const std::shared_ptr<arrow::DataType> &type)
{
  const auto &fields = type->fields();
  if (fields.empty()) return 1;
  int count = 0;
  for (const auto &field : fields) count += parquet_leaf_count(field->type());
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

inline bool is_parquet_float(const arrow::DataType &type)
{
  return type.id() == arrow::Type::FLOAT || type.id() == arrow::Type::DOUBLE;
}

inline bool is_parquet_string(const arrow::DataType &type)
{
  return type.id() == arrow::Type::STRING || type.id() == arrow::Type::LARGE_STRING;
}

inline bool is_parquet_series_column(
  const std::shared_ptr<arrow::DataType> &type)
{
  return is_parquet_float(*parquet_series_value_type(type));
}

/**
 * @brief Where a file's series are, with identical eager and streaming rules.
 *
 * `skip_cols` drops the file's first columns, as it drops a CSV row's first fields: a dropped column is neither read
 * nor a name. `column_name` reads that column: a list is one series per row, a scalar one series. Otherwise the first
 * Float32/Float64 column or list of them decides: a list is one series per row; a scalar is the one series when it is
 * the file's only Float32/Float64 column, and otherwise each row is a series of the row's values in the file's
 * columns, string columns aside, every one of which must then be Float32/Float64 (an IOError names the first that is
 * not). The first string column names the rows of a list or of the rows.
 */
inline ParquetSeriesLayout resolve_parquet_layout(
  const std::shared_ptr<arrow::Schema> &schema,
  const std::string &column_name, std::int64_t skip_cols = 0)
{
  std::vector<ParquetColumn> columns; // those skip_cols leaves
  int leaf_index = 0;
  for (int index = 0; index < schema->num_fields(); ++index) {
    const auto &type = schema->field(index)->type();
    if (index >= skip_cols) columns.push_back({ index, leaf_index, type });
    leaf_index += parquet_leaf_count(type);
  }
  const auto name_of = [&](const ParquetColumn &column) -> const std::string & {
    return schema->field(column.arrow_field_index)->name();
  };
  const std::string after_skip = skip_cols > 0
    ? " after the " + std::to_string(skip_cols) + " columns --skip-cols drops" : "";

  ParquetSeriesLayout layout;
  const auto first_string = std::find_if(columns.begin(), columns.end(),
                                         [](const auto &column) { return is_parquet_string(*column.type); });
  if (first_string != columns.end()) layout.names = *first_string;

  if (!column_name.empty()) {
    const auto named = std::find_if(columns.begin(), columns.end(),
                                    [&](const auto &column) { return name_of(column) == column_name; });
    if (named == columns.end())
      throw dtwc::InvalidInput("Column '" + column_name + "' not found in Parquet schema" + after_skip);
    if (!is_parquet_series_column(named->type))
      throw dtwc::IOError(
        "Parquet column '" + column_name +
        "' must be Float32, Float64, List<Float32/Float64>, or "
        "LargeList<Float32/Float64>");
    layout.samples = { *named };
  } else {
    const auto first = std::find_if(columns.begin(), columns.end(),
                                    [](const auto &column) { return is_parquet_series_column(column.type); });
    if (first == columns.end())
      throw dtwc::IOError(
        "No scalar/list Float32 or Float64 column found in Parquet schema" + after_skip
        + ". Use --column to specify one.");
    const auto scalars = std::count_if(columns.begin(), columns.end(),
                                       [](const auto &column) { return is_parquet_float(*column.type); });
    if (is_parquet_float(*first->type) && scalars > 1) {
      layout.kind = ParquetSeriesLayout::Kind::Rows;
      for (const auto &column : columns) {
        if (is_parquet_string(*column.type)) continue;
        if (!is_parquet_float(*column.type))
          throw dtwc::IOError(
            "Parquet column '" + name_of(column) + "' is " + column.type->ToString()
            + ", neither Float32/Float64 nor Utf8/LargeUtf8, so it cannot be a sample of the series each row "
              "holds; drop the leading columns with --skip-cols (skip_cols), or read one column with --column.");
        layout.samples.push_back(column);
      }
      return layout;
    }
    layout.samples = { *first };
  }
  if (is_parquet_float(*layout.samples.front().type)) {
    layout.kind = ParquetSeriesLayout::Kind::OneSeries; // named by its file
    layout.names.reset();
  } else {
    layout.kind = ParquetSeriesLayout::Kind::List;
  }
  return layout;
}

/// Reject an Arrow array that carries nulls.
///
/// @details Neither Parquet reader used to inspect nulls at all
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
    throw dtwc::InvalidInput(
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
    throw dtwc::IOError(
      "Parquet reader: list offset [" + std::to_string(start) + ", "
      + std::to_string(end) + ") is outside the values buffer [0, "
      + std::to_string(values_length) + ")");
}

/// Copy `count` values starting at `start` from a Float64/Float32 Arrow array
/// into `dst`, converting to the destination element type.
///
/// @details One type dispatch per array, never per element. This single
/// definition replaces the six near-identical extractor bodies that each
/// re-implemented the Float32/Float64 branch.
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
    throw dtwc::IOError(
      "Parquet reader: value type must be Float64 or Float32, got "
      + values.type()->ToString());
  }
}

} // namespace dtwc::io::detail

#endif // DTWC_HAS_PARQUET
