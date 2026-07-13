/**
 * @file parquet_schema.hpp
 * @brief Shared Parquet time-series column selection.
 */

#pragma once

#ifdef DTWC_HAS_PARQUET

#include <arrow/api.h>

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

} // namespace dtwc::io::detail

#endif // DTWC_HAS_PARQUET
