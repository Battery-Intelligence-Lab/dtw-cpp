/**
 * @file f13_poison_parquet_writer.cc
 * @brief Build-local Parquet fixture generator for the F13 streamed gate.
 */

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

template <typename ValueBuilder, typename Value>
int write_fixture(
  const std::string &path, const std::shared_ptr<arrow::DataType> &value_type)
{
  auto *const pool = arrow::default_memory_pool();
  auto values = std::make_shared<ValueBuilder>(pool);
  arrow::ListBuilder lists(pool, values);
  const Value maximum = std::numeric_limits<Value>::max();

  for (int row = 0; row < 129; ++row) {
    if (const auto status = lists.Append(); !status.ok()) {
      std::cerr << "F13 fixture list append failed: "
                << status.ToString() << '\n';
      return 1;
    }
    const Value value = row == 0 ? -maximum : maximum;
    if (const auto status = values->Append(value); !status.ok()) {
      std::cerr << "F13 fixture value append failed: "
                << status.ToString() << '\n';
      return 1;
    }
  }

  std::shared_ptr<arrow::Array> array;
  if (const auto status = lists.Finish(&array); !status.ok()) {
    std::cerr << "F13 fixture finish failed: "
              << status.ToString() << '\n';
    return 1;
  }

  const auto list_type = arrow::list(value_type);
  const auto schema = arrow::schema({arrow::field("series", list_type)});
  const auto table = arrow::Table::Make(schema, {std::move(array)});
  if (const auto status = table->ValidateFull(); !status.ok()) {
    std::cerr << "F13 fixture table validation failed: "
              << status.ToString() << '\n';
    return 1;
  }

  auto output_result = arrow::io::FileOutputStream::Open(path);
  if (!output_result.ok()) {
    std::cerr << "F13 fixture open failed: "
              << output_result.status().ToString() << '\n';
    return 1;
  }
  const auto output = std::move(output_result).ValueOrDie();
  if (const auto status =
        parquet::arrow::WriteTable(*table, pool, output, 65);
      !status.ok()) {
    std::cerr << "F13 fixture write failed: "
              << status.ToString() << '\n';
    return 1;
  }
  if (const auto status = output->Close(); !status.ok()) {
    std::cerr << "F13 fixture close failed: "
              << status.ToString() << '\n';
    return 1;
  }

  return 0;
}

} // namespace

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "usage: f13_poison_parquet_writer PATH f64|f32\n";
    return 2;
  }

  const std::string path(argv[1]);
  const std::string precision(argv[2]);
  int result = 2;
  if (precision == "f64") {
    result = write_fixture<arrow::DoubleBuilder, double>(
      path, arrow::float64());
  } else if (precision == "f32") {
    result = write_fixture<arrow::FloatBuilder, float>(
      path, arrow::float32());
  } else {
    std::cerr << "F13 fixture precision must be f64 or f32.\n";
  }

  if (result == 0) {
    std::cout << "F13_POISON_PARQUET precision=" << precision
              << " rows=129 row_groups=2\n";
  }
  return result;
}
