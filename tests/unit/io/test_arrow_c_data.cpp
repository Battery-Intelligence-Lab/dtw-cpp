/**
 * @file test_arrow_c_data.cpp
 * @brief Gate for Arrow C Data interface ingest (dtwc::io::data_from_arrow).
 *
 * Producer side uses the vendored nanoarrow builder (an independent code path
 * from the reader) to synthesise ArrowSchema/ArrowArray structs, exactly as a
 * real producer (polars/DuckDB) hands them across `__arrow_c_array__`. The
 * end-to-end independent-producer check (polars, no pyarrow) lives in the Python
 * suite; this gate pins the reader logic + wiring in C++.
 *
 * @author Claude Opus 4.8
 * @date 09 Jul 2026
 */

#include "io/arrow_c_data.hpp"
#include "error.hpp"

#include "nanoarrow/nanoarrow.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using dtwc::data_t;

namespace {

// Build a (large_)list<floating> ArrowSchema/ArrowArray from series. list_type is
// NANOARROW_TYPE_LIST or NANOARROW_TYPE_LARGE_LIST; value_type FLOAT or DOUBLE.
// null_rows marks whole series appended as null. Caller releases both structs.
void build_list(const std::vector<std::vector<double>> &series, ArrowType list_type,
                ArrowType value_type, ArrowSchema *schema, ArrowArray *array,
                const std::vector<int64_t> &null_rows = {})
{
  ArrowSchemaInit(schema);
  REQUIRE(ArrowSchemaSetType(schema, list_type) == NANOARROW_OK);
  REQUIRE(ArrowSchemaSetType(schema->children[0], value_type) == NANOARROW_OK);

  ArrowError error;
  REQUIRE(ArrowArrayInitFromSchema(array, schema, &error) == NANOARROW_OK);
  REQUIRE(ArrowArrayStartAppending(array) == NANOARROW_OK);
  ArrowArray *child = array->children[0];

  for (int64_t i = 0; i < static_cast<int64_t>(series.size()); ++i) {
    bool is_null = false;
    for (int64_t nr : null_rows)
      if (nr == i) is_null = true;
    if (is_null) {
      REQUIRE(ArrowArrayAppendNull(array, 1) == NANOARROW_OK);
      continue;
    }
    for (double v : series[static_cast<size_t>(i)])
      REQUIRE(ArrowArrayAppendDouble(child, v) == NANOARROW_OK);
    REQUIRE(ArrowArrayFinishElement(array) == NANOARROW_OK);
  }
  REQUIRE(ArrowArrayFinishBuildingDefault(array, &error) == NANOARROW_OK);
}

void release_both(ArrowSchema *schema, ArrowArray *array)
{
  if (array->release) ArrowArrayRelease(array);
  if (schema->release) ArrowSchemaRelease(schema);
}

} // namespace

TEST_CASE("Arrow ingest: large_list<double> round-trips to Data", "[arrow][oracle]")
{
  const std::vector<std::vector<double>> series{
    { 1.0, 2.0, 3.0 }, { -4.5, 6.25 }, { 0.0, 0.0, 0.0, 7.0 }
  };
  ArrowSchema schema;
  ArrowArray array;
  build_list(series, NANOARROW_TYPE_LARGE_LIST, NANOARROW_TYPE_DOUBLE, &schema, &array);

  dtwc::Data data = dtwc::io::data_from_arrow(&schema, &array);
  release_both(&schema, &array);

  REQUIRE(data.size() == 3);
  REQUIRE(data.ndim == 1);
  REQUIRE(data.p_vec == series); // exact copy, no reordering, no precision loss
  REQUIRE(data.p_names[0] == "series_0");
  REQUIRE(data.p_names[2] == "series_2");
}

TEST_CASE("Arrow ingest: list<float32> converts to double", "[arrow][float32]")
{
  const std::vector<std::vector<double>> series{ { 1.5, 2.5 }, { 3.5 } };
  ArrowSchema schema;
  ArrowArray array;
  build_list(series, NANOARROW_TYPE_LIST, NANOARROW_TYPE_FLOAT, &schema, &array);

  dtwc::Data data = dtwc::io::data_from_arrow(&schema, &array);
  release_both(&schema, &array);

  REQUIRE(data.size() == 2);
  // 1.5/2.5/3.5 are exactly representable in float32, so equality is exact.
  REQUIRE(data.p_vec == series);
}

TEST_CASE("Arrow ingest: caller names override defaults", "[arrow][names]")
{
  const std::vector<std::vector<double>> series{ { 1.0 }, { 2.0 } };
  ArrowSchema schema;
  ArrowArray array;
  build_list(series, NANOARROW_TYPE_LARGE_LIST, NANOARROW_TYPE_DOUBLE, &schema, &array);

  dtwc::Data data = dtwc::io::data_from_arrow(&schema, &array, { "alpha", "beta" });
  release_both(&schema, &array);

  REQUIRE(data.p_names == std::vector<std::string>{ "alpha", "beta" });
}

TEST_CASE("Arrow ingest: empty list yields empty Data", "[arrow][empty]")
{
  ArrowSchema schema;
  ArrowArray array;
  build_list({}, NANOARROW_TYPE_LARGE_LIST, NANOARROW_TYPE_DOUBLE, &schema, &array);

  dtwc::Data data = dtwc::io::data_from_arrow(&schema, &array);
  release_both(&schema, &array);

  REQUIRE(data.size() == 0);
}

TEST_CASE("Arrow ingest: null series is rejected", "[arrow][reject]")
{
  const std::vector<std::vector<double>> series{ { 1.0 }, { 2.0 }, { 3.0 } };
  ArrowSchema schema;
  ArrowArray array;
  build_list(series, NANOARROW_TYPE_LARGE_LIST, NANOARROW_TYPE_DOUBLE, &schema, &array,
             /*null_rows=*/{ 1 });

  REQUIRE_THROWS_AS(dtwc::io::data_from_arrow(&schema, &array), dtwc::InvalidInput);
  release_both(&schema, &array);
}

TEST_CASE("Arrow ingest: non-floating child is rejected", "[arrow][reject]")
{
  // Build a list<int64> by hand (build_list only does floating); reject expected.
  ArrowSchema schema;
  ArrowArray array;
  ArrowSchemaInit(&schema);
  REQUIRE(ArrowSchemaSetType(&schema, NANOARROW_TYPE_LARGE_LIST) == NANOARROW_OK);
  REQUIRE(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT64) == NANOARROW_OK);
  ArrowError error;
  REQUIRE(ArrowArrayInitFromSchema(&array, &schema, &error) == NANOARROW_OK);
  REQUIRE(ArrowArrayStartAppending(&array) == NANOARROW_OK);
  REQUIRE(ArrowArrayAppendInt(array.children[0], 42) == NANOARROW_OK);
  REQUIRE(ArrowArrayFinishElement(&array) == NANOARROW_OK);
  REQUIRE(ArrowArrayFinishBuildingDefault(&array, &error) == NANOARROW_OK);

  REQUIRE_THROWS_AS(dtwc::io::data_from_arrow(&schema, &array), dtwc::InvalidInput);
  release_both(&schema, &array);
}

TEST_CASE("Arrow ingest: top-level non-list is rejected", "[arrow][reject]")
{
  ArrowSchema schema;
  ArrowArray array;
  REQUIRE(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_DOUBLE) == NANOARROW_OK);
  ArrowError error;
  REQUIRE(ArrowArrayInitFromSchema(&array, &schema, &error) == NANOARROW_OK);
  REQUIRE(ArrowArrayStartAppending(&array) == NANOARROW_OK);
  REQUIRE(ArrowArrayAppendDouble(&array, 1.0) == NANOARROW_OK);
  REQUIRE(ArrowArrayFinishBuildingDefault(&array, &error) == NANOARROW_OK);

  REQUIRE_THROWS_AS(dtwc::io::data_from_arrow(&schema, &array), dtwc::InvalidInput);
  release_both(&schema, &array);
}
