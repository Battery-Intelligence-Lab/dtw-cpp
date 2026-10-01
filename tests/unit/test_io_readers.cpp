/**
 * @file test_io_readers.cpp
 * @brief The Arrow IPC and Parquet routes of dtwc::read_data, and the Parquet
 *        reader's streaming entries: names, values, nulls, offsets, metadata.
 *
 * Each test builds an Arrow/Parquet fixture with the Arrow C++ API in a temp
 * directory, then exercises the reader. The tests are gated on DTWC_HAS_ARROW /
 * DTWC_HAS_PARQUET exactly like the readers themselves (see dtwc/CMakeLists.txt);
 * when Arrow is not built the file registers a single skipped case.
 *
 * @date 2026-07-07
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <cstring>
#include <utility>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../support/scratch_directory.hpp"

#if !defined(DTWC_HAS_ARROW)

TEST_CASE("I/O reader hardening tests skipped", "[io]")
{
  SKIP("DTWC_HAS_ARROW not defined — Arrow/Parquet readers not built");
}

#else

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/util/key_value_metadata.h>

#include <base/error.hpp>
#include <fileOperations.hpp>
#include <io/read_data.hpp>

#if defined(DTWC_HAS_PARQUET)
#include <parquet/arrow/writer.h>
#include <algorithms/fast_clara.hpp>
#include <io/parquet_chunk_reader.hpp>
#include <Problem.hpp>
#endif

using Catch::Matchers::WithinAbs;

namespace {

template <class T>
T unwrap(arrow::Result<T> r)
{
  REQUIRE(r.ok());
  return r.ValueOrDie(); // reference, copied out (cheap for shared_ptr)
}

/// The directory of every file this binary writes: unique to the process, removed at exit.
std::filesystem::path tmpdir()
{
  static const dtwc::test_support::ScratchDirectory dir{ "io_reader_test" };
  return dir.path;
}

// Build a List<Float64> array from ragged series.
std::shared_ptr<arrow::Array> make_list_f64(const std::vector<std::vector<double>> &series)
{
  auto pool = arrow::default_memory_pool();
  auto vb = std::make_shared<arrow::DoubleBuilder>(pool);
  arrow::ListBuilder lb(pool, vb);
  for (const auto &s : series) {
    REQUIRE(lb.Append().ok());
    REQUIRE(vb->AppendValues(s).ok());
  }
  std::shared_ptr<arrow::Array> out;
  REQUIRE(lb.Finish(&out).ok());
  return out;
}

// Write an Arrow IPC (Feather v2) file, one record batch per column set.
void write_ipc(const std::filesystem::path &path,
               const std::shared_ptr<arrow::Schema> &schema,
               const std::vector<std::vector<std::shared_ptr<arrow::Array>>> &batches)
{
  auto out = unwrap(arrow::io::FileOutputStream::Open(dtwc::path_to_utf8(path))); // Arrow paths are UTF-8
  auto writer = unwrap(arrow::ipc::MakeFileWriter(out, schema));
  for (const auto &columns : batches)
    REQUIRE(writer->WriteRecordBatch(*arrow::RecordBatch::Make(schema, columns.front()->length(), columns)).ok());
  REQUIRE(writer->Close().ok());
  REQUIRE(out->Close().ok());
}

// A list column holding a null cell, and one holding a null inside a cell.
std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>> null_lists()
{
  auto pool = arrow::default_memory_pool();
  std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>> out;
  for (const bool null_cell : { true, false }) {
    auto vb = std::make_shared<arrow::DoubleBuilder>(pool);
    arrow::ListBuilder lb(pool, vb);
    REQUIRE(lb.Append().ok());
    if (null_cell) {
      REQUIRE(vb->AppendValues(std::vector<double>{ 1.0, 2.0 }).ok());
      REQUIRE(lb.AppendNull().ok());
    } else {
      REQUIRE(vb->Append(1.0).ok());
      REQUIRE(vb->AppendNull().ok());
      REQUIRE(vb->Append(3.0).ok());
    }
    std::shared_ptr<arrow::Array> arr;
    REQUIRE(lb.Finish(&arr).ok());
    out.emplace_back(null_cell ? "null cell" : "null element", arr);
  }
  return out;
}

} // namespace

// --------------------- Arrow IPC (read_data, via a C stream) ---------------------

TEMPLATE_TEST_CASE("ArrowIPC: every record batch, its names and ndim reach the Data", "[io][arrow]",
                   arrow::StringBuilder, arrow::LargeStringBuilder)
{
  // The C stream path dropped the 'name' column for series_<i>, and the IPC
  // reader refused a second record batch. LargeUtf8 is Polars' default string type.
  const auto names = [](const std::vector<std::string> &values) {
    TestType builder;
    REQUIRE(builder.AppendValues(values).ok());
    std::shared_ptr<arrow::Array> out;
    REQUIRE(builder.Finish(&out).ok());
    return out;
  };
  const auto first_names = names({ "first", "second" });
  const auto first_data = make_list_f64({ { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0 } });
  const auto schema = arrow::schema(
    { arrow::field("name", first_names->type()), arrow::field("data", first_data->type()) },
    arrow::key_value_metadata({ "ndim" }, { "2" }));
  const auto tmp = tmpdir() / (first_names->type()->ToString() + "_batches.arrow");
  write_ipc(tmp, schema, { { first_names, first_data }, { names({ "third" }), make_list_f64({ { 7.0, 8.0 } }) } });

  const auto data = dtwc::read_data(tmp);
  CHECK(data.ndim == 2);
  CHECK(data.series_length(0) == 2); // 4 flat values / ndim 2
  CHECK(data.p_names == std::vector<std::string>{ "first", "second", "third" });
  CHECK(data.p_vec == std::vector<std::vector<double>>{ { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
  std::filesystem::remove(tmp);
}

TEST_CASE("ArrowIPC: out-of-bounds list offset rejected", "[io][arrow][security]")
{
  // A crafted file: int32 offsets [0, 6, 3] over 5 values. The IPC writer keeps
  // values [offsets[0], offsets[length]) = 3 of them, so the middle offset, 6,
  // points past the mapped values; reading it would run off the buffer.
  auto pool = arrow::default_memory_pool();
  arrow::DoubleBuilder vb(pool);
  REQUIRE(vb.AppendValues(std::vector<double>{ 1.0, 2.0, 3.0, 4.0, 5.0 }).ok());
  std::shared_ptr<arrow::Array> values;
  REQUIRE(vb.Finish(&values).ok());
  auto offsets = arrow::Buffer::FromVector(std::vector<int32_t>{ 0, 6, 3 });
  auto list_type = arrow::list(arrow::float64());
  auto list = std::static_pointer_cast<arrow::Array>(
    std::make_shared<arrow::ListArray>(list_type, /*length=*/2, offsets, values));
  auto tmp = tmpdir() / "crafted.arrow"; // a name the message matcher cannot match
  write_ipc(tmp, arrow::schema({ arrow::field("data", list_type) }), { { list } });

  CHECK_THROWS_AS(dtwc::read_data(tmp), dtwc::IOError);
  CHECK_THROWS_WITH(dtwc::read_data(tmp), Catch::Matchers::ContainsSubstring("is outside the values"));
  std::filesystem::remove(tmp);
}

TEST_CASE("ArrowIPC: a column of the wrong type is rejected by type", "[io][arrow]")
{
  // A bad Arrow type is an IOError naming the file and the column: a scalar
  // 'data' column, and an Int64 'name' column, which must not silently become
  // series_<i>.
  arrow::DoubleBuilder scalar_builder;
  REQUIRE(scalar_builder.AppendValues(std::vector<double>{ 1.0, 2.0 }).ok());
  std::shared_ptr<arrow::Array> scalar;
  REQUIRE(scalar_builder.Finish(&scalar).ok());
  const auto scalar_file = tmpdir() / "scalar_data.arrow";
  write_ipc(scalar_file, arrow::schema({ arrow::field("data", scalar->type()) }), { { scalar } });
  REQUIRE_THROWS_AS(dtwc::read_data(scalar_file), dtwc::IOError);
  REQUIRE_THROWS_WITH(dtwc::read_data(scalar_file),
                      Catch::Matchers::ContainsSubstring("'data'")
                        && Catch::Matchers::ContainsSubstring("scalar_data.arrow"));
  std::filesystem::remove(scalar_file);

  auto data = make_list_f64({ { 1.0, 2.0 }, { 3.0, 4.0 } });
  arrow::Int64Builder nb;
  REQUIRE(nb.AppendValues(std::vector<int64_t>{ 7, 8 }).ok());
  std::shared_ptr<arrow::Array> names;
  REQUIRE(nb.Finish(&names).ok());
  auto schema = arrow::schema(
    { arrow::field("data", data->type()), arrow::field("name", names->type()) });
  auto tmp = tmpdir() / "int64_names.arrow";
  write_ipc(tmp, schema, { { data, names } });

  REQUIRE_THROWS_AS(dtwc::read_data(tmp), dtwc::IOError);
  REQUIRE_THROWS_WITH(dtwc::read_data(tmp),
                      Catch::Matchers::ContainsSubstring("'name'")
                        && Catch::Matchers::ContainsSubstring("int64_names.arrow"));
  std::filesystem::remove(tmp);
}

TEST_CASE("ArrowIPC: ndim metadata must be a whole positive integer", "[io][arrow]")
{
  // An unguarded std::stoul let "abc" escape as a context-free
  // std::invalid_argument, wrapped "-1" to 2^64 - 1 and read "2x" as 2; 0 would
  // divide every series length by zero. Each is an IOError naming the file.
  for (const std::string text : { "0", "abc", "-1", "2x", "", "99999999999999999999999" }) {
    INFO("ndim metadata '" << text << "'");
    auto arr = make_list_f64({ { 1.0, 2.0, 3.0, 4.0 } });
    auto meta = arrow::key_value_metadata({ "ndim" }, { text });
    auto schema = arrow::schema({ arrow::field("data", arr->type()) }, meta);
    auto tmp = tmpdir() / "ndim_text.arrow";
    write_ipc(tmp, schema, { { arr } });

    REQUIRE_THROWS_AS(dtwc::read_data(tmp), dtwc::IOError);
    REQUIRE_THROWS_WITH(dtwc::read_data(tmp),
                        Catch::Matchers::ContainsSubstring("ndim")
                          && Catch::Matchers::ContainsSubstring("ndim_text.arrow"));
    std::filesystem::remove(tmp);
  }
}

TEST_CASE("ArrowIPC: a null series or a null value is rejected", "[io][arrow][security]")
{
  // The IPC reader never looked at the validity bitmap: a null slot became
  // whatever bytes the values buffer held, straight into the distances.
  for (const auto &[label, arr] : null_lists()) {
    CAPTURE(label);
    auto tmp = tmpdir() / ("ipc_" + label.substr(5) + ".arrow");
    write_ipc(tmp, arrow::schema({ arrow::field("data", arr->type()) }), { { arr } });
    CHECK_THROWS_AS(dtwc::read_data(tmp), dtwc::InvalidInput);
    CHECK_THROWS_WITH(dtwc::read_data(tmp), Catch::Matchers::ContainsSubstring("null"));
    std::filesystem::remove(tmp);
  }
}

// -------------------------- Parquet reader ----------------------------

#if defined(DTWC_HAS_PARQUET)

namespace {

void write_parquet(const std::filesystem::path &path,
                   const std::shared_ptr<arrow::Table> &table)
{
  auto out = unwrap(arrow::io::FileOutputStream::Open(dtwc::path_to_utf8(path))); // Arrow paths are UTF-8
  REQUIRE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1024).ok());
  REQUIRE(out->Close().ok());
}

// Build a List<Float32> array from ragged series.
std::shared_ptr<arrow::Array> make_list_f32(const std::vector<std::vector<float>> &series)
{
  auto pool = arrow::default_memory_pool();
  auto vb = std::make_shared<arrow::FloatBuilder>(pool);
  arrow::ListBuilder lb(pool, vb);
  for (const auto &s : series) {
    REQUIRE(lb.Append().ok());
    REQUIRE(vb->AppendValues(s).ok());
  }
  std::shared_ptr<arrow::Array> out;
  REQUIRE(lb.Finish(&out).ok());
  return out;
}

} // namespace

TEST_CASE("Parquet chunk extractors reject corrupt list offsets",
          "[io][parquet][streaming][security]")
{
  arrow::DoubleBuilder values_builder(arrow::default_memory_pool());
  REQUIRE(values_builder.AppendValues(
    std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}).ok());
  std::shared_ptr<arrow::Array> values;
  REQUIRE(values_builder.Finish(&values).ok());

  auto offsets = arrow::Buffer::FromVector(
    std::vector<int32_t>{0, 6, 3});
  auto type = arrow::list(arrow::float64());
  auto list = std::static_pointer_cast<arrow::Array>(
    std::make_shared<arrow::ListArray>(
      type, /*length=*/2, offsets, values));
  auto column = std::make_shared<arrow::ChunkedArray>(list);

  std::vector<std::vector<dtwc::data_t>> f64;
  std::vector<std::vector<float>> f32;
  CHECK_THROWS_WITH(
    dtwc::io::detail::extract_series_from_column(column, type, f64),
    Catch::Matchers::ContainsSubstring("outside the values buffer"));
  CHECK_THROWS_WITH(
    dtwc::io::detail::extract_series_from_column(column, type, f32),
    Catch::Matchers::ContainsSubstring("outside the values buffer"));
}

TEST_CASE("Parquet: scalar Float64 column still reads", "[io][parquet]")
{
  // Positive control: the common Float64 scalar path must survive the fix.
  auto pool = arrow::default_memory_pool();
  arrow::DoubleBuilder db(pool);
  REQUIRE(db.AppendValues(std::vector<double>{ 1.0, 2.0, 3.0 }).ok());
  std::shared_ptr<arrow::Array> arr;
  REQUIRE(db.Finish(&arr).ok());
  auto schema = arrow::schema({ arrow::field("v", arrow::float64()) });
  auto table = arrow::Table::Make(schema, { arr });
  auto tmp = tmpdir() / "scalar_f64.parquet";
  write_parquet(tmp, table);

  auto data = dtwc::read_data(tmp, 0, 0, '\0', "v");
  REQUIRE(data.size() == 1);
  CHECK(data.name(0) == "scalar_f64"); // one scalar column is one series, named by its file
  REQUIRE(data.p_vec[0].size() == 3);
  CHECK_THAT(data.p_vec[0][0], WithinAbs(1.0, 1e-12));
  CHECK_THAT(data.p_vec[0][2], WithinAbs(3.0, 1e-12));
  std::filesystem::remove(tmp);
}

TEST_CASE("Parquet chunk metadata preserves scalar-column series semantics",
          "[io][parquet][streaming]")
{
  auto pool = arrow::default_memory_pool();
  arrow::DoubleBuilder db(pool);
  REQUIRE(db.AppendValues(std::vector<double>{ 1.0, 2.0, 3.0, 4.0 }).ok());
  std::shared_ptr<arrow::Array> arr;
  REQUIRE(db.Finish(&arr).ok());
  auto schema = arrow::schema({ arrow::field("v", arrow::float64()) });
  auto tmp = tmpdir() / "scalar_chunk_contract.parquet";
  write_parquet(tmp, arrow::Table::Make(schema, { arr }));

  {
    dtwc::io::ParquetChunkReader reader(tmp, "v");
    CHECK(reader.total_rows() == 4);
    CHECK_FALSE(reader.is_list_layout());
    CHECK(reader.logical_series_count() == 1);
    CHECK(reader.estimated_resident_bytes(false) > 0);
  }

  dtwc::Problem settings_only{"scalar_stream_defense"};
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 1;
  options.sample_size = 1;
  options.ram_limit_bytes = 1;
  options.parquet_path = tmp;
  options.parquet_column = "v";
  CHECK_THROWS_WITH(
    dtwc::algorithms::fast_clara(settings_only, options),
    Catch::Matchers::ContainsSubstring("scalar column is one time series"));
  std::filesystem::remove(tmp);
}

TEST_CASE("Parquet: scalar Float32 column converted to double", "[io][parquet]")
{
  // Bug 1 (scalar path): the eager reader cast every scalar chunk to
  // DoubleArray; find_column accepts Float32 columns, so an f32 file was read as
  // doubles (garbage + OOB). The fix adds the FLOAT branch (convert to data_t).
  // PRE-FIX the value assertions see garbage (or crash on the OOB read);
  // POST-FIX they see the exact converted values.
  auto pool = arrow::default_memory_pool();
  arrow::FloatBuilder fb(pool);
  REQUIRE(fb.AppendValues(std::vector<float>{ 1.5f, 2.5f, 3.5f, 4.5f }).ok());
  std::shared_ptr<arrow::Array> arr;
  REQUIRE(fb.Finish(&arr).ok());
  auto schema = arrow::schema({ arrow::field("v", arrow::float32()) });
  auto table = arrow::Table::Make(schema, { arr });
  auto tmp = tmpdir() / "scalar_f32.parquet";
  write_parquet(tmp, table);

  auto data = dtwc::read_data(tmp, 0, 0, '\0', "v");
  REQUIRE(data.size() == 1);
  const auto &s = data.p_vec[0];
  REQUIRE(s.size() == 4);
  CHECK_THAT(s[0], WithinAbs(1.5, 1e-6));
  CHECK_THAT(s[1], WithinAbs(2.5, 1e-6));
  CHECK_THAT(s[2], WithinAbs(3.5, 1e-6));
  CHECK_THAT(s[3], WithinAbs(4.5, 1e-6));
  std::filesystem::remove(tmp);
}

TEST_CASE("Parquet: List<Float32> column converted to double", "[io][parquet]")
{
  // Bug 1 (list path): list values were cast to DoubleArray with no FLOAT
  // branch. PRE-FIX garbage/OOB; POST-FIX exact conversion.
  auto arr = make_list_f32({ { 1.5f, 2.5f, 3.5f }, { 4.5f, 5.5f } });
  auto schema = arrow::schema({ arrow::field("data", arr->type()) });
  auto table = arrow::Table::Make(schema, { arr });
  auto tmp = tmpdir() / "list_f32.parquet";
  write_parquet(tmp, table);

  auto data = dtwc::read_data(tmp, 0, 0, '\0', "data");
  REQUIRE(data.size() == 2);
  REQUIRE(data.p_vec[0].size() == 3);
  REQUIRE(data.p_vec[1].size() == 2);
  CHECK_THAT(data.p_vec[0][0], WithinAbs(1.5, 1e-6));
  CHECK_THAT(data.p_vec[0][2], WithinAbs(3.5, 1e-6));
  CHECK_THAT(data.p_vec[1][1], WithinAbs(5.5, 1e-6));

  auto auto_detected = dtwc::read_data(tmp);
  REQUIRE(auto_detected.size() == data.size());
  CHECK(auto_detected.p_vec == data.p_vec);

  {
    dtwc::io::ParquetChunkReader reader(tmp); // same shared auto-detection
    CHECK(reader.is_list_layout());
    CHECK(reader.logical_series_count() == 2);
    CHECK(reader.estimated_resident_bytes(false)
          >= reader.estimated_resident_bytes(true));
    CHECK(reader.estimated_materialization_peak_bytes(true)
          >= reader.estimated_resident_bytes(false));
    CHECK_THROWS_WITH(
      reader.row_groups_per_batch(0, true),
      Catch::Matchers::ContainsSubstring("one row group needs"));
    CHECK(reader.row_groups_per_batch(
            std::numeric_limits<size_t>::max(), true) == 1);
    auto sparse = reader.read_rows({1, 0});
    REQUIRE(sparse.size() == 2);
    CHECK(sparse.name(0) == "series_1");
    CHECK(sparse.name(1) == "series_0");
    auto sparse_f32 = reader.read_rows<float>({1, 0});
    REQUIRE(sparse_f32.is_f32());
    CHECK(sparse_f32.name(0) == "series_1");
    CHECK_THAT(sparse_f32.series_f32(1)[2], WithinAbs(3.5f, 1e-6f));
    auto duplicates = reader.read_rows({1, 1});
    REQUIRE(duplicates.size() == 2);
    CHECK(duplicates.p_vec[0] == duplicates.p_vec[1]);
  }
  std::filesystem::remove(tmp);
}

TEST_CASE("Parquet metadata estimate uses decoded list values, not dictionary pages",
          "[io][parquet][streaming][ram]")
{
  constexpr size_t rows = 128;
  constexpr size_t values_per_row = 256;
  std::vector<std::vector<double>> series(
    rows, std::vector<double>(values_per_row, 7.0));
  auto arr = make_list_f64(series);
  auto schema = arrow::schema({ arrow::field("series", arr->type()) });
  auto tmp = tmpdir() / "dictionary_compressed_lists.parquet";
  write_parquet(tmp, arrow::Table::Make(schema, { arr }));

  {
    dtwc::io::ParquetChunkReader reader(tmp, "series");
    const size_t decoded_values = rows * values_per_row * sizeof(double);
    CHECK(reader.logical_series_count() == static_cast<int64_t>(rows));
    CHECK(reader.estimated_resident_bytes(false) >= decoded_values);
    CHECK(reader.estimated_materialization_peak_bytes(false)
          >= 2 * decoded_values);
    const size_t metadata_only_boundary =
      reader.estimated_materialization_peak_bytes(false)
      + sizeof(std::vector<dtwc::data_t>) + sizeof(std::string);
    CHECK_THROWS_WITH(
      reader.read_rows({0}, metadata_only_boundary),
      Catch::Matchers::ContainsSubstring("copying selected series"));
  }
  std::filesystem::remove(tmp);
}

TEST_CASE("Parquet series selection maps Arrow fields to physical leaf columns",
          "[io][parquet][streaming][schema]")
{
  arrow::DoubleBuilder a_builder(arrow::default_memory_pool());
  arrow::DoubleBuilder b_builder(arrow::default_memory_pool());
  REQUIRE(a_builder.AppendValues(std::vector<double>{11.0, 12.0}).ok());
  REQUIRE(b_builder.AppendValues(std::vector<double>{21.0, 22.0}).ok());
  std::shared_ptr<arrow::Array> a;
  std::shared_ptr<arrow::Array> b;
  REQUIRE(a_builder.Finish(&a).ok());
  REQUIRE(b_builder.Finish(&b).ok());
  const arrow::FieldVector metadata_fields{
    arrow::field("a", arrow::float64()),
    arrow::field("b", arrow::float64())};
  auto metadata = unwrap(arrow::StructArray::Make({a, b}, metadata_fields));
  auto series = make_list_f64({{1.0, 2.0}, {3.0, 4.0, 5.0}});
  auto schema = arrow::schema({
    arrow::field("metadata", metadata->type()),
    arrow::field("series", series->type())});
  auto tmp = tmpdir() / "preceding_multi_leaf.parquet";
  write_parquet(tmp, arrow::Table::Make(schema, {metadata, series}));

  auto eager = dtwc::read_data(tmp, 0, 0, '\0', "series");
  REQUIRE(eager.size() == 2);
  CHECK(eager.p_vec[0] == std::vector<double>{1.0, 2.0});
  CHECK(eager.p_vec[1] == std::vector<double>{3.0, 4.0, 5.0});
  auto eager_auto = dtwc::read_data(tmp);
  CHECK(eager_auto.p_vec == eager.p_vec);
  {
    dtwc::io::ParquetChunkReader reader(tmp, "series");
    CHECK(reader.logical_series_count() == 2);
    auto sparse = reader.read_rows({1});
    CHECK(sparse.p_vec[0] == eager.p_vec[1]);
  }
  std::filesystem::remove(tmp);
}

TEST_CASE("Parquet folder: the text reader's file order, unique names, UTF-8 stems",
          "[io][parquet]")
{
  // Each list file restarted its names at series_0, a scalar file was named
  // by its stem in the ANSI code page (and Arrow could not open the file),
  // and a hidden file was read as series.
  const auto folder = tmpdir() / "parquet_folder";
  std::filesystem::create_directories(folder);
  const auto write_list = [&](const std::string &stem, const std::vector<std::vector<double>> &rows) {
    const auto series = make_list_f64(rows);
    write_parquet(folder / (stem + ".parquet"),
                  arrow::Table::Make(arrow::schema({ arrow::field("series", series->type()) }), { series }));
  };
  write_list("b", { { 3.0 }, { 4.0 } });
  write_list("a", { { 1.0 }, { 2.0 } });
  write_list(".hidden", { { 9.0 } });
  arrow::DoubleBuilder scalar_builder;
  REQUIRE(scalar_builder.AppendValues(std::vector<double>{ 5.0, 6.0 }).ok());
  std::shared_ptr<arrow::Array> scalar;
  REQUIRE(scalar_builder.Finish(&scalar).ok());
  write_parquet(folder / dtwc::utf8_to_path("caf\xC3\xA9.parquet"),
                arrow::Table::Make(arrow::schema({ arrow::field("v", arrow::float64()) }), { scalar }));

  const auto data = dtwc::read_data(folder);
  CHECK(data.p_names
        == std::vector<std::string>{ "series_0", "series_1", "series_2", "series_3", "caf\xC3\xA9" });
  CHECK(data.p_vec == std::vector<std::vector<double>>{ { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0, 6.0 } });
  std::filesystem::remove_all(folder);
}

TEST_CASE("Parquet: a null in a scalar column is rejected, not read as garbage",
          "[io][parquet][security]")
{
  // Audit 2026-09-02 A3: neither Parquet reader looked at nulls. A null slot in
  // a scalar column carries whatever the values buffer happens to hold, and
  // that value went straight into the DTW distances. arrow_c_data.cpp already
  // rejects nulls; the Parquet readers must match. PRE-FIX both calls below
  // return successfully with a fabricated value.
  auto pool = arrow::default_memory_pool();
  arrow::DoubleBuilder db(pool);
  REQUIRE(db.Append(1.0).ok());
  REQUIRE(db.AppendNull().ok());
  REQUIRE(db.Append(3.0).ok());
  std::shared_ptr<arrow::Array> arr;
  REQUIRE(db.Finish(&arr).ok());
  auto schema = arrow::schema({ arrow::field("v", arrow::float64()) });
  auto tmp = tmpdir() / "scalar_null.parquet";
  write_parquet(tmp, arrow::Table::Make(schema, { arr }));

  CHECK_THROWS_WITH(dtwc::read_data(tmp, 0, 0, '\0', "v"),
                    Catch::Matchers::ContainsSubstring("null"));

  auto column = std::make_shared<arrow::ChunkedArray>(arr);
  std::vector<std::vector<dtwc::data_t>> f64;
  std::vector<std::vector<float>> f32;
  CHECK_THROWS_WITH(
    dtwc::io::detail::extract_series_from_column(column, arrow::float64(), f64),
    Catch::Matchers::ContainsSubstring("null"));
  CHECK_THROWS_WITH(
    dtwc::io::detail::extract_series_from_column(column, arrow::float64(), f32),
    Catch::Matchers::ContainsSubstring("null"));
  std::filesystem::remove(tmp);
}

TEST_CASE("Parquet: a null list cell or null list element is rejected",
          "[io][parquet][security]")
{
  // A null LIST cell used to yield a wrong (empty or shifted) series; a null
  // element inside an otherwise valid cell used to yield raw buffer bytes.
  for (const auto &[label, arr] : null_lists()) {
    CAPTURE(label);
    auto schema = arrow::schema({ arrow::field("series", arr->type()) });
    auto tmp = tmpdir() / ("list_" + label.substr(5) + ".parquet");
    write_parquet(tmp, arrow::Table::Make(schema, { arr }));
    CHECK_THROWS_WITH(dtwc::read_data(tmp, 0, 0, '\0', "series"),
                      Catch::Matchers::ContainsSubstring("null"));

    auto column = std::make_shared<arrow::ChunkedArray>(arr);
    std::vector<std::vector<dtwc::data_t>> f64;
    std::vector<std::vector<float>> f32;
    CHECK_THROWS_WITH(
      dtwc::io::detail::extract_series_from_column(column, arr->type(), f64),
      Catch::Matchers::ContainsSubstring("null"));
    CHECK_THROWS_WITH(
      dtwc::io::detail::extract_series_from_column(column, arr->type(), f32),
      Catch::Matchers::ContainsSubstring("null"));
    std::filesystem::remove(tmp);
  }
}

// The streamed nearest-medoid assignment runs in parallel over each chunk (more
// than 64 rows). The in-RAM route must give the same clustering bit for bit.
TEST_CASE("Parquet: streamed FastCLARA equals the in-RAM run",
          "[io][parquet][streaming]")
{
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (int i = 0; i < 150; ++i) {
    const double base = 10.0 * (i % 3) + 0.01 * i;
    series.push_back({ base, base + 1.0, base + 0.5, base + 2.0 });
    names.push_back("s" + std::to_string(i));
  }

  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 3;
  options.sample_size = 30;
  options.n_samples = 2;
  options.random_seed = 7;

  dtwc::Problem in_ram{"clara_in_ram"};
  in_ram.set_data(dtwc::Data(std::vector<std::vector<double>>(series),
                             std::vector<std::string>(names)));
  const auto expected = dtwc::algorithms::fast_clara(in_ram, options);

  auto schema = arrow::schema({ arrow::field("data", arrow::list(arrow::float64())) });
  auto tmp = tmpdir() / "clara_streaming.parquet";
  write_parquet(tmp, arrow::Table::Make(schema, { make_list_f64(series) }));
  options.ram_limit_bytes = 1u << 20;
  options.parquet_path = tmp;
  options.parquet_column = "data";
  options.force_parquet_streaming = true;
  dtwc::Problem settings_only{"clara_streamed"};
  const auto streamed = dtwc::algorithms::fast_clara(settings_only, options);
  std::filesystem::remove(tmp);

  CHECK(streamed.medoid_indices == expected.medoid_indices);
  CHECK(streamed.labels == expected.labels);
  CHECK(streamed.total_cost == expected.total_cost);
  CHECK(streamed.labels.size() == series.size());
}

#else

// An Arrow build without Parquet must not pass on the Arrow cases alone: this
// test is not MAY_SKIP, so the harness scores the skip as a failure.
TEST_CASE("Parquet readers are built with Arrow", "[io][parquet]")
{
  SKIP("DTWC_HAS_ARROW is defined without DTWC_HAS_PARQUET");
}

#endif // DTWC_HAS_PARQUET

#endif // DTWC_HAS_ARROW
