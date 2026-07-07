/**
 * @file test_io_readers.cpp
 * @brief Regression tests for Arrow IPC + Parquet reader hardening (Task 0.8).
 *
 * Targets three confirmed audit findings (2026-06-01, "io-security"):
 *   1. arrow_ipc_reader / parquet_reader cast list/scalar values to DoubleArray
 *      with NO Float64 check -> a Float32 file is reinterpreted as doubles
 *      (garbage values + out-of-bounds read of the trailing element).
 *   2. list offsets are used to index the values buffer with NO bounds check ->
 *      a crafted file whose offset exceeds the values length reads OOB.
 *   3. ArrowIPCDataSource reads `ndim` from schema metadata with no lower bound;
 *      series_length() divides by ndim_, so ndim=0 is a division by zero.
 *
 * Each test builds an Arrow/Parquet fixture with the Arrow C++ API in a temp
 * directory, then exercises the reader. The tests are gated on DTWC_HAS_ARROW /
 * DTWC_HAS_PARQUET exactly like the readers themselves (see dtwc/CMakeLists.txt);
 * when Arrow is not built the file registers a single skipped case.
 *
 * @date 2026-07-07
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

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

#include <io/arrow_ipc_reader.hpp>

#if defined(DTWC_HAS_PARQUET)
#include <parquet/arrow/writer.h>
#include <io/parquet_reader.hpp>
#endif

using Catch::Matchers::WithinAbs;

namespace {

template <class T>
T unwrap(arrow::Result<T> r)
{
  REQUIRE(r.ok());
  return r.ValueOrDie(); // reference, copied out (cheap for shared_ptr)
}

std::filesystem::path tmpdir()
{
  auto d = std::filesystem::temp_directory_path() / "dtwc_io_reader_test";
  std::filesystem::create_directories(d);
  return d;
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

// Write a single-column Arrow IPC (Feather v2) file.
void write_ipc(const std::filesystem::path &path,
               const std::shared_ptr<arrow::Schema> &schema,
               const std::shared_ptr<arrow::Array> &arr)
{
  auto batch = arrow::RecordBatch::Make(schema, arr->length(), { arr });
  auto out = unwrap(arrow::io::FileOutputStream::Open(path.string()));
  auto writer = unwrap(arrow::ipc::MakeFileWriter(out, schema));
  REQUIRE(writer->WriteRecordBatch(*batch).ok());
  REQUIRE(writer->Close().ok());
  REQUIRE(out->Close().ok());
}

} // namespace

// -------------------------- Arrow IPC reader --------------------------

TEST_CASE("ArrowIPC: valid Float64 file reads with ndim>1", "[io][arrow]")
{
  // Positive control: proves the new value-type / offset / ndim guards do NOT
  // over-reject a well-formed multivariate Float64 file.
  auto arr = make_list_f64({ { 1.0, 2.0, 3.0, 4.0 } }); // one series, 4 flat values
  auto meta = arrow::key_value_metadata({ "ndim" }, { "2" });
  auto schema = arrow::schema({ arrow::field("data", arr->type()) }, meta);
  auto tmp = tmpdir() / "valid_ndim2.arrow";
  write_ipc(tmp, schema, arr);

  auto src = dtwc::io::ArrowIPCDataSource::open(tmp);
  REQUIRE(src.size() == 1);
  REQUIRE(src.ndim() == 2);
  REQUIRE(src.series_length(0) == 2); // 4 flat values / ndim 2
  auto sp = src.series(0);
  REQUIRE(sp.size() == 4);
  CHECK_THAT(sp[0], WithinAbs(1.0, 1e-12));
  CHECK_THAT(sp[3], WithinAbs(4.0, 1e-12));
  std::filesystem::remove(tmp);
}

TEST_CASE("ArrowIPC: Float32 list values rejected by name", "[io][arrow]")
{
  // Bug 1: open() cast list values to DoubleArray with no Float64 check, so a
  // List<Float32> file was reinterpreted as doubles (garbage + OOB). The fix
  // rejects non-Float64 values by name. PRE-FIX open() succeeds (no throw) so
  // this REQUIRE_THROWS_AS fails; POST-FIX open() throws.
  auto arr = make_list_f32({ { 1.5f, 2.5f, 3.5f }, { 4.5f, 5.5f } });
  auto schema = arrow::schema({ arrow::field("data", arr->type()) });
  auto tmp = tmpdir() / "f32_list.arrow";
  write_ipc(tmp, schema, arr);

  REQUIRE_THROWS_AS(dtwc::io::ArrowIPCDataSource::open(tmp), std::runtime_error);
  std::filesystem::remove(tmp);
}

TEST_CASE("ArrowIPC: ndim=0 metadata rejected", "[io][arrow]")
{
  // Bug 3: ndim read from schema metadata with no lower bound; series_length()
  // divides the flat size by ndim_, so ndim=0 is a division by zero. PRE-FIX
  // open() succeeds with ndim_==0 (and series_length() later divides by zero);
  // POST-FIX open() throws.
  auto arr = make_list_f64({ { 1.0, 2.0, 3.0 } });
  auto meta = arrow::key_value_metadata({ "ndim" }, { "0" });
  auto schema = arrow::schema({ arrow::field("data", arr->type()) }, meta);
  auto tmp = tmpdir() / "ndim0.arrow";
  write_ipc(tmp, schema, arr);

  REQUIRE_THROWS_AS(dtwc::io::ArrowIPCDataSource::open(tmp), std::runtime_error);
  std::filesystem::remove(tmp);
}

TEST_CASE("ArrowIPC: out-of-bounds list offset rejected", "[io][arrow]")
{
  // Bug 2: offsets were used to index the mmap'd values buffer with no bounds
  // check, so a crafted file whose offset exceeds the values length makes
  // series() read OOB. We construct a ListArray directly from raw buffers with a
  // corrupt offset (6) that exceeds the values length (5). The IPC writer only
  // reads offsets[0] and offsets[length] (=3) to slice the values it emits, so
  // the file writes cleanly, but offset 6 exceeds the values length however the
  // writer slices. PRE-FIX open() succeeds (series() would read OOB); POST-FIX
  // open() throws at the offset-validation loop.
  auto pool = arrow::default_memory_pool();
  arrow::DoubleBuilder vb(pool);
  REQUIRE(vb.AppendValues(std::vector<double>{ 1.0, 2.0, 3.0, 4.0, 5.0 }).ok());
  std::shared_ptr<arrow::Array> values;
  REQUIRE(vb.Finish(&values).ok());

  // Corrupt int32 offsets for a length-2 list: [0, 6, 3]. offset 6 is OOB.
  auto offsets = arrow::Buffer::FromVector(std::vector<int32_t>{ 0, 6, 3 });
  auto list_type = arrow::list(arrow::float64());
  auto list = std::static_pointer_cast<arrow::Array>(
    std::make_shared<arrow::ListArray>(list_type, /*length=*/2, offsets, values));

  auto schema = arrow::schema({ arrow::field("data", list_type) });
  auto tmp = tmpdir() / "oob_offsets.arrow";
  write_ipc(tmp, schema, list);

  REQUIRE_THROWS_AS(dtwc::io::ArrowIPCDataSource::open(tmp), std::runtime_error);
  std::filesystem::remove(tmp);
}

// -------------------------- Parquet reader ----------------------------

#if defined(DTWC_HAS_PARQUET)

namespace {

void write_parquet(const std::filesystem::path &path,
                   const std::shared_ptr<arrow::Table> &table)
{
  auto out = unwrap(arrow::io::FileOutputStream::Open(path.string()));
  REQUIRE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1024).ok());
  REQUIRE(out->Close().ok());
}

} // namespace

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

  auto data = dtwc::io::load_parquet_file(tmp, "v");
  REQUIRE(data.size() == 1);
  REQUIRE(data.p_vec[0].size() == 3);
  CHECK_THAT(data.p_vec[0][0], WithinAbs(1.0, 1e-12));
  CHECK_THAT(data.p_vec[0][2], WithinAbs(3.0, 1e-12));
  std::filesystem::remove(tmp);
}

TEST_CASE("Parquet: scalar Float32 column converted to double", "[io][parquet]")
{
  // Bug 1 (scalar path): load_parquet_file cast every scalar chunk to
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

  auto data = dtwc::io::load_parquet_file(tmp, "v");
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

  auto data = dtwc::io::load_parquet_file(tmp, "data");
  REQUIRE(data.size() == 2);
  REQUIRE(data.p_vec[0].size() == 3);
  REQUIRE(data.p_vec[1].size() == 2);
  CHECK_THAT(data.p_vec[0][0], WithinAbs(1.5, 1e-6));
  CHECK_THAT(data.p_vec[0][2], WithinAbs(3.5, 1e-6));
  CHECK_THAT(data.p_vec[1][1], WithinAbs(5.5, 1e-6));
  std::filesystem::remove(tmp);
}

#endif // DTWC_HAS_PARQUET

#endif // DTWC_HAS_ARROW
