/**
 * @file unit_test_checkpoint.cpp
 * @brief save_checkpoint / load_checkpoint: one `.dtwm` file and its typed outcomes.
 *
 * @details The table is the contract: no file is `false`; a file for other
 * series or other distance settings is InvalidInput; a file that is not a whole
 * `.dtwm` file is IOError; a round trip is bit-exact. A checkpoint has the layout
 * of a mapped matrix, so with llfio every row also runs through
 * use_mmap_distance_matrix, which must answer the same.
 *
 * @author Volkan Kumtepeli
 * @date 29 Sep 2026
 */

#include <dtwc.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

using namespace dtwc;
namespace fs = std::filesystem;
using dtwc::test_support::ScratchDirectory;

namespace {

/// n short integer-valued series, all of one length.
Problem make_problem(const std::string &name, std::size_t n = 5)
{
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (std::size_t i = 0; i < n; ++i) {
    series.push_back({ double(i), double(i * i % 7), double(3 * i % 5), 1.0, double(i % 2) });
    names.push_back("s" + std::to_string(i));
  }
  Problem prob(name);
  prob.set_data(Data(std::move(series), std::move(names)));
  return prob;
}

std::vector<std::uint64_t> packed_bits(const Problem &prob)
{
  const auto &m = prob.distance_matrix();
  std::vector<std::uint64_t> bits;
  for (std::size_t k = 0; k < m.packed_count(); ++k) bits.push_back(std::bit_cast<std::uint64_t>(m.raw()[k]));
  return bits;
}

std::vector<char> read_bytes(const fs::path &path)
{
  std::ifstream in(path, std::ios::binary);
  return { std::istreambuf_iterator<char>{ in }, std::istreambuf_iterator<char>{} };
}

void write_bytes(const fs::path &path, const std::vector<char> &bytes)
{
  std::ofstream(path, std::ios::binary | std::ios::trunc).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

enum class Outcome { Loaded, Absent, InvalidInput, IOError };

Outcome load_outcome(Problem &prob, const std::string &dir)
{
  try {
    return load_checkpoint(prob, dir) ? Outcome::Loaded : Outcome::Absent;
  } catch (const dtwc::InvalidInput &) {
    return Outcome::InvalidInput;
  } catch (const dtwc::IOError &) {
    return Outcome::IOError;
  }
}

#ifdef DTWC_HAS_MMAP
Outcome map_outcome(Problem &prob, const fs::path &file)
{
  try {
    prob.use_mmap_distance_matrix(file);
    return Outcome::Loaded;
  } catch (const dtwc::InvalidInput &) {
    return Outcome::InvalidInput;
  } catch (const dtwc::IOError &) {
    return Outcome::IOError;
  }
}
#endif

} // namespace

TEST_CASE("A checkpoint round trip is bit-exact, full and partial", "[checkpoint]")
{
  const ScratchDirectory scratch("round_trip");
  auto full = make_problem("full");
  full.fill_distance_matrix();
  save_checkpoint(full, scratch.path.string());
  REQUIRE(fs::file_size(scratch.path / "full.dtwm") == 48 + 15 * sizeof(double));

  auto partial = make_problem("partial");
  partial.distance_matrix().resize(5);
  partial.distance_matrix().set(3, 1, -0.0);
  partial.distance_matrix().set(4, 4, 5e-324);
  save_checkpoint(partial, scratch.path.string());

  for (const Problem *source : { &full, &partial }) {
    auto target = make_problem(source->name());
    REQUIRE(load_checkpoint(target, scratch.path.string()));
    REQUIRE_FALSE(target.distance_matrix().is_mapped());
    REQUIRE(packed_bits(target) == packed_bits(*source)); // NaN = not computed, bit for bit
#ifdef DTWC_HAS_MMAP
    auto mapped = make_problem(source->name()); // the same file, mapped
    mapped.use_mmap_distance_matrix(checkpoint_path(mapped, scratch.path.string()));
    REQUIRE(packed_bits(mapped) == packed_bits(*source));
#endif
  }
  REQUIRE(partial.distance_matrix().count_computed() == 2);
}

TEST_CASE("Loading a checkpoint: the table of typed outcomes", "[checkpoint][error]")
{
  struct Row
  {
    const char *what;
    Outcome expected;
    std::function<void(std::vector<char> &)> damage; // applied to the saved file
    std::size_t reader_n = 5;
    int reader_band = -1;
  };
  const std::vector<Row> rows{
    { "other N", Outcome::InvalidInput, [](std::vector<char> &) {}, 4 },
    { "other identity (band)", Outcome::InvalidInput, [](std::vector<char> &) {}, 5, 2 },
    { "short", Outcome::IOError, [](std::vector<char> &b) { b.resize(20); } },
    { "empty", Outcome::IOError, [](std::vector<char> &b) { b.clear(); } },
    { "bad magic", Outcome::IOError, [](std::vector<char> &b) { b[0] = 'X'; } },
    { "bad version", Outcome::IOError, [](std::vector<char> &b) { b[4] = 3; } },
    { "wrong length (a double more)", Outcome::IOError, [](std::vector<char> &b) { b.resize(b.size() + 8); } },
    { "wrong length (a byte less)", Outcome::IOError, [](std::vector<char> &b) { b.pop_back(); } },
  };
  const ScratchDirectory scratch("table");
  auto writer = make_problem("table");
  writer.fill_distance_matrix();
  save_checkpoint(writer, scratch.path.string());
  const fs::path file = scratch.path / "table.dtwm";
  const auto saved = read_bytes(file);

  {
    auto reader = make_problem("absent");
    REQUIRE(load_outcome(reader, scratch.path.string()) == Outcome::Absent);
    REQUIRE(reader.distance_matrix().size() == 0);
  }
  for (const auto &row : rows) {
    INFO(row.what);
    auto bytes = saved;
    row.damage(bytes);
    write_bytes(file, bytes);
    auto reader = make_problem("table", row.reader_n);
    reader.set_band(row.reader_band);
    CHECK(load_outcome(reader, scratch.path.string()) == row.expected);
    CHECK(reader.distance_matrix().size() == 0); // nothing was installed
#ifdef DTWC_HAS_MMAP
    CHECK(map_outcome(reader, file) == row.expected);
    CHECK_FALSE(reader.distance_matrix().is_mapped());
#endif
    CHECK(read_bytes(file) == bytes); // a rejected file is left as it is
  }
}

TEST_CASE("save_checkpoint makes its directory, replaces its file and fails typed", "[checkpoint]")
{
  const ScratchDirectory scratch("save");
  const fs::path dir = scratch.path / "a" / "b";
  auto prob = make_problem("replace");
  save_checkpoint(prob, dir.string()); // nothing computed yet: all NaN
  prob.fill_distance_matrix();
  save_checkpoint(prob, dir.string());
  REQUIRE_FALSE(fs::exists(dir / "replace.dtwm.tmp"));
  auto reader = make_problem("replace");
  REQUIRE(load_checkpoint(reader, dir.string()));
  REQUIRE(reader.is_distance_matrix_filled());

  std::ofstream(scratch.path / "plain") << "a file where the directory belongs";
  REQUIRE_THROWS_AS(save_checkpoint(prob, (scratch.path / "plain").string()), dtwc::IOError);
  REQUIRE_THROWS_AS(save_checkpoint(prob, (scratch.path / "plain" / "sub").string()), dtwc::IOError);
  REQUIRE_THROWS_AS(save_checkpoint(Problem("empty"), scratch.path.string()), dtwc::InvalidInput);
}

TEST_CASE("Automatic checkpointing saves after every row block", "[checkpoint][fill]")
{
  const ScratchDirectory scratch("auto");
  auto reference = make_problem("auto", 6);
  reference.fill_distance_matrix();

  auto prob = make_problem("auto", 6);
  prob.checkpoint = { scratch.path.string(), 1, true };
  prob.fill_distance_matrix();
  auto restored = make_problem("auto", 6);
  REQUIRE(load_checkpoint(restored, scratch.path.string()));
  REQUIRE(packed_bits(restored) == packed_bits(reference));

  auto disabled = make_problem("off", 6);
  disabled.checkpoint.directory = (scratch.path / "never").string();
  disabled.fill_distance_matrix();
  REQUIRE_FALSE(fs::exists(scratch.path / "never"));

  for (const CheckpointOptions bad : { CheckpointOptions{ scratch.path.string(), 0, true }, CheckpointOptions{ "", 1, true } }) {
    auto rejected = make_problem("bad", 6);
    rejected.checkpoint = bad;
    REQUIRE_THROWS_AS(rejected.fill_distance_matrix(), dtwc::InvalidInput);
    REQUIRE(rejected.distance_matrix().count_computed() == 0);
  }
}

TEST_CASE("A resumed fill computes only the missing cells", "[checkpoint][fill]")
{
  constexpr std::size_t N = 6;
  const ScratchDirectory scratch("resume");
  auto reference = make_problem("resume", N);
  reference.fill_distance_matrix();

  // A crash after two rows: those cells hold values no DTW kernel produces.
  auto crashed = make_problem("resume", N);
  crashed.distance_matrix().resize(N);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = i + 1; j < N; ++j) crashed.distance_matrix().set(i, j, 900.0 + 10.0 * i + j);
  save_checkpoint(crashed, scratch.path.string());

  auto resumed = make_problem("resume", N);
  REQUIRE(load_checkpoint(resumed, scratch.path.string()));
  REQUIRE(resumed.distance_matrix().count_computed() == 9);
  resumed.checkpoint = { scratch.path.string(), 2, true };
  resumed.fill_distance_matrix();
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j)
      REQUIRE(resumed.dist_by_ind(int(i), int(j)) == (i < 2 ? 900.0 + 10.0 * i + j : reference.dist_by_ind(int(i), int(j))));

  auto reread = make_problem("resume", N); // the last block's save holds the whole matrix
  REQUIRE(load_checkpoint(reread, scratch.path.string()));
  REQUIRE(packed_bits(reread) == packed_bits(resumed));
}

TEST_CASE("A failing automatic save keeps the computed distances", "[checkpoint][fill]")
{
  constexpr std::size_t N = 6;
  const ScratchDirectory scratch("save_fails");
  fs::create_directories(scratch.path);
  std::ofstream(scratch.path / "not_a_directory") << "occupied";

  auto prob = make_problem("fails", N);
  prob.checkpoint = { (scratch.path / "not_a_directory").string(), 2, true };
  REQUIRE_THROWS_AS(prob.fill_distance_matrix(), dtwc::IOError);
  // The first block ran before its save failed: the diagonal and rows 0-1.
  REQUIRE(prob.distance_matrix().count_computed() == N + (N - 1) + (N - 2));
  prob.checkpoint.enabled = false;
  prob.fill_distance_matrix();
  REQUIRE(prob.is_distance_matrix_filled());
}

#ifdef DTWC_HAS_MMAP
TEST_CASE("A matrix mapped to its checkpoint file is saved in place", "[checkpoint][mmap]")
{
  const ScratchDirectory scratch("mapped");
  fs::create_directories(scratch.path);
  auto reference = make_problem("mapped", 6);
  reference.fill_distance_matrix();

  {
    auto prob = make_problem("mapped", 6);
    prob.use_mmap_distance_matrix(checkpoint_path(prob, scratch.path.string()));
    prob.checkpoint = { scratch.path.string(), 2, true };
    prob.fill_distance_matrix(); // each block's save flushes the mapping
    REQUIRE(prob.distance_matrix().is_mapped());
    REQUIRE_FALSE(fs::exists(scratch.path / "mapped.dtwm.tmp"));
    save_checkpoint(prob, scratch.path.string());
  }
  auto reader = make_problem("mapped", 6);
  REQUIRE(load_checkpoint(reader, scratch.path.string()));
  REQUIRE(packed_bits(reader) == packed_bits(reference));
}
#endif
