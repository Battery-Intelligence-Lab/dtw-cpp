/**
 * @file unit_test_problem_storage_policy.cpp
 * @brief F20 live Problem series-storage routing contract.
 *
 * The decisive subject is Problem::set_storage_policy followed by the existing
 * owning Problem::set_data(Data) call.  DataLoader-only coverage is not enough:
 * Python and MATLAB expose Problem policy plus set_data, but no DataLoader.
 */

#include <dtwc.hpp>
#include <error.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

#ifndef DTWC_F20_TEST_ROOT
#define DTWC_F20_TEST_ROOT "./build/f20-problem-storage"
#endif

namespace fs = std::filesystem;

namespace dtwc {

struct ProblemStoragePolicyTestAccess
{
  static const LoadedData *series_storage_owner(const Problem &problem) noexcept
  {
    return problem.series_storage_owner_.get();
  }
};

} // namespace dtwc

namespace {

constexpr std::size_t kSeries = 6;
constexpr std::size_t kValuesPerSeries = 6;
constexpr std::size_t kValues = kSeries * kValuesPerSeries;
constexpr std::size_t kFootprint = kValues * sizeof(double);
constexpr std::size_t kDirectNdim = 2;
constexpr std::size_t kStoreBytes = 408;

constexpr std::array<std::uint64_t, 15> kIndependentPairBits{
  0x4071fdcac083126fULL,
  0x406456b851eb851fULL,
  0x406c7e76c8b43958ULL,
  0x4070914395810625ULL,
  0x4072b9ef9db22d0fULL,
  0x4071476c8b439581ULL,
  0x406ecdb22d0e5604ULL,
  0x406cd3ef9db22d0eULL,
  0x406a6b7ced916872ULL,
  0x406b8978d4fdf3b7ULL,
  0x406e59c28f5c28f6ULL,
  0x40713c147ae147afULL,
  0x4057ebd70a3d70a4ULL,
  0x406d6a353f7ced91ULL,
  0x406fdb3333333332ULL,
};

#ifdef DTWC_HAS_MMAP
void require_mmap_name_owner(const dtwc::Problem &problem)
{
  const auto *owner =
    dtwc::ProblemStoragePolicyTestAccess::series_storage_owner(problem);
  REQUIRE(owner != nullptr);
  REQUIRE(owner->is_mmap());
  REQUIRE(owner->names.size() == problem.size());
  REQUIRE(owner->data.size() == problem.size());
  for (std::size_t i = 0; i < problem.size(); ++i) {
    REQUIRE(owner->data.name(i).data() == owner->names[i].data());
    REQUIRE(owner->data.name(i).size() == owner->names[i].size());
    REQUIRE(problem.series_name(i).data() == owner->names[i].data());
    REQUIRE(problem.series_name(i).size() == owner->names[i].size());
  }
}
#endif

fs::path fixture_path()
{
  return fs::path(DTWC_TEST_DATA_DIR) / "test" / "nonUnimodular_1_Nc_2.csv";
}

fs::path test_root()
{
  return fs::path(DTWC_F20_TEST_ROOT);
}

void reset_test_root()
{
  const auto root = test_root().lexically_normal();
  REQUIRE(!root.empty());
  fs::remove_all(root);
  fs::create_directories(root);
  REQUIRE(fs::exists(root));
  REQUIRE(fs::equivalent(fs::temp_directory_path(), root));
}

dtwc::Data load_fixture(std::size_t ndim)
{
  dtwc::DataLoader loader{ fixture_path() };
  loader.verbosity(0);
  auto native = loader.load_local();
  REQUIRE(native.size() == kSeries);
  REQUIRE(native.p_vec.size() == kSeries);
  REQUIRE(native.p_names.size() == kSeries);
  for (std::size_t i = 0; i < kSeries; ++i) {
    REQUIRE(native.p_vec[i].size() == kValuesPerSeries);
    REQUIRE(native.p_names[i] == std::to_string(i + 1));
  }
  return dtwc::Data(
    std::move(native.p_vec), std::move(native.p_names), ndim);
}

dtwc::Data sentinel_data()
{
  return dtwc::Data(
    std::vector<std::vector<double>>{ { -7.0, -3.0, 11.0, 2.0 } },
    std::vector<std::string>{ "sentinel" },
    2);
}

std::vector<fs::path> store_files()
{
  std::vector<fs::path> result;
  if (!fs::exists(test_root()))
    return result;
  for (const auto &entry : fs::directory_iterator(test_root())) {
    if (entry.is_regular_file() && entry.path().extension() == ".dtws")
      result.push_back(entry.path());
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::vector<std::byte> read_bytes(const fs::path &path)
{
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  REQUIRE(end >= 0);
  std::vector<std::byte> bytes(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(
    reinterpret_cast<char *>(bytes.data()),
    static_cast<std::streamsize>(bytes.size()));
  REQUIRE(input.gcount() == static_cast<std::streamsize>(bytes.size()));
  REQUIRE_FALSE(input.bad());
  return bytes;
}

std::uint16_t read_u16_le(
  const std::vector<std::byte> &bytes, std::size_t offset)
{
  REQUIRE(offset + 2 <= bytes.size());
  return static_cast<std::uint16_t>(std::to_integer<unsigned char>(bytes[offset]))
       | static_cast<std::uint16_t>(
           std::to_integer<unsigned char>(bytes[offset + 1]) << 8);
}

std::uint32_t read_u32_le(
  const std::vector<std::byte> &bytes, std::size_t offset)
{
  REQUIRE(offset + 4 <= bytes.size());
  std::uint32_t result = 0;
  for (std::size_t i = 0; i < 4; ++i)
    result |= static_cast<std::uint32_t>(
      std::to_integer<unsigned char>(bytes[offset + i])) << (8 * i);
  return result;
}

std::uint64_t read_u64_le(
  const std::vector<std::byte> &bytes, std::size_t offset)
{
  REQUIRE(offset + 8 <= bytes.size());
  std::uint64_t result = 0;
  for (std::size_t i = 0; i < 8; ++i)
    result |= static_cast<std::uint64_t>(
      std::to_integer<unsigned char>(bytes[offset + i])) << (8 * i);
  return result;
}

double independent_dependent_l1(
  std::span<const double> lhs,
  std::span<const double> rhs,
  std::size_t ndim)
{
  REQUIRE(ndim > 0);
  REQUIRE(lhs.size() % ndim == 0);
  REQUIRE(rhs.size() % ndim == 0);
  const auto n = lhs.size() / ndim;
  const auto m = rhs.size() / ndim;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> matrix((n + 1) * (m + 1), inf);
  const auto cell = [m](std::size_t i, std::size_t j) {
    return i * (m + 1) + j;
  };
  matrix[cell(0, 0)] = 0.0;
  for (std::size_t i = 1; i <= n; ++i) {
    for (std::size_t j = 1; j <= m; ++j) {
      double local = 0.0;
      for (std::size_t d = 0; d < ndim; ++d) {
        local += std::abs(
          lhs[(i - 1) * ndim + d] - rhs[(j - 1) * ndim + d]);
      }
      matrix[cell(i, j)] = local + std::min({
        matrix[cell(i - 1, j)],
        matrix[cell(i, j - 1)],
        matrix[cell(i - 1, j - 1)],
      });
    }
  }
  return matrix[cell(n, m)];
}

void verify_data_exact(
  const dtwc::Problem &problem, const dtwc::Data &expected)
{
  REQUIRE(problem.size() == expected.size());
  REQUIRE(problem.data().ndim == expected.ndim);
  REQUIRE(problem.data().precision == expected.precision);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(problem.series_name(i) == expected.name(i));
    const auto observed = problem.series(i);
    const auto reference = expected.series(i);
    REQUIRE(observed.size() == reference.size());
    for (std::size_t j = 0; j < reference.size(); ++j) {
      CHECK(
        std::bit_cast<std::uint64_t>(observed[j])
        == std::bit_cast<std::uint64_t>(reference[j]));
    }
  }
}

void verify_independent_oracle(
  dtwc::Problem &problem, const dtwc::Data &expected)
{
  std::size_t upper_index = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    for (std::size_t j = i + 1; j < expected.size(); ++j) {
      const double oracle = independent_dependent_l1(
        expected.series(i), expected.series(j), expected.ndim);
      REQUIRE(upper_index < kIndependentPairBits.size());
      CHECK(std::bit_cast<std::uint64_t>(oracle)
            == kIndependentPairBits[upper_index]);
      ++upper_index;
    }
  }
  REQUIRE(upper_index == kIndependentPairBits.size());

  for (std::size_t i = 0; i < expected.size(); ++i) {
    for (std::size_t j = 0; j < expected.size(); ++j) {
      const double oracle = i == j
        ? 0.0
        : independent_dependent_l1(
            expected.series(i), expected.series(j), expected.ndim);
      const double observed = problem.dist_by_ind(
        static_cast<int>(i), static_cast<int>(j));
      CHECK(std::bit_cast<std::uint64_t>(observed)
            == std::bit_cast<std::uint64_t>(oracle));
    }
  }
}

void verify_store_artifact(
  const fs::path &path, const dtwc::Data &expected)
{
  const auto bytes = read_bytes(path);
  REQUIRE(bytes.size() == kStoreBytes);
  REQUIRE(std::to_integer<char>(bytes[0]) == 'D');
  REQUIRE(std::to_integer<char>(bytes[1]) == 'T');
  REQUIRE(std::to_integer<char>(bytes[2]) == 'W');
  REQUIRE(std::to_integer<char>(bytes[3]) == 'S');
  REQUIRE(read_u16_le(bytes, 4) == 1);
  REQUIRE(read_u32_le(bytes, 6) == 0x01020304U);
  REQUIRE(std::to_integer<unsigned char>(bytes[10]) == 8);
  REQUIRE(read_u64_le(bytes, 12) == kSeries);
  REQUIRE(read_u64_le(bytes, 20) == kDirectNdim);
  REQUIRE(read_u32_le(bytes, 28) == 0xab81a0e4U);

  constexpr std::size_t header_size = 64;
  constexpr std::size_t offset_count = kSeries + 1;
  constexpr std::size_t data_start =
    header_size + offset_count * sizeof(std::uint64_t);
  for (std::size_t i = 0; i < offset_count; ++i)
    REQUIRE(read_u64_le(bytes, header_size + i * 8) == i * 48);

  std::size_t payload_offset = data_start;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto series = expected.series(i);
    const auto series_bytes = series.size_bytes();
    REQUIRE(payload_offset + series_bytes <= bytes.size());
    CHECK(std::memcmp(
      bytes.data() + payload_offset, series.data(), series_bytes) == 0);
    payload_offset += series_bytes;
  }
  REQUIRE(payload_offset == bytes.size());
}

struct CompletionState
{
  bool direct = false;
  bool transactions = false;
  bool view = false;
  bool loader = false;
  bool printed = false;
};

CompletionState &completion_state()
{
  static CompletionState state;
  return state;
}

enum class CompletedCase {
  Direct,
  Transactions,
  View,
  Loader,
};

void complete_case(CompletedCase completed)
{
  auto &state = completion_state();
  switch (completed) {
  case CompletedCase::Direct: state.direct = true; break;
  case CompletedCase::Transactions: state.transactions = true; break;
  case CompletedCase::View: state.view = true; break;
  case CompletedCase::Loader: state.loader = true; break;
  }
  if (!state.printed && state.direct && state.transactions
      && state.view && state.loader) {
#ifdef DTWC_HAS_MMAP
    std::cout
      << "F20_PROBLEM_STORAGE_POLICY build=llfio-on footprint=288 "
         "heap=owning mmap=view values=72/72 names=12/12 ndim_routes=2/2 "
         "ordered_pairs=72/72 artifact=pass lifetime=pass loader_auto=mmap "
         "view_override=pass subject_skips=0 verdict=PASS\n";
#else
    std::cout
      << "F20_PROBLEM_STORAGE_POLICY build=llfio-off footprint=288 "
         "heap=owning mmap=rejected values=36/36 names=6/6 ndim_routes=1/1 "
         "ordered_pairs=36/36 transaction=pass loader_auto=heap-warning "
         "view_override=pass subject_skips=0 verdict=PASS\n";
#endif
    state.printed = true;
  }
}

class CerrCapture
{
  std::ostringstream stream_;
  std::streambuf *previous_;

public:
  CerrCapture() : previous_(std::cerr.rdbuf(stream_.rdbuf())) {}
  ~CerrCapture() { std::cerr.rdbuf(previous_); }
  std::string str() const { return stream_.str(); }
};

} // namespace

TEST_CASE(
  "F20 forced Heap and Mmap retain exact multivariate series semantics",
  "[f20][problem][storage]")
{
  reset_test_root();
  REQUIRE(kFootprint == 288);
  const auto expected = load_fixture(kDirectNdim);

  dtwc::Problem heap("f20_heap");
  heap.set_storage_policy(dtwc::core::StoragePolicy::Heap);
  heap.set_data(expected);
  REQUIRE(heap.storage_policy() == dtwc::core::StoragePolicy::Heap);
  REQUIRE_FALSE(heap.data().is_view());
  REQUIRE(heap.data().p_vec.size() == kSeries);
  REQUIRE(store_files().empty());
  verify_data_exact(heap, expected);
  verify_independent_oracle(heap, expected);

  std::optional<dtwc::Problem> source;
  source.emplace("f20_mmap");
  source->set_storage_policy(dtwc::core::StoragePolicy::Heap);
  source->set_data(expected);
  REQUIRE_FALSE(source->data().is_view());
  source->set_storage_policy(dtwc::core::StoragePolicy::Mmap);
  REQUIRE(source->storage_policy() == dtwc::core::StoragePolicy::Mmap);
  REQUIRE_FALSE(source->data().is_view()); // policy changes are non-retroactive

#ifdef DTWC_HAS_MMAP
  source->set_data(load_fixture(kDirectNdim));
  const auto observed_mode =
    source->data().is_view() ? std::string_view("view")
                             : std::string_view("owning");
  std::cout << "F20_RED_OBSERVATION footprint=288 heap=owning mmap="
            << observed_mode << '\n';
  REQUIRE(source->data().is_view());
  REQUIRE(source->data().p_vec.empty());
  const auto files = store_files();
  REQUIRE(files.size() == 1);
  require_mmap_name_owner(*source);

  // Clear the matrix while the dispatcher is still bound to the source. A
  // move must rebind that self-referential function before any uncached pair
  // is evaluated by the destination.
  source->refresh_distance_matrix();
  const double primed_distance = source->dist_by_ind(0, 1);
  REQUIRE(std::bit_cast<std::uint64_t>(primed_distance)
          == kIndependentPairBits[0]);
  std::optional<dtwc::Problem> mapped;
  mapped.emplace(std::move(*source));
  require_mmap_name_owner(*mapped);
  source.reset();
  source.emplace("f20_move_poison");
  source->band = 0;
  const double move_oracle = independent_dependent_l1(
    expected.series(0), expected.series(2), expected.ndim);
  const dtwc::Problem &const_mapped = *mapped;
  const double const_moved_distance = const_mapped.dtw_function()(
    const_mapped.series(0), const_mapped.series(2));
  REQUIRE(std::bit_cast<std::uint64_t>(const_moved_distance)
          == std::bit_cast<std::uint64_t>(move_oracle));
  const double moved_distance = mapped->dist_by_ind(0, 2);
  REQUIRE(std::bit_cast<std::uint64_t>(moved_distance)
          == std::bit_cast<std::uint64_t>(move_oracle));
  source.reset();

  mapped->refresh_distance_matrix();
  const double assignment_primed = mapped->dist_by_ind(0, 1);
  REQUIRE(std::bit_cast<std::uint64_t>(assignment_primed)
          == kIndependentPairBits[0]);
  std::optional<dtwc::Problem> assigned;
  assigned.emplace("f20_move_assignment_target");
  assigned->set_storage_policy(dtwc::core::StoragePolicy::Heap);
  assigned->set_data(sentinel_data());
  *assigned = std::move(*mapped);
  require_mmap_name_owner(*assigned);
  mapped.reset();
  mapped.emplace("f20_move_assignment_poison");
  mapped->band = 0;
  const double assigned_distance = assigned->dist_by_ind(0, 2);
  REQUIRE(std::bit_cast<std::uint64_t>(assigned_distance)
          == std::bit_cast<std::uint64_t>(move_oracle));
  mapped.reset();

  std::vector<std::string> churn(2048, std::string(256, 'x'));
  REQUIRE(churn.size() == 2048);
  verify_data_exact(*assigned, expected);
  verify_independent_oracle(*assigned, expected);

  assigned->refresh_distance_matrix();
  assigned->distance_strategy = dtwc::DistanceMatrixStrategy::CUDA;
  REQUIRE_THROWS_WITH(
    assigned->fill_distance_matrix(),
    "Problem::fill_distance_matrix: CUDA does not support mmap-backed series "
    "data; no backend call or CPU fallback was attempted. Select "
    "StoragePolicy::Heap before set_data.");
  assigned->refresh_distance_matrix();
  assigned->distance_strategy = dtwc::DistanceMatrixStrategy::Metal;
  REQUIRE_THROWS_WITH(
    assigned->fill_distance_matrix(),
    "Problem::fill_distance_matrix: Metal does not support mmap-backed series "
    "data; no backend call or CPU fallback was attempted. Select "
    "StoragePolicy::Heap before set_data.");
  assigned.reset();
  verify_store_artifact(files.front(), expected);

  auto long_name_data = load_fixture(kDirectNdim);
  std::vector<std::string> long_names;
  long_names.reserve(kSeries);
  for (std::size_t i = 0; i < kSeries; ++i) {
    long_names.push_back(
      "f20-mapped-name-" + std::to_string(i) + "-"
      + std::string(96, static_cast<char>('a' + i)));
    long_name_data.p_names[i] = long_names.back();
  }
  auto routed = dtwc::detail::route_series_storage(
    std::move(long_name_data),
    dtwc::core::StoragePolicy::Mmap,
    0,
    test_root() / "long_names.dtws",
    "F20 long-name router");
  REQUIRE(routed.is_mmap());
  REQUIRE(routed.names.size() == kSeries);
  auto moved_routed = std::move(routed);
  REQUIRE(moved_routed.is_mmap());
  for (std::size_t i = 0; i < kSeries; ++i) {
    REQUIRE(moved_routed.data.name(i) == long_names[i]);
    REQUIRE(moved_routed.data.name(i).data()
            == moved_routed.names[i].data());
    REQUIRE(moved_routed.data.name(i).size()
            == moved_routed.names[i].size());
  }
#else
  try {
    source->set_data(load_fixture(kDirectNdim));
    FAIL("explicit Problem Mmap set_data succeeded without llfio");
  } catch (const dtwc::IOError &error) {
    REQUIRE(std::string(error.what()) ==
      "Problem::set_data: StoragePolicy::Mmap requested but mmap support "
      "(llfio) is not compiled in. Rebuild with -DDTWC_ENABLE_LLFIO=ON.");
  }
  std::cout
    << "F20_RED_OBSERVATION footprint=288 heap=owning mmap=rejected\n";
  REQUIRE_FALSE(source->data().is_view());
  REQUIRE(source->data().p_vec.size() == kSeries);
  verify_data_exact(*source, expected);
  REQUIRE(store_files().empty());
#endif

  complete_case(CompletedCase::Direct);
}

TEST_CASE(
  "F20 Problem moves retain raw semantic-cache invalidation",
  "[f20][problem][storage][move]")
{
  dtwc::Problem source("f20_move_stale_cache");
  source.set_storage_policy(dtwc::core::StoragePolicy::Heap);
  source.set_data(dtwc::Data(
    std::vector<std::vector<double>>{
      { 0.0, 0.0, 0.0, 10.0 },
      { 0.0, 10.0, 10.0, 10.0 },
    },
    std::vector<std::string>{ "move-a", "move-b" }));
  const double unbanded = source.dist_by_ind(0, 1);
  REQUIRE(unbanded == 0.0);

  source.band = 0; // legacy raw mutation: cached unbanded distance is stale
  dtwc::Problem moved(std::move(source));
  const dtwc::Problem &const_moved = moved;
  REQUIRE_THROWS_WITH(
    (void)const_moved.dtw_function(),
    "Problem: bound DTW function configuration changed through a raw or nested "
    "mutation. Use a semantic setter or a mutable dtw_function accessor to "
    "refresh the dispatcher before const access.");

  const double diagonal_only = moved.dist_by_ind(0, 1);
  REQUIRE(diagonal_only == 20.0);
  REQUIRE(moved.band == 0);
  REQUIRE(moved.series_name(0) == "move-a");
  REQUIRE(moved.series_name(1) == "move-b");
}

TEST_CASE(
  "F20 storage failures are typed and transactional",
  "[f20][problem][storage][transaction]")
{
  reset_test_root();
  dtwc::Problem problem("f20_transaction");
  problem.set_storage_policy(dtwc::core::StoragePolicy::Heap);
  problem.set_data(sentinel_data());
  problem.fill_distance_matrix();
  REQUIRE(problem.is_distance_matrix_filled());
  const auto sentinel_bits =
    std::bit_cast<std::uint64_t>(problem.series(0)[0]);

  problem.set_storage_policy(dtwc::core::StoragePolicy::Mmap);
  auto invalid = load_fixture(kDirectNdim);
  invalid.ndim = 4;
  REQUIRE_THROWS_WITH(
    problem.set_data(std::move(invalid)),
    "Series 0 has flat size 6 which is not divisible by ndim=4");
  REQUIRE(problem.size() == 1);
  REQUIRE(problem.series_name(0) == "sentinel");
  REQUIRE(std::bit_cast<std::uint64_t>(problem.series(0)[0])
          == sentinel_bits);
  REQUIRE(problem.is_distance_matrix_filled());
  REQUIRE(store_files().empty());

  dtwc::Data f32(
    std::vector<std::vector<float>>{
      { 1.0F, 2.0F, 3.0F, 4.0F },
      { 5.0F, 6.0F, 7.0F, 8.0F },
    },
    std::vector<std::string>{ "f32-a", "f32-b" });
  try {
    problem.set_data(f32);
    FAIL("explicit Float32 Mmap set_data succeeded");
  } catch (const dtwc::InvalidInput &error) {
    REQUIRE(std::string(error.what()) ==
      "Problem::set_data: StoragePolicy::Mmap supports Float64 series only; "
      "Float32 mmap requires a new .dtws format version.");
  }
  REQUIRE(problem.size() == 1);
  REQUIRE(problem.series_name(0) == "sentinel");
  REQUIRE(std::bit_cast<std::uint64_t>(problem.series(0)[0])
          == sentinel_bits);
  REQUIRE(problem.is_distance_matrix_filled());
  REQUIRE(store_files().empty());

  dtwc::Problem heap_f32("f20_heap_f32");
  heap_f32.set_storage_policy(dtwc::core::StoragePolicy::Heap);
  heap_f32.set_data(std::move(f32));
  REQUIRE(heap_f32.data().is_f32());
  REQUIRE_FALSE(heap_f32.data().is_view());
  REQUIRE(heap_f32.data().p_vec.empty());
  REQUIRE(heap_f32.data().series_f32(0).size() == 4);
  REQUIRE(heap_f32.data().series_f32(0)[2] == 3.0F);

#ifndef DTWC_HAS_MMAP
  try {
    problem.set_data(load_fixture(kDirectNdim));
    FAIL("explicit Float64 Mmap set_data succeeded without llfio");
  } catch (const dtwc::IOError &error) {
    REQUIRE(std::string(error.what()) ==
      "Problem::set_data: StoragePolicy::Mmap requested but mmap support "
      "(llfio) is not compiled in. Rebuild with -DDTWC_ENABLE_LLFIO=ON.");
  }
  REQUIRE(problem.size() == 1);
  REQUIRE(problem.series_name(0) == "sentinel");
  REQUIRE(std::bit_cast<std::uint64_t>(problem.series(0)[0])
          == sentinel_bits);
  REQUIRE(problem.is_distance_matrix_filled());
  REQUIRE(store_files().empty());
#endif

  complete_case(CompletedCase::Transactions);
}

TEST_CASE(
  "F20 set_view_data remains a pointer-identical policy bypass",
  "[f20][problem][storage][view]")
{
  reset_test_root();
  auto owner = load_fixture(kDirectNdim);
  std::vector<std::span<const double>> spans;
  std::vector<std::string_view> names;
  for (std::size_t i = 0; i < owner.size(); ++i) {
    spans.push_back(owner.series(i));
    names.push_back(owner.name(i));
  }

  for (const auto policy : {
         dtwc::core::StoragePolicy::Heap,
         dtwc::core::StoragePolicy::Mmap,
       }) {
    dtwc::Problem problem("f20_view");
    problem.set_storage_policy(policy);
    auto candidate_spans = spans;
    auto candidate_names = names;
    problem.set_view_data(dtwc::Data(
      std::move(candidate_spans), std::move(candidate_names), kDirectNdim));
    REQUIRE(problem.storage_policy() == policy);
    REQUIRE(problem.data().is_view());
    REQUIRE(problem.data().ndim == kDirectNdim);
    REQUIRE(problem.size() == owner.size());
    for (std::size_t i = 0; i < owner.size(); ++i) {
      REQUIRE(problem.series(i).data() == owner.series(i).data());
      REQUIRE(problem.series_name(i).data() == owner.name(i).data());
      REQUIRE(problem.series(i).size() == owner.series(i).size());
    }
    REQUIRE(store_files().empty());
  }

  complete_case(CompletedCase::View);
}

TEST_CASE(
  "F20 Problem loader constructor owns the policy-routed result",
  "[f20][problem][storage][loader]")
{
  reset_test_root();
  const auto native_expected = load_fixture(1);

  dtwc::DataLoader heap_loader{ fixture_path() };
  heap_loader.verbosity(0)
    .storage_policy(dtwc::core::StoragePolicy::Heap)
    .ram_limit(1)
    .mmap_cache_path(test_root() / "loader_heap_forbidden.dtws");
  std::optional<dtwc::Problem> heap;
  std::string heap_warning;
  {
    CerrCapture capture;
    heap.emplace("f20_loader_heap", heap_loader);
    heap_warning = capture.str();
  }
  REQUIRE(heap->storage_policy() == dtwc::core::StoragePolicy::Heap);
  REQUIRE_FALSE(heap->data().is_view());
  REQUIRE_FALSE(fs::exists(test_root() / "loader_heap_forbidden.dtws"));
  REQUIRE(heap_warning.empty());
  verify_data_exact(*heap, native_expected);

  std::optional<dtwc::Problem> auto_problem;
  std::string warning;
  {
    dtwc::DataLoader auto_loader{ fixture_path() };
    auto_loader.verbosity(0)
      .storage_policy(dtwc::core::StoragePolicy::Auto)
      .ram_limit(1)
      .mmap_cache_path(test_root() / "loader_auto.dtws");
#ifdef DTWC_HAS_MMAP
    auto_problem.emplace("f20_loader_auto", auto_loader);
#else
    {
      CerrCapture capture;
      auto_problem.emplace("f20_loader_auto", auto_loader);
      warning = capture.str();
    }
#endif
  } // loader lifetime ends here

  REQUIRE(auto_problem.has_value());
  REQUIRE(auto_problem->storage_policy() == dtwc::core::StoragePolicy::Auto);
#ifdef DTWC_HAS_MMAP
  REQUIRE(auto_problem->data().is_view());
  REQUIRE(fs::exists(test_root() / "loader_auto.dtws"));
#else
  REQUIRE_FALSE(auto_problem->data().is_view());
  REQUIRE(warning ==
    "[dtwc] warning: dataset footprint (288 B) exceeds the storage threshold "
    "(1 B) but mmap support is not compiled in; keeping data in RAM. Rebuild "
    "with -DDTWC_ENABLE_LLFIO=ON to enable the mmap-backed store.\n");
  REQUIRE_FALSE(fs::exists(test_root() / "loader_auto.dtws"));
#endif
  std::vector<std::string> churn(2048, std::string(256, 'y'));
  REQUIRE(churn.size() == 2048);
  verify_data_exact(*auto_problem, native_expected);

  complete_case(CompletedCase::Loader);
}
