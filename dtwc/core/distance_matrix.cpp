/**
 * @file distance_matrix.cpp
 * @brief DistanceMatrix storage and the `.dtwm` file: the one translation unit
 *        that includes llfio.
 *
 * @author Volkan Kumtepeli
 * @date 29 Sep 2026
 */

#include "distance_matrix.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp" // run_openmp
#include "../fileOperations.hpp" // open_output, close_output

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h> // FlushFileBuffers
#else
#include <fcntl.h>  // open
#include <unistd.h> // fsync, close
#endif

#ifdef DTWC_HAS_MMAP
// quickcpplib's ringbuffer_log.hpp leaves an unbalanced Clang ignore for
// -Wdeprecated-declarations on Windows, and Apple libc++ deprecates
// std::char_traits<std::byte> inside llfio: keep both inside this include.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <llfio/v2.0/llfio.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

namespace dtwc::core {

namespace fs = std::filesystem;

struct DistanceMatrix::Mapping
{
#ifdef DTWC_HAS_MMAP
  LLFIO_V2_NAMESPACE::mapped_file_handle file;
#endif
  fs::path path;
  size_t header_bytes; ///< 48 in a version-4 file, else 64
};

namespace {

/// A version-5 header (distance_matrix.hpp); a version-4 file has its first 48 bytes.
struct Header
{
  char magic[4];
  std::uint32_t version;
  std::uint64_t n;
  DistanceMatrix::fingerprint_type fingerprint;
  std::uint32_t value_bytes;
  std::uint32_t zero;
  std::uint64_t rows_written;
};
static_assert(sizeof(Header) == 64 && std::is_trivially_copyable_v<Header>);
constexpr size_t v4_header_bytes = 48;
constexpr size_t rows_written_at = 56; // the byte offset of rows_written

constexpr char magic[4] = { 'D', 'T', 'W', 'M' };
constexpr std::uint32_t version = 5;

Header header_for(size_t n, const DistanceMatrix::fingerprint_type &fingerprint, Precision width, size_t rows)
{
  Header header{};
  std::memcpy(header.magic, magic, sizeof magic);
  header.version = version;
  header.n = n;
  header.fingerprint = fingerprint;
  header.value_bytes = static_cast<std::uint32_t>(value_bytes(width));
  header.rows_written = rows;
  return header;
}

std::string shown(const fs::path &path)
{
  const auto utf8 = path.u8string();
  return "'" + std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size()) + "'";
}

/// Where a file's values start, and its rows written.
struct Layout
{
  size_t header_bytes;
  size_t rows;
};

/// The checks every open of a `.dtwm` file makes, before any distance is
/// exposed. `bytes` holds the file's first min(length, 64) bytes.
Layout check_header(const char *bytes, std::uint64_t length, size_t n, const DistanceMatrix::fingerprint_type &fingerprint,
                    Precision width, const fs::path &path)
{
  const std::string fix = " Delete or rename it to compute the distances again.";
  const auto too_short = [&] {
    return IOError(shown(path) + " is " + std::to_string(length) + " bytes, too short for a .dtwm file." + fix);
  };
  if (length < v4_header_bytes) throw too_short();
  Header header{};
  std::memcpy(&header, bytes, static_cast<size_t>(std::min<std::uint64_t>(length, sizeof header)));
  if (std::memcmp(header.magic, magic, sizeof magic) != 0)
    throw IOError(shown(path) + " is not a .dtwm distance matrix (no DTWM magic)." + fix);
  if (header.version != 4 && header.version != version)
    throw IOError(shown(path) + " is a .dtwm file of version " + std::to_string(header.version)
                  + "; this build reads versions 4 and " + std::to_string(version) + "." + fix);
  // Version 4: binary64 values after 48 bytes, every row written.
  const bool v4 = header.version == 4;
  if (!v4 && length < sizeof header) throw too_short();
  const Layout layout{ v4 ? v4_header_bytes : sizeof header, v4 ? static_cast<size_t>(header.n) : header.rows_written };
  const std::uint64_t bytes_per_value = v4 ? sizeof(double) : header.value_bytes;
  const bool fields = v4 || ((bytes_per_value == sizeof(double) || bytes_per_value == sizeof(float)) && header.zero == 0
                             && header.rows_written <= header.n);
  // The N bound keeps packed_size exact in 64 bits; a file that large cannot exist.
  const std::uint64_t payload = length - layout.header_bytes;
  if (header.n >= (std::uint64_t{ 1 } << 32) || !fields || payload % bytes_per_value != 0
      || payload / bytes_per_value != packed_size(static_cast<size_t>(header.n)))
    throw IOError(shown(path) + " is " + std::to_string(length) + " bytes, which is not the size of the "
                  + std::to_string(header.n) + "-series matrix its header declares (truncated or corrupt)."
                  + fix);
  if (header.n != n)
    throw InvalidInput(shown(path) + " holds distances between " + std::to_string(header.n)
                       + " series, but this Problem has " + std::to_string(n)
                       + ". Use the file written for these series, or another path.");
  if (header.fingerprint != fingerprint)
    throw InvalidInput(shown(path) + " was computed for other data or other distance settings "
                       "(fingerprint mismatch): its distances are not this Problem's. Use the original "
                       "data, band, variant, missing-data strategy, metric and precision, or "
                       "another path.");
  const Precision stored = bytes_per_value == sizeof(float) ? Precision::Float32 : Precision::Float64;
  if (stored != width)
    throw InvalidInput(shown(path) + " holds " + std::string(name_of(precision_names, stored))
                       + " distances, but this Problem stores " + std::string(name_of(precision_names, width))
                       + " (matrix_dtype): set matrix_dtype " + std::string(name_of(precision_names, stored))
                       + ", or use another path.");
  return layout;
}

/// Push a file's bytes to the device. A file renamed into place with its data
/// still in the cache can come back from a power cut at its full length with a
/// zero tail, and a zero reads as a computed distance.
void flush_to_device(const fs::path &path)
{
#ifdef _WIN32
  const HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  const bool flushed = file != INVALID_HANDLE_VALUE && FlushFileBuffers(file);
  if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
  const int file = ::open(path.c_str(), O_WRONLY);
  const bool flushed = file >= 0 && ::fsync(file) == 0;
  if (file >= 0) ::close(file);
#endif
  if (!flushed) throw IOError("Cannot flush " + shown(path) + " to disk.");
}

/// NaN in `count` values, written by the OpenMP threads in contiguous chunks:
/// first touch places a page where a thread runs, rather than all where one runs.
template <typename T>
void fill_not_computed(T *data, size_t count)
{
  if (count == 0) return;
  const auto chunks = static_cast<size_t>(get_max_threads());
  auto chunk = [&](size_t k) {
    std::fill(data + count * k / chunks, data + count * (k + 1) / chunks, std::numeric_limits<T>::quiet_NaN());
  };
  run_openmp(chunk, chunks, true, 1);
}

#ifdef DTWC_HAS_MMAP
namespace llfio = LLFIO_V2_NAMESPACE;

/// The value of an llfio result, or IOError naming the step and the file: llfio's
/// own error is no dtwc::Error, so Python saw RuntimeError and MATLAB dtwc:runtime.
template <typename Result>
auto checked(Result &&result, const char *step, const fs::path &path)
{
  if (!result)
    throw IOError(std::string("Cannot ") + step + " the distance matrix file " + shown(path) + ": "
                  + result.error().message());
  return std::forward<Result>(result).value();
}

/// Write bytes [first, last) of the file `file` maps back to it, from the page
/// holding `first` (msync takes a page-aligned address): started, or finished
/// with `kind` wait_data_only. Linux writes back that range only
/// (sync_file_range); Windows and macOS flush the file.
void flush_range(llfio::mapped_file_handle &file, size_t first, size_t last,
                 llfio::mapped_file_handle::barrier_kind kind, const fs::path &path)
{
  first -= first % llfio::utils::page_size();
  llfio::mapped_file_handle::const_buffer_type range(file.address() + first, last - first);
  checked(file.barrier({ llfio::mapped_file_handle::const_buffers_type(&range, 1), first }, kind), "flush", path);
}
#endif

/// DistanceMatrix::all_computed's pass over packed rows [0, rows) of an n x n matrix's `values`, `width` wide. A
/// function, not a lambda, with the float32 hint built where it is thrown: GCC 13 kept each value on the stack too
/// with the hint live through the loop (2026-10-11).
template <typename T>
bool all_computed_in(const T *values, size_t n, size_t rows, Precision width, std::string_view where,
                     bool negative_is_distance, std::string_view negative_way_out)
{
  const auto pair = [&](size_t i, size_t j) {
    return std::string(where) + ": the distance between series " + std::to_string(j) + " and " + std::to_string(i)
           + " is ";
  };
  bool all = rows == n;
  for (size_t i = 0, k = 0; i < rows; ++i)
    for (size_t j = 0; j <= i; ++j, ++k) {
      const double d = values[k];
      if (std::isnan(d))
        all = false;
      else if (std::isinf(d))
        throw InvalidInput(pair(i, j) + (d > 0 ? "+inf" : "-inf") + "; a distance must be finite."
                           // A float32 matrix rounds a distance beyond its range to ±inf as it stores it.
                           + (width == Precision::Float32 ? " A float32 matrix (matrix_dtype float32) holds distances "
                                                            "up to 3.4e38 and stores a larger one as inf: set "
                                                            "matrix_dtype float64."
                                                          : ""));
      else if (d < 0 && !negative_is_distance) {
        std::array<char, 32> text{}; // the shortest round-trip form of a double is at most 24 characters
        char *const end = std::to_chars(text.data(), text.data() + text.size(), d).ptr;
        throw InvalidInput(pair(i, j) + std::string(text.data(), end) + "; only a Soft-DTW distance can be negative."
                           + (negative_way_out.empty() ? "" : " ") + std::string(negative_way_out));
      }
    }
  return all;
}

} // namespace

void refuse_float32_range(std::string_view where, size_t i, size_t j, double distance)
{
  std::array<char, 32> text{}; // the shortest round-trip form of a double is at most 24 characters
  char *const end = std::to_chars(text.data(), text.data() + text.size(), distance).ptr;
  throw InvalidInput(std::string(where) + ": the distance between series " + std::to_string(j) + " and "
                     + std::to_string(i) + " is " + std::string(text.data(), end)
                     + (std::isinf(distance) ? "; a distance must be finite."
                                             : ", beyond float32's range (3.4e38), which a float32 matrix holds "
                                               "(matrix_dtype float32): set matrix_dtype float64."));
}

size_t available_memory()
{
#ifdef DTWC_HAS_MMAP
  using usage = llfio::utils::process_memory_usage;
  const auto system = llfio::utils::current_process_memory_usage(usage::want::system_physical_memory_available);
  std::uint64_t bytes = system ? system.value().system_physical_memory_available : 0;
#ifdef __linux__
  // cgroup v2 ("0::<group>" in /proc/self/cgroup): the smallest memory.max of the
  // group and its ancestors; "max", no limit, reads as no number.
  std::ifstream cgroups("/proc/self/cgroup");
  for (std::string line; std::getline(cgroups, line);) {
    if (line.rfind("0::", 0) != 0) continue;
    for (fs::path group = line.substr(3);; group = group.parent_path()) {
      std::ifstream limit(fs::path("/sys/fs/cgroup") / group.relative_path() / "memory.max");
      std::uint64_t value = 0;
      if (limit >> value && (bytes == 0 || value < bytes)) bytes = value;
      if (group == group.parent_path()) break;
    }
  }
#endif
  return static_cast<size_t>(bytes);
#else
  return 0;
#endif
}

DistanceMatrix::DistanceMatrix() noexcept = default;

DistanceMatrix::DistanceMatrix(size_t n, Precision width)
  : heap_(std::make_unique_for_overwrite<std::byte[]>(packed_size(n) * value_bytes(width))), n_(n), rows_(n),
    width_(width)
{
  point_at(heap_.get());
  visit([&](auto *values) { fill_not_computed(values, packed_size(n)); });
}

DistanceMatrix::~DistanceMatrix() = default;

// A moved pointer keeps its buffer, so the values stay valid in the destination; the
// source is left empty rather than pointing into memory it no longer owns.
DistanceMatrix::DistanceMatrix(DistanceMatrix &&other) noexcept
  : heap_(std::move(other.heap_)), mapping_(std::move(other.mapping_)), f64_(std::exchange(other.f64_, nullptr)),
    f32_(std::exchange(other.f32_, nullptr)), n_(std::exchange(other.n_, 0)), rows_(std::exchange(other.rows_, 0)),
    width_(other.width_) {}

DistanceMatrix &DistanceMatrix::operator=(DistanceMatrix &&other) noexcept
{
  if (this != &other) {
    heap_ = std::move(other.heap_);
    mapping_ = std::move(other.mapping_);
    f64_ = std::exchange(other.f64_, nullptr);
    f32_ = std::exchange(other.f32_, nullptr);
    n_ = std::exchange(other.n_, 0);
    rows_ = std::exchange(other.rows_, 0);
    width_ = other.width_;
  }
  return *this;
}

void DistanceMatrix::point_at(std::byte *values) noexcept
{
  if (width_ == Precision::Float32)
    f32_ = reinterpret_cast<float *>(values);
  else
    f64_ = reinterpret_cast<double *>(values);
}

size_t DistanceMatrix::row_offset(size_t r) const noexcept
{
  return (mapping_ ? mapping_->header_bytes : sizeof(Header)) + packed_size(r) * value_bytes(width_);
}

void DistanceMatrix::resize(size_t n) { *this = DistanceMatrix(n, width_); }

double DistanceMatrix::max() const
{
  return visit([&](const auto *values) {
    double result = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < packed_size(rows_); ++i)
      if (!std::isnan(values[i]) && values[i] > result) result = values[i];
    return std::isfinite(result) ? result : 0.0;
  });
}

size_t DistanceMatrix::count_computed() const
{
  return visit([&](const auto *values) {
    return static_cast<size_t>(
      std::count_if(values, values + packed_size(rows_), [](auto d) { return !std::isnan(d); }));
  });
}

bool DistanceMatrix::all_computed(std::string_view where, bool negative_is_distance,
                                  std::string_view negative_way_out) const
{
  return visit([&](const auto *values) {
    return all_computed_in(values, n_, rows_, width_, where, negative_is_distance, negative_way_out);
  });
}

void DistanceMatrix::clear_rows(size_t r0, size_t r1)
{
  if (r0 >= r1) return;
  visit([&](auto *values) { fill_not_computed(values + packed_size(r0), packed_size(r1) - packed_size(r0)); });
}

std::pair<size_t, size_t> DistanceMatrix::first_infinite(size_t r0, size_t r1) const
{
  const size_t first = packed_size(r0), count = r0 < r1 ? packed_size(r1) - first : 0;
  const auto chunks = static_cast<size_t>(get_max_threads());
  std::vector<size_t> hit(chunks, count); // each chunk's first ±inf, its own slot
  visit([&](const auto *values) {
    auto scan = [&](size_t c) {
      const size_t end = count * (c + 1) / chunks;
      for (size_t k = count * c / chunks; k < end; ++k)
        if (std::isinf(values[first + k])) {
          hit[c] = k;
          return;
        }
    };
    run_openmp(scan, chunks, true, 1);
  });
  const size_t k = first + *std::min_element(hit.begin(), hit.end());
  if (k == first + count) return { n_, n_ };
  size_t i = r0;
  while (packed_size(i + 1) <= k) ++i;
  return { i, k - packed_size(i) };
}

DistanceMatrix DistanceMatrix::map(const fs::path &path, size_t n, const fingerprint_type &fingerprint, Precision width)
{
#ifdef DTWC_HAS_MMAP
  std::error_code ec;
  const bool exists = fs::exists(path, ec);
  if (ec) throw IOError("Cannot inspect " + shown(path) + ": " + ec.message());

  auto mapping = std::make_unique<Mapping>();
  mapping->path = path;
  std::byte *base = nullptr;
  Layout layout{ sizeof(Header), 0 };
  if (exists) {
    mapping->file = checked(llfio::mapped_file_handle::mapped_file(
                              0, {}, path, llfio::file_handle::mode::write, llfio::file_handle::creation::open_existing,
                              llfio::file_handle::caching::all, llfio::file_handle::flag::none),
                            "open", path);
    const auto length = checked(mapping->file.maximum_extent(), "measure", path);
    // A short file may not map at all: judge the length before reading a byte.
    char header[sizeof(Header)]{};
    if (length >= v4_header_bytes) {
      base = reinterpret_cast<std::byte *>(mapping->file.address());
      if (base == nullptr) throw IOError("Cannot map the distance matrix file " + shown(path) + ".");
      std::memcpy(header, base, static_cast<size_t>(std::min<std::uint64_t>(length, sizeof header)));
    }
    layout = check_header(header, length, n, fingerprint, width, path);
  } else {
    // llfio makes a new file sparse on NTFS unless told otherwise, and random
    // reads from a filled sparse file measured 1.9x slower.
    const size_t length = sizeof(Header) + packed_size(n) * value_bytes(width);
    mapping->file = checked(llfio::mapped_file_handle::mapped_file(
                              length, {}, path, llfio::file_handle::mode::write,
                              llfio::file_handle::creation::only_if_not_exist, llfio::file_handle::caching::all,
                              llfio::file_handle::flag::win_disable_sparse_file_creation),
                            "create", path);
    checked(mapping->file.truncate(length), "size", path);
    checked(mapping->file.update_map(), "map", path);
    base = reinterpret_cast<std::byte *>(mapping->file.address());
    if (base == nullptr) throw IOError("Cannot map the distance matrix file " + shown(path) + ".");
    // A new file is zeros, and a zero reads as a computed distance: its header
    // says no row is written, so none is read, and a fill makes each row NaN
    // before it computes it.
    const Header header = header_for(n, fingerprint, width, 0);
    std::memcpy(base, &header, sizeof header);
    flush_range(mapping->file, 0, sizeof header, llfio::mapped_file_handle::barrier_kind::wait_data_only, path);
  }

  DistanceMatrix matrix;
  matrix.width_ = width;
  matrix.point_at(base + layout.header_bytes);
  matrix.n_ = n;
  matrix.rows_ = layout.rows;
  mapping->header_bytes = layout.header_bytes;
  matrix.mapping_ = std::move(mapping);
  return matrix;
#else
  (void)n;
  (void)fingerprint;
  (void)width;
  throw IOError("Cannot map the distance matrix file " + shown(path)
                + ": this build has no memory-mapped support (rebuild with -DDTWC_ENABLE_LLFIO=ON).");
#endif
}

DistanceMatrix DistanceMatrix::read(const fs::path &path, size_t n, const fingerprint_type &fingerprint, Precision width)
{
  std::error_code ec;
  const std::uintmax_t length = fs::file_size(path, ec);
  if (ec) throw IOError("Cannot read the size of " + shown(path) + ": " + ec.message());
  std::ifstream in(path, std::ios::binary);
  if (!in) throw IOError("Cannot open " + shown(path) + " for reading.");
  char header[sizeof(Header)]{};
  in.read(header, static_cast<std::streamsize>(std::min<std::uintmax_t>(length, sizeof header)));
  const Layout layout = check_header(header, length, n, fingerprint, width, path);

  DistanceMatrix matrix(n, width);
  in.seekg(static_cast<std::streamoff>(layout.header_bytes));
  const auto bytes = static_cast<std::streamsize>(packed_size(layout.rows) * value_bytes(width));
  matrix.visit([&](auto *values) { in.read(reinterpret_cast<char *>(values), bytes); });
  if (in.gcount() != bytes) throw IOError("Cannot read the distances in " + shown(path) + ".");
  return matrix;
}

void DistanceMatrix::write(const fs::path &path, const fingerprint_type &fingerprint) const
{
  std::error_code ec;
  if (mapping_ && fs::equivalent(mapping_->path, path, ec)) {
    sync(); // renaming over a mapped file fails on Windows and orphans the map on POSIX
    return;
  }

  fs::path temporary = path;
  temporary += ".tmp";
  {
    auto out = open_output(temporary, std::ios::binary | std::ios::trunc);
    const Header header = header_for(n_, fingerprint, width_, rows_);
    out.write(reinterpret_cast<const char *>(&header), sizeof header);
    visit([&](const auto *values) {
      out.write(reinterpret_cast<const char *>(values), static_cast<std::streamsize>(packed_count() * value_bytes(width_)));
    });
    close_output(out, temporary);
  }
  flush_to_device(temporary);
  fs::rename(temporary, path, ec);
  if (ec) throw IOError("Cannot rename " + shown(temporary) + " to " + shown(path) + ": " + ec.message());
}

void DistanceMatrix::write(const fs::path &path, const fingerprint_type &fingerprint, size_t r0, size_t r1)
{
  std::error_code ec;
  if (mapping_ && fs::equivalent(mapping_->path, path, ec)) {
    flush_rows(r0, r1, true);
    return;
  }
  // In place only over this matrix's own file: the rows outside [r0, r1) there
  // are NaN or this matrix's distances, written by an earlier save.
  const Header expected = header_for(n_, fingerprint, width_, rows_);
  Header header{};
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  if (!file.read(reinterpret_cast<char *>(&header), sizeof header) || std::memcmp(&header, &expected, sizeof header) != 0
      || fs::file_size(path, ec) != row_offset(n_)) {
    file.close();
    write(path, fingerprint);
    return;
  }
  file.seekp(static_cast<std::streamoff>(row_offset(r0)));
  visit([&](const auto *values) {
    file.write(reinterpret_cast<const char *>(values + packed_size(r0)),
               static_cast<std::streamsize>((packed_size(r1) - packed_size(r0)) * value_bytes(width_)));
  });
  file.close();
  if (!file) throw IOError("Cannot write rows " + std::to_string(r0) + " to " + std::to_string(r1) + " of " + shown(path) + ".");
  flush_to_device(path);
}

void DistanceMatrix::prefetch_rows([[maybe_unused]] size_t r0, [[maybe_unused]] size_t r1) const
{
#ifdef DTWC_HAS_MMAP
  r1 = std::min(r1, rows_); // a row never written holds nothing to read
  if (!mapping_ || r0 >= r1) return;
  const size_t first = row_offset(r0) - row_offset(r0) % llfio::utils::page_size();
  const llfio::map_handle::buffer_type rows(mapping_->file.address() + first, row_offset(r1) - first);
  (void)llfio::map_handle::prefetch(rows);
#endif
}

void DistanceMatrix::flush_rows([[maybe_unused]] size_t r0, [[maybe_unused]] size_t r1, [[maybe_unused]] bool wait)
{
#ifdef DTWC_HAS_MMAP
  if (!mapping_ || r0 >= r1) return;
  using kind = llfio::mapped_file_handle::barrier_kind;
  flush_range(mapping_->file, row_offset(r0), row_offset(r1), wait ? kind::wait_data_only : kind::nowait_data_only,
              mapping_->path);
  // Every row before r0 was on the device (r0 <= rows_), and these are now: the header says so once they are,
  // never before, so a file that a crash stops holds no row it calls written and lacks.
  if (wait && r0 <= rows_ && rows_ < r1) {
    rows_ = r1;
    const std::uint64_t rows = r1;
    std::memcpy(mapping_->file.address() + rows_written_at, &rows, sizeof rows);
    flush_range(mapping_->file, 0, sizeof(Header), kind::nowait_data_only, mapping_->path);
  }
#endif
}

void DistanceMatrix::sync() const
{
#ifdef DTWC_HAS_MMAP
  if (mapping_)
    checked(mapping_->file.barrier({}, llfio::mapped_file_handle::barrier_kind::wait_data_only), "flush",
            mapping_->path);
#endif
}

} // namespace dtwc::core
