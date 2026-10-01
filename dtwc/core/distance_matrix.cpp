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
#include "../fileOperations.hpp" // open_output, close_output

#include <algorithm>
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
};

namespace {

constexpr double not_computed = std::numeric_limits<double>::quiet_NaN();

struct Header
{
  char magic[4];
  std::uint32_t version;
  std::uint64_t n;
  DistanceMatrix::fingerprint_type fingerprint;
};
static_assert(sizeof(Header) == 48 && std::is_trivially_copyable_v<Header>);

constexpr char magic[4] = { 'D', 'T', 'W', 'M' };
constexpr std::uint32_t version = 4;

Header header_for(size_t n, const DistanceMatrix::fingerprint_type &fingerprint)
{
  Header header{};
  std::memcpy(header.magic, magic, sizeof magic);
  header.version = version;
  header.n = n;
  header.fingerprint = fingerprint;
  return header;
}

std::string shown(const fs::path &path)
{
  const auto utf8 = path.u8string();
  return "'" + std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size()) + "'";
}

/// The checks every open of a `.dtwm` file makes, before any distance is
/// exposed. `bytes` holds the file's first min(length, 48) bytes.
void check_header(const char *bytes, std::uint64_t length, size_t n,
                  const DistanceMatrix::fingerprint_type &fingerprint, const fs::path &path)
{
  const std::string fix = " Delete or rename it to compute the distances again.";
  if (length < sizeof(Header))
    throw IOError(shown(path) + " is " + std::to_string(length)
                  + " bytes, too short for a .dtwm file." + fix);
  Header header;
  std::memcpy(&header, bytes, sizeof header);
  if (std::memcmp(header.magic, magic, sizeof magic) != 0)
    throw IOError(shown(path) + " is not a .dtwm distance matrix (no DTWM magic)." + fix);
  if (header.version != version)
    throw IOError(shown(path) + " is a .dtwm file of version " + std::to_string(header.version)
                  + "; this build reads version " + std::to_string(version) + "." + fix);
  // The N bound keeps packed_size exact in 64 bits; a file that large cannot exist.
  const std::uint64_t payload = length - sizeof(Header);
  if (header.n >= (std::uint64_t{ 1 } << 32) || payload % sizeof(double) != 0
      || payload / sizeof(double) != packed_size(static_cast<size_t>(header.n)))
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
                       "data, band, variant, missing-data strategy, metric, precision and device, "
                       "or another path.");
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
#endif

} // namespace

DistanceMatrix::DistanceMatrix() noexcept = default;

DistanceMatrix::DistanceMatrix(size_t n)
  : heap_(packed_size(n), not_computed), data_(heap_.data()), n_(n) {}

DistanceMatrix::~DistanceMatrix() = default;

// A moved vector keeps its buffer, so data_ stays valid in the destination; the
// source is left empty rather than pointing into memory it no longer owns.
DistanceMatrix::DistanceMatrix(DistanceMatrix &&other) noexcept
  : heap_(std::move(other.heap_)), mapping_(std::move(other.mapping_)),
    data_(std::exchange(other.data_, nullptr)), n_(std::exchange(other.n_, 0)) {}

DistanceMatrix &DistanceMatrix::operator=(DistanceMatrix &&other) noexcept
{
  if (this != &other) {
    heap_ = std::move(other.heap_);
    mapping_ = std::move(other.mapping_);
    data_ = std::exchange(other.data_, nullptr);
    n_ = std::exchange(other.n_, 0);
  }
  return *this;
}

void DistanceMatrix::resize(size_t n) { *this = DistanceMatrix(n); }

double DistanceMatrix::max() const
{
  double result = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < packed_count(); ++i)
    if (!std::isnan(data_[i]) && data_[i] > result) result = data_[i];
  return std::isfinite(result) ? result : 0.0;
}

size_t DistanceMatrix::count_computed() const
{
  return static_cast<size_t>(std::count_if(data_, data_ + packed_count(), [](double d) { return !std::isnan(d); }));
}

bool DistanceMatrix::all_computed(std::string_view where) const
{
  bool all = true;
  for (size_t i = 0, k = 0; i < n_; ++i)
    for (size_t j = 0; j <= i; ++j, ++k) {
      const double d = data_[k];
      if (std::isnan(d))
        all = false;
      else if (std::isinf(d))
        throw InvalidInput(std::string(where) + ": the distance between series " + std::to_string(j) + " and "
                           + std::to_string(i) + " is " + (d > 0 ? "+inf" : "-inf") + "; a distance must be finite.");
    }
  return all;
}

DistanceMatrix DistanceMatrix::map(const fs::path &path, size_t n, const fingerprint_type &fingerprint)
{
#ifdef DTWC_HAS_MMAP
  std::error_code ec;
  const bool exists = fs::exists(path, ec);
  if (ec) throw IOError("Cannot inspect " + shown(path) + ": " + ec.message());

  auto mapping = std::make_unique<Mapping>();
  mapping->path = path;
  std::byte *base = nullptr;
  if (exists) {
    mapping->file = checked(llfio::mapped_file_handle::mapped_file(
                              0, {}, path, llfio::file_handle::mode::write, llfio::file_handle::creation::open_existing,
                              llfio::file_handle::caching::all, llfio::file_handle::flag::none),
                            "open", path);
    const auto length = checked(mapping->file.maximum_extent(), "measure", path);
    // A short file may not map at all: judge the length before reading a byte.
    char header[sizeof(Header)]{};
    if (length >= sizeof(Header)) {
      base = reinterpret_cast<std::byte *>(mapping->file.address());
      if (base == nullptr) throw IOError("Cannot map the distance matrix file " + shown(path) + ".");
      std::memcpy(header, base, sizeof header);
    }
    check_header(header, length, n, fingerprint, path);
  } else {
    // llfio makes a new file sparse on NTFS unless told otherwise, and random
    // reads from a filled sparse file measured 1.9x slower.
    const size_t length = sizeof(Header) + packed_size(n) * sizeof(double);
    mapping->file = checked(llfio::mapped_file_handle::mapped_file(
                              length, {}, path, llfio::file_handle::mode::write,
                              llfio::file_handle::creation::only_if_not_exist, llfio::file_handle::caching::all,
                              llfio::file_handle::flag::win_disable_sparse_file_creation),
                            "create", path);
    checked(mapping->file.truncate(length), "size", path);
    checked(mapping->file.update_map(), "map", path);
    base = reinterpret_cast<std::byte *>(mapping->file.address());
    if (base == nullptr) throw IOError("Cannot map the distance matrix file " + shown(path) + ".");
    // A new file is zeros, and a zero reads as a computed distance: the NaN
    // payload is on the device before the header that makes the file readable.
    std::fill_n(reinterpret_cast<double *>(base + sizeof(Header)), packed_size(n), not_computed);
    checked(mapping->file.barrier({}, llfio::mapped_file_handle::barrier_kind::wait_all), "flush", path);
    const Header header = header_for(n, fingerprint);
    std::memcpy(base, &header, sizeof header);
  }

  DistanceMatrix matrix;
  matrix.data_ = reinterpret_cast<double *>(base + sizeof(Header));
  matrix.n_ = n;
  matrix.mapping_ = std::move(mapping);
  return matrix;
#else
  (void)n;
  (void)fingerprint;
  throw IOError("Cannot map the distance matrix file " + shown(path)
                + ": this build has no memory-mapped support (rebuild with -DDTWC_ENABLE_LLFIO=ON).");
#endif
}

DistanceMatrix DistanceMatrix::read(const fs::path &path, size_t n, const fingerprint_type &fingerprint)
{
  std::error_code ec;
  const std::uintmax_t length = fs::file_size(path, ec);
  if (ec) throw IOError("Cannot read the size of " + shown(path) + ": " + ec.message());
  std::ifstream in(path, std::ios::binary);
  if (!in) throw IOError("Cannot open " + shown(path) + " for reading.");
  char header[sizeof(Header)]{};
  in.read(header, static_cast<std::streamsize>(std::min<std::uintmax_t>(length, sizeof header)));
  check_header(header, length, n, fingerprint, path);

  DistanceMatrix matrix(n);
  const auto bytes = static_cast<std::streamsize>(matrix.packed_count() * sizeof(double));
  in.read(reinterpret_cast<char *>(matrix.data_), bytes);
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
    const Header header = header_for(n_, fingerprint);
    out.write(reinterpret_cast<const char *>(&header), sizeof header);
    out.write(reinterpret_cast<const char *>(data_), static_cast<std::streamsize>(packed_count() * sizeof(double)));
    close_output(out, temporary);
  }
  flush_to_device(temporary);
  fs::rename(temporary, path, ec);
  if (ec) throw IOError("Cannot rename " + shown(temporary) + " to " + shown(path) + ": " + ec.message());
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
