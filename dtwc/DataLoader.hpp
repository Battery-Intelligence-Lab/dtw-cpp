/**
 * @file DataLoader.hpp
 * @brief Encapsulating DTWC data loading configurations in a class.
 * Uses method chaining for easier input taking.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 04 Dec 2022
 */

#pragma once

#include "Data.hpp"           //!< For Data class
#include "fileOperations.hpp" //!< For load_batch_file(), load_folder(), ignoreBOM()
#include "settings.hpp"       //!< For data_t type
#include "core/storage.hpp"   //!< For core::StoragePolicy
#include "env.hpp"            //!< For dtwc::env(), dtwc::Device (Task 1.3, built concurrently)

#include <cstddef>    //!< For size_t
#include <cstdint>    //!< For uintptr_t
#include <filesystem> //!< For filesystem objects like path
#include <fstream>    //!< For std::ifstream (used in count())
#include <iostream>   //!< For std::cerr (loud no-mmap warning)
#include <limits>     //!< For std::numeric_limits
#include <memory>     //!< For std::unique_ptr
#include <span>       //!< For std::span (mmap-view construction)
#include <sstream>    //!< For std::istringstream (metadata field counting)
#include <string>     //!< For std::string
#include <string_view>//!< For std::string_view (mmap-view names)
#include <tuple>      //!< For std::tie(), std::tuple
#include <vector>     //!< For std::vector

#ifdef DTWC_HAS_MMAP
#include "core/mmap_data_store.hpp" //!< For core::MmapDataStore (mmap-backed store; llfio)
#endif

// Platform free-RAM query for the StoragePolicy::Auto default threshold. Lightweight
// POSIX headers only — deliberately NO <windows.h> in this widely-included header
// (it is unused elsewhere in the core and would leak min/max macros).
#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace dtwc {

namespace detail {
/// @brief Best-effort free physical RAM in bytes.
/// @details Returns 0 when the platform query is unavailable (e.g. Windows here);
///          callers then treat the auto-mmap threshold as effectively unlimited, so
///          StoragePolicy::Auto stays on heap unless an explicit ram_limit() is set.
inline std::size_t available_ram_bytes()
{
#if defined(__linux__)
  const long pages = ::sysconf(_SC_AVPHYS_PAGES);
  const long psize = ::sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && psize > 0)
    return static_cast<std::size_t>(pages) * static_cast<std::size_t>(psize);
  return 0;
#elif defined(__APPLE__)
  std::uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  int mib[2] = { CTL_HW, HW_MEMSIZE }; // total physical (a safe upper proxy for free)
  if (::sysctl(mib, 2, &mem, &len, nullptr, 0) == 0)
    return static_cast<std::size_t>(mem);
  return 0;
#else
  return 0;
#endif
}
} // namespace detail

/**
 * @brief Result of DataLoader::load_stored() — the Data plus, when the storage
 *        policy routes to the mmap-backed store, the owning MmapDataStore and the
 *        name strings the returned Data's view spans reference.
 * @details Move-only when mmap is compiled in (owns a unique_ptr). On the mmap
 *          route `data` is a view into `store` — the same view-mode span machinery
 *          CLARA uses (preserved untouched); `store`/`names` keep it alive.
 */
struct LoadedData
{
  Data data; //!< Heap-owning (heap route), view-into-store (mmap route), or metadata-only (hpc).
#ifdef DTWC_HAS_MMAP
  std::unique_ptr<core::MmapDataStore> store; //!< Backing store; non-null iff mmap-routed.
  std::vector<std::string> names;             //!< Owns the strings `data`'s name views point into.
  bool is_mmap() const { return store != nullptr; }
#else
  bool is_mmap() const { return false; }
#endif
};

/**
 * @brief Data loader class
 */
class DataLoader
{
  int start_col{ 0 };                     //!< Starting column for data extraction
  int start_row{ 0 };                     //!< Starting row for data extraction
  int Ndata{ -1 };                        //!< Number of data rows to load
  int verbose{ 1 };                       //!< Verbosity level
  char delim{ ',' };                      //!< Column delimiter character
  std::filesystem::path data_path{ "." }; //!< Path to data file or folder

  core::StoragePolicy storage_policy_{ core::StoragePolicy::Auto }; //!< Routing policy for load_stored().
  std::size_t ram_limit_bytes_{ 0 };      //!< Footprint threshold override (bytes); 0 = default (50% free RAM).
  std::filesystem::path mmap_cache_path_{}; //!< mmap store file for Auto/Mmap routes; empty = a temp file.

  /// Count of bulk-reader (load_folder/load_batch_file) invocations — instrumentation
  /// for the metadata-only load test (proves the hpc path materialises no payload).
  static inline std::size_t s_bulk_read_invocations = 0;

public:
  // Constructors
  DataLoader() = default;                                  //!< Default constructor.
  DataLoader(const fs::path &path_) { this->path(path_); } //!< Constructor with path initialization.
  DataLoader(const fs::path &path_, int Ndata_)
  {
    this->path(path_);
    this->n_data(Ndata_);
  }

  // Accessor methods
  auto startColumn() { return start_col; } //!< Get the starting column for data loading.
  auto startRow() { return start_row; }    //!< Get the starting row for data loading.
  auto n_data() { return Ndata; }          //!< Get the number of data points to load.
  auto delimiter() { return delim; }       //!< Get the delimiter used in data files.
  auto path() { return data_path; }        //!< Get the path of the data file or directory.
  auto verbosity() { return verbose; }     //!< Get the verbosity level for data loading.

  auto storage_policy() { return storage_policy_; }   //!< Get the storage routing policy.
  auto ram_limit() { return ram_limit_bytes_; }       //!< Get the footprint threshold override (bytes; 0 = default).
  auto mmap_cache_path() { return mmap_cache_path_; } //!< Get the mmap store file path (empty = temp file).

  /// Bulk-reader invocation count (test instrumentation).
  static std::size_t bulk_read_count() { return s_bulk_read_invocations; }
  /// Reset the bulk-reader invocation counter (test instrumentation).
  static void reset_bulk_read_count() { s_bulk_read_invocations = 0; }


  // Setters with chaining

  /**
   * @brief Set start column
   * @param N Starting column
   * @return Reference to self for chaining
   */
  DataLoader &startColumn(int N)
  {
    start_col = N;
    return *this;
  }

  //!< Set start row
  DataLoader &startRow(int N)
  {
    start_row = N;
    return *this;
  }

  //!< Set number of data rows
  DataLoader &n_data(int N)
  {
    Ndata = N;
    return *this;
  }
  //!< Set delimiter
  DataLoader &delimiter(char delim_)
  {
    delim = delim_;
    return *this;
  }

  /**
   * @brief Set data path
   *
   * Sets delimiter based on file extension
   *
   * @param data_path_ Path to data
   * @return Reference to self for chaining
   */
  DataLoader &path(const std::filesystem::path &data_path_)
  {
    data_path = data_path_;
    if (data_path_.extension() == ".csv")
      delim = ',';
    else if (data_path_.extension() == ".tsv" || data_path_.extension() == ".txt")
      delim = '\t';
    return *this;
  }

  //!< Set verbosity level
  DataLoader &verbosity(int N)
  {
    verbose = N;
    return *this;
  }

  //!< Set the storage routing policy (Auto/Heap/Mmap) used by load_stored().
  DataLoader &storage_policy(core::StoragePolicy p)
  {
    storage_policy_ = p;
    return *this;
  }

  //!< Set the footprint threshold override in bytes (0 = default: 50% of free RAM).
  //!< Auto routes to the mmap store when the estimated footprint exceeds this.
  DataLoader &ram_limit(std::size_t bytes)
  {
    ram_limit_bytes_ = bytes;
    return *this;
  }

  //!< Set the mmap store file path for Auto/Mmap routes (empty = a temp file).
  DataLoader &mmap_cache_path(std::filesystem::path p)
  {
    mmap_cache_path_ = std::move(p);
    return *this;
  }

  /**
   * @brief Load data.
   * @details Calls the appropriate loader based on the path being a file or folder.
   *          When the process device is 'hpc' (dtwc::env().device()), performs a
   *          metadata-only load (shapes/counts/names) instead — the bulk payload is
   *          streamed to the cluster at submit, and local series access then throws
   *          (api-contract-2.0.md §6.3). Local devices ('cpu'/'gpu') load into heap.
   * @return Loaded data (heap on local devices; metadata-only on 'hpc').
   */
  Data load()
  {
    if (dtwc::env().device() == dtwc::Device::HPC)
      return load_metadata();
    return load_heap();
  }

  /**
   * @brief Materialise the payload irrespective of the process-wide device.
   * @details The Tier-1 `cluster(..., device="cpu")` API supports a per-call
   *          override.  That override must not mutate the global Env merely so
   *          DataLoader::load() takes its local branch.  This explicit entry
   *          point keeps the loader policy visible and avoids hidden global
   *          state changes; normal callers should continue to use load().
   */
  Data load_local() { return load_heap(); }

  /**
   * @brief Metadata-only load: reads shapes/counts/names WITHOUT materialising any
   *        series payload — it never calls the bulk reader (load_folder/load_batch_file).
   * @details This is the 'hpc' load path (api-contract-2.0.md §6.3). The returned
   *          Data reports the correct size()/series_length()/name(), but series()/
   *          series_f32() throw ("data not resident locally").
   * @return Metadata-only Data.
   */
  Data load_metadata()
  {
    if (fs::is_directory(data_path))
      return load_metadata_folder();
    return load_metadata_file();
  }

  /**
   * @brief Storage-policy-aware load (StoragePolicy::Auto/Heap/Mmap routing, Task 1.4).
   * @details Auto spills to the mmap-backed store when the estimated footprint
   *          (rows x lengths x sizeof(data_t)) exceeds the threshold (ram_limit(),
   *          else 50% of free RAM). The mmap route returns a Data that is a view into
   *          the store — the SAME view-mode span machinery CLARA uses (preserved
   *          untouched) — bundled with the owning store so the view stays valid. On
   *          device='hpc' this returns a metadata-only Data (no store). No silent
   *          fallback: an explicit Mmap request without llfio compiled in throws.
   * @return LoadedData bundle (see is_mmap()).
   */
  LoadedData load_stored()
  {
    LoadedData out;
    if (dtwc::env().device() == dtwc::Device::HPC) {
      out.data = load_metadata();
      return out;
    }

    Data heap = load_heap();
    const std::size_t footprint = footprint_bytes(heap);

    std::size_t threshold;
    if (ram_limit_bytes_ > 0)
      threshold = ram_limit_bytes_;
    else {
      const std::size_t avail = detail::available_ram_bytes();
      threshold = (avail > 0) ? (avail / 2) : std::numeric_limits<std::size_t>::max();
    }

    const bool want_mmap =
      (storage_policy_ == core::StoragePolicy::Mmap)
      || (storage_policy_ == core::StoragePolicy::Auto && footprint > threshold);

    if (!want_mmap) {
      out.data = std::move(heap);
      return out;
    }

#ifdef DTWC_HAS_MMAP
    const std::filesystem::path cache =
      mmap_cache_path_.empty() ? default_cache_path() : mmap_cache_path_;
    // Spill the heap data to the mmap store, then hand back a non-owning view into it.
    auto store = std::make_unique<core::MmapDataStore>(
      core::MmapDataStore::create(cache, heap));
    out.names = std::move(heap.p_names); // own the strings the view name-spans reference
    const std::size_t n = store->size();
    std::vector<std::span<const data_t>> spans;
    std::vector<std::string_view> name_views;
    spans.reserve(n);
    name_views.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      spans.push_back(store->series(i));
      name_views.push_back(std::string_view(out.names[i]));
    }
    out.data = Data(std::move(spans), std::move(name_views), store->ndim());
    out.store = std::move(store);
    return out;
#else
    if (storage_policy_ == core::StoragePolicy::Mmap)
      throw std::runtime_error(
        "DataLoader::load_stored: StoragePolicy::Mmap requested but mmap support "
        "(llfio) is not compiled in. Rebuild with -DDTWC_ENABLE_LLFIO=ON.");
    // Auto wanted mmap but llfio is absent: warn loudly and keep heap. Auto is
    // best-effort, so this is a LOUD degrade, never a silent one (global constraint).
    std::cerr << "[dtwc] warning: dataset footprint (" << footprint
              << " B) exceeds the storage threshold (" << threshold
              << " B) but mmap support is not compiled in; keeping data in RAM. "
                 "Rebuild with -DDTWC_ENABLE_LLFIO=ON to enable the mmap-backed store.\n";
    out.data = std::move(heap);
    return out;
#endif
  }

  /**
   * @brief Count series without loading data.
   * @details Directory mode: counts entries via directory_iterator.
   *          Batch file mode: counts lines (skips start_row headers).
   *          Respects Ndata limit in both modes.
   * @return Number of series that would be loaded.
   */
  size_t count() const
  {
    if (fs::is_directory(data_path)) {
      size_t n = 0;
      for ([[maybe_unused]] const auto &entry : fs::directory_iterator(data_path)) {
        ++n;
        if (Ndata >= 0 && static_cast<int>(n) >= Ndata) break;
      }
      return n;
    }
    // Batch file: count lines after skipping start_row, respecting Ndata
    std::ifstream in(data_path, std::ios_base::in);
    if (!in.good())
      throw std::runtime_error("DataLoader::count: cannot open " + data_path.string());

    std::string line;
    int line_no = 0;
    size_t n_rows = 0;
    while (std::getline(in, line)) {
      if (line_no++ < start_row) continue;
      ++n_rows;
      if (Ndata >= 0 && static_cast<int>(n_rows) >= Ndata) break;
    }
    return n_rows;
  }

private:
  /// Raw heap load (the local-device path). Increments the bulk-reader counter — this
  /// is the payload-materialising path that the metadata-only load must NOT touch.
  Data load_heap()
  {
    Data d;
    const LoadOptions opts{ Ndata, verbose, start_row, start_col, delim };
    ++s_bulk_read_invocations;
    if (fs::is_directory(data_path))
      std::tie(d.p_vec, d.p_names) = load_folder<data_t>(data_path, opts);
    else
      std::tie(d.p_vec, d.p_names) = load_batch_file<data_t>(data_path, opts);
    return d;
  }

  /// Batch-file metadata: series count + per-row field count + row-number names.
  /// Field counting mirrors load_batch_file's extraction EXACTLY (same start_row/
  /// start_col skips, same delimiter handling) so shapes equal a full load — but the
  /// numbers are counted, never stored (no payload materialised, bulk reader untouched).
  Data load_metadata_file()
  {
    std::ifstream in(data_path, std::ios_base::in);
    if (!in.good())
      throw std::runtime_error("DataLoader::load_metadata: cannot open " + data_path.string());
    ignoreBOM(in);

    std::vector<std::string> names;
    std::vector<std::size_t> flat_sizes;
    std::string line;
    int line_no = 0;
    int n_rows = 0;
    while ((Ndata == -1 || n_rows < Ndata) && std::getline(in, line)) {
      if (line_no++ < start_row) continue;
      ++n_rows;
      const std::size_t count = text_io_detail::parse_numeric_row<data_t>(
        line, data_path, static_cast<std::size_t>(line_no), start_col, delim,
        [](data_t) {});
      names.push_back(std::to_string(n_rows));
      flat_sizes.push_back(count);
    }
    return Data::metadata_only(std::move(names), std::move(flat_sizes), 1);
  }

  /// Folder metadata: one series per file, name = file stem, shape = value count.
  Data load_metadata_folder()
  {
    std::vector<std::string> names;
    std::vector<std::size_t> flat_sizes;
    int i_data = 0;
    for (const auto &entry : fs::directory_iterator(data_path)) {
      names.push_back(entry.path().stem().string());
      flat_sizes.push_back(count_series_values(entry.path()));
      if (++i_data == Ndata) break;
    }
    return Data::metadata_only(std::move(names), std::move(flat_sizes), 1);
  }

  /// Count the values readFile<data_t> would extract from one file (one value per
  /// data line after skipping start_row rows / start_col columns) — without storing.
  std::size_t count_series_values(const fs::path &file) const
  {
    std::ifstream in(file, std::ios_base::in);
    if (!in.good())
      throw std::runtime_error("DataLoader::load_metadata: cannot open " + file.string());
    ignoreBOM(in);
    std::string line;
    for (int i = 0; i < start_row; ++i) std::getline(in, line);
    std::size_t count = 0;
    std::size_t row = static_cast<std::size_t>(start_row);
    bool first_data_line = true;
    while (std::getline(in, line)) {
      ++row;
      const auto value = text_io_detail::parse_series_value_row<data_t>(
        line, file, row, start_col, delim, first_data_line);
      first_data_line = false;
      if (value) ++count;
    }
    return count;
  }

  /// Estimated in-RAM footprint of `d` in bytes (rows x lengths x sizeof(data_t)).
  static std::size_t footprint_bytes(const Data &d)
  {
    std::size_t f = 0;
    const std::size_t n = d.size();
    for (std::size_t i = 0; i < n; ++i)
      f += d.series_flat_size(i) * sizeof(data_t);
    return f;
  }

#ifdef DTWC_HAS_MMAP
  /// A unique temp-file path for the mmap store when the caller sets none.
  static std::filesystem::path default_cache_path()
  {
    static std::size_t counter = 0;
    const auto uniq = std::to_string(reinterpret_cast<std::uintptr_t>(&counter))
                    + "_" + std::to_string(counter++);
    return std::filesystem::temp_directory_path() / ("dtwc_store_" + uniq + ".dtws");
  }
#endif
};

} // namespace dtwc
