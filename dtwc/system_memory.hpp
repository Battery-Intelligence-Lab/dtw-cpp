/**
 * @file system_memory.hpp
 * @brief Declaration of the platform free-RAM query implemented in system_memory.cpp.
 *
 * Exists so that system_memory.cpp has somewhere in the *base* layer to get its
 * own declaration from (ledger X-05). It previously included DataLoader.hpp for
 * that one line, which was the only upward include edge in the tree that did not
 * point at Problem.hpp.
 *
 * Deliberately no platform headers here: <windows.h> stays inside
 * system_memory.cpp, because this declaration reaches every consumer through
 * DataLoader.hpp and <dtwc/dtwc.hpp>, and would leak ERROR, GetMessage and the
 * min/max macros.
 */

#pragma once

#include <cstddef>

namespace dtwc::detail {

/// @brief Best-effort free physical RAM in bytes (see system_memory.cpp).
/// @details The platforms report three different quantities, all "memory a new
///          allocation can plausibly use right now":
///          Linux `_SC_AVPHYS_PAGES` — MemFree, EXCLUDING reclaimable page cache;
///          Windows `MEMORYSTATUSEX::ullAvailPhys` — free + standby (≈ MemAvailable);
///          macOS `host_statistics64(HOST_VM_INFO64)` — free + inactive pages.
///          They are NOT comparable across platforms: the same dataset can spill
///          to mmap on Linux and stay on the heap on Windows or macOS.
/// @return 0 means UNKNOWN — the platform has no supported query, or the query
///         failed. Unknown is not "no free RAM": callers must treat it as "do not
///         auto-spill" (see choose_storage), so StoragePolicy::Auto stays on heap
///         unless an explicit ram_limit() is set.
std::size_t available_ram_bytes();

} // namespace dtwc::detail
