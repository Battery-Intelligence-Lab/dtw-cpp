/**
 * @file system_memory.cpp
 * @brief Platform free-RAM query behind dtwc::detail::available_ram_bytes().
 *
 * @details Deliberately a .cpp: the Windows branch needs <windows.h>, whose
 * ERROR/GetMessage/min/max macros must never reach DataLoader.hpp — a header
 * pulled in by the public umbrella <dtwc/dtwc.hpp> and therefore by every
 * consumer TU.
 */

#include "DataLoader.hpp" //!< For the dtwc::detail::available_ram_bytes() declaration

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#endif

namespace dtwc::detail {

std::size_t available_ram_bytes()
{
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (::GlobalMemoryStatusEx(&status) == 0) return 0;
  // ullAvailPhys is 64-bit; clamp for 32-bit hosts where size_t is narrower.
  constexpr auto cap =
    static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  return static_cast<std::size_t>(
    status.ullAvailPhys < cap ? status.ullAvailPhys : cap);
#elif defined(__linux__)
  const long pages = ::sysconf(_SC_AVPHYS_PAGES);
  const long psize = ::sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && psize > 0)
    return static_cast<std::size_t>(pages) * static_cast<std::size_t>(psize);
  return 0;
#elif defined(__APPLE__)
  // free + inactive pages: inactive pages are reclaimable without swapping, so
  // this is the closest analogue of the Windows ullAvailPhys figure. HW_MEMSIZE
  // (total RAM) was used before and made the Auto threshold far too permissive.
  const mach_port_t host = ::mach_host_self();
  vm_statistics64_data_t vmstat{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_size_t page_size = 0;
  const kern_return_t stats = ::host_statistics64(
    host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vmstat), &count);
  const kern_return_t page = ::host_page_size(host, &page_size);
  ::mach_port_deallocate(::mach_task_self(), host);
  if (stats != KERN_SUCCESS || page != KERN_SUCCESS || page_size == 0) return 0;
  const std::uint64_t free_pages = static_cast<std::uint64_t>(vmstat.free_count)
                                 + static_cast<std::uint64_t>(vmstat.inactive_count);
  const std::uint64_t bytes = free_pages * static_cast<std::uint64_t>(page_size);
  constexpr auto cap =
    static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  return static_cast<std::size_t>(bytes < cap ? bytes : cap);
#else
  return 0;
#endif
}

} // namespace dtwc::detail
