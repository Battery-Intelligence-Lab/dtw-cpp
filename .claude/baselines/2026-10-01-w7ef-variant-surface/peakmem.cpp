// peakmem <affinity mask, hex; 0 = any CPU> <program> [args...]
// Runs the program pinned to the mask, waits for it, and prints its exit code, wall
// time, peak working set and peak private bytes (GetProcessMemoryInfo on the exited
// process). W7ef measurement tool; not part of the library.
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: peakmem <hex mask> <program> [args...]\n");
    return 2;
  }
  const auto mask = static_cast<DWORD_PTR>(std::strtoull(argv[1], nullptr, 16));
  std::string command;
  for (int i = 2; i < argc; ++i) {
    if (i > 2) command += ' ';
    command += '"';
    command += argv[i];
    command += '"';
  }
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  const auto t0 = std::chrono::steady_clock::now();
  if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &si,
                      &pi)) {
    std::fprintf(stderr, "CreateProcess failed: %lu\n", GetLastError());
    return 2;
  }
  if (mask != 0 && !SetProcessAffinityMask(pi.hProcess, mask)) {
    std::fprintf(stderr, "SetProcessAffinityMask failed: %lu\n", GetLastError());
    TerminateProcess(pi.hProcess, 2);
    return 2;
  }
  ResumeThread(pi.hThread);
  WaitForSingleObject(pi.hProcess, INFINITE);
  const auto t1 = std::chrono::steady_clock::now();
  PROCESS_MEMORY_COUNTERS_EX c{};
  c.cb = sizeof(c);
  const BOOL ok = GetProcessMemoryInfo(pi.hProcess, reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&c), sizeof(c));
  DWORD code = 0;
  GetExitCodeProcess(pi.hProcess, &code);
  const double MB = 1024.0 * 1024.0;
  std::printf("PEAKMEM exit=%lu wall_s=%.3f peak_working_set_MB=%.1f peak_private_MB=%.1f counters_ok=%d\n", code,
              std::chrono::duration<double>(t1 - t0).count(), c.PeakWorkingSetSize / MB, c.PeakPagefileUsage / MB,
              ok ? 1 : 0);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  return 0;
}
