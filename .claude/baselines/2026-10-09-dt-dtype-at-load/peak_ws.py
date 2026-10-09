"""Run a command and print its peak working set (and peak private commit) once it has exited,
read with GetProcessMemoryInfo on the process handle the wrapper still holds.

usage: peak_ws.py <exe> [args...]   (stdlib only; Windows)

@author Volkan Kumtepeli
@author Claude Opus 5.5
"""
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


proc = subprocess.Popen(sys.argv[1:], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
_, err = proc.communicate()
counters = PROCESS_MEMORY_COUNTERS()
counters.cb = ctypes.sizeof(counters)
psapi = ctypes.WinDLL("psapi")
psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wt.DWORD]
if not psapi.GetProcessMemoryInfo(wt.HANDLE(int(proc._handle)), ctypes.byref(counters), counters.cb):
    raise ctypes.WinError()
print(f"exit={proc.returncode} peak_working_set={counters.PeakWorkingSetSize} "
      f"peak_private={counters.PeakPagefileUsage} page_faults={counters.PageFaultCount}")
print("stderr:", err.decode(errors="replace").strip()[:300])
