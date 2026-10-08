#!/bin/sh
# load.sh <label> [windows]: XG's load record (C:/D/git/wt/tmp/XG/load.sh, itself TB's approach). Windows: total CPU from
# typeperf (10 one-second samples), the ten busiest processes now in % of all logical processors (the Process counter
# is % of one), WSL's VM by name (vmmemWSL, whose CPU is every vCPU's), any COMSOL process. Then WSL: uptime (load
# averages), the systemd jobs running, the busiest processes over one second (top's second frame); `windows` skips WSL. Appends to load.txt.
PATH=/usr/bin:$PATH
{
  echo "== $1 $(date '+%Y-%m-%d %H:%M:%S')"
  typeperf "\Processor(_Total)\% Processor Time" -sc 10
  powershell -NoProfile -Command '$n = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors; "logical processors: $n"; (Get-Counter "\Process(*)\% Processor Time" -SampleInterval 3 -MaxSamples 1 -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.InstanceName -notin @("_total","idle") } | Sort-Object CookedValue -Descending | Select-Object -First 10 | ForEach-Object { "{0,-28} {1,6:N2} % of all cores" -f $_.InstanceName, ($_.CookedValue / $n) }; (Get-Counter "\Process(vmmem*)\% Processor Time" -ErrorAction SilentlyContinue).CounterSamples | ForEach-Object { "WSL VM {0,-21} {1,6:N2} % of all cores" -f $_.InstanceName, ($_.CookedValue / $n) }; "comsol processes: " + @(Get-Process -Name *comsol* -ErrorAction SilentlyContinue).Count'
  if [ "$2" != windows ]; then  # the cl-only re-time does not boot WSL
    echo "-- WSL"
    wsl.exe -d Ubuntu-24.04 -e sh -c 'uptime; echo "systemd jobs: $(systemctl list-jobs --no-legend | wc -l)"; systemctl list-jobs --no-legend; COLUMNS=200 top -b -n 2 -d 1 -w 200 -o %CPU | awk "/^top -/ { n++ } n == 2" | head -n 17'
  fi
} 2>&1 | tee -a "$(dirname "$0")/load.txt"
