#!/bin/sh
# load.sh <label>: total CPU (typeperf, 10 one-second samples) and the ten busiest processes now, in % of all
# logical processors (the Process counter is % of one), plus any COMSOL process. Appends to load.txt.
# (The approach of C:/D/git/wt/tmp/TB/load.sh.)
PATH=/usr/bin:$PATH
{
  echo "== $1 $(date '+%Y-%m-%d %H:%M:%S')"
  typeperf "\Processor(_Total)\% Processor Time" -sc 10
  powershell -NoProfile -Command '$n = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors; "logical processors: $n"; (Get-Counter "\Process(*)\% Processor Time" -SampleInterval 3 -MaxSamples 1).CounterSamples | Where-Object { $_.InstanceName -notin @("_total","idle") } | Sort-Object CookedValue -Descending | Select-Object -First 10 | ForEach-Object { "{0,-28} {1,6:N2} % of all cores" -f $_.InstanceName, ($_.CookedValue / $n) }; "comsol processes: " + @(Get-Process -Name *comsol* -ErrorAction SilentlyContinue).Count'
} 2>&1 | tee -a "$(dirname "$0")/load.txt"
