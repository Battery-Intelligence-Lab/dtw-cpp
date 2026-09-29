@echo off
rem PF-5 single-thread timing, pinned to logical CPU 22 (a P-core): affinity mask 0x400000.
cd /d "%~dp0"
start "" /b /wait /affinity 400000 pf5_lanes.exe single 15 > single.txt 2>&1
echo exit=%ERRORLEVEL%
