@echo off
rem kernel_pinned.bat <L> <band>: p1_kernel_ab on logical CPU 22 (a P-core), 11 interleaved rounds.
cd /d C:\D\git\wt\tmp\P1
start "" /b /wait /affinity 400000 p1_kernel_ab.exe %1 %2 11
