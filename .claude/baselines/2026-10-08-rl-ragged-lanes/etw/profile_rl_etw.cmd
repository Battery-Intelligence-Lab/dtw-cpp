@echo off
rem DTWC++ unit RL (ragged lanes): hardware counters per run through Windows' own ETW counters (xperf), base against
rem the final candidate B3 (pb/RL at f8c3d23a). VTune's driver cannot sample under this machine's Hyper-V/VBS; Windows'
rem counters can (diag\, 2026-10-08). The first run (rl_etw\) measured B, before its L 8000 regression was fixed.
rem Run ONCE, from an Administrator terminal, on a quiet machine:   C:\D\git\wt\tmp\vtune\profile_rl_etw.cmd
rem About 3 minutes. Results: C:\D\git\wt\tmp\vtune\rl_etw2\<build>_<shape>.pmc.txt (counters of the probe process) and
rem <build>_<shape>.probe.txt (its fill time and cells). A second copy started while one runs stops at once (lock file);
rem a case whose .pmc.txt exists is skipped, so a rerun repeats only what is missing. Ends with rl_etw2\DONE.txt.
rem Each run is 2 fills (one warm-up, one timed); single-thread shapes pin themselves to CPU 12 (a P-core).
setlocal
set HERE=C:\D\git\wt\tmp\vtune
set OUT=%HERE%\rl_etw2
set XP="C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\xperf.exe"
set UV=C:\Users\engs2321\.local\bin\uv.exe
set PATH=C:\Program Files\LLVM\bin;C:\gurobi1301\win64\bin;%PATH%
set PMC=InstructionRetired,UnhaltedCoreCycles,LLCMisses,LLCReference,CacheMisses,BranchMispredictions
if not exist %OUT% mkdir %OUT%
if exist %OUT%\RUNNING.txt (
  echo Another run holds %OUT%\RUNNING.txt: not starting. If no run is going, delete that file and start again.
  exit /b 1
)
echo %DATE% %TIME% > %OUT%\RUNNING.txt
if exist %OUT%\DONE.txt del /q %OUT%\DONE.txt
echo start %DATE% %TIME% >> %OUT%\when.txt
typeperf "\Processor(_Total)\%% Processor Time" -sc 10 > %OUT%\load_before.txt

rem s1: f64 L1, N 1000, lengths 90-110, unbanded, 1 thread pinned
call :run clang_base s1 f64 L1 1000 90 110 -1 1
call :run clang_B3   s1 f64 L1 1000 90 110 -1 1
rem s2: f64 L1, N 150, lengths 100-1000, unbanded, 1 thread pinned
call :run clang_base s2 f64 L1 150 100 1000 -1 1
call :run clang_B3   s2 f64 L1 150 100 1000 -1 1
rem s5: f64 L1, N 24, every length 8000, unbanded, 1 thread pinned (B ran 4-5 %% slower here; B3 is to match base)
call :run clang_base s5 f64 L1 24 8000 8000 -1 1
call :run clang_B3   s5 f64 L1 24 8000 8000 -1 1
rem s6: the same under cl at N 8 (the kit's shape: cl's lanes are scalar)
call :run msvc_base  s6 f64 L1 8 8000 8000 -1 1
call :run msvc_B3    s6 f64 L1 8 8000 8000 -1 1
rem s3: f64 L1, N 8000, lengths 40-60, unbanded, 24 threads (a 256 MB matrix: write locality)
call :run clang_base s3 f64 L1 8000 40 60 -1 0
call :run clang_B3   s3 f64 L1 8000 40 60 -1 0

typeperf "\Processor(_Total)\%% Processor Time" -sc 10 > %OUT%\load_after.txt
echo end %DATE% %TIME% >> %OUT%\when.txt
echo done > %OUT%\DONE.txt
del /q %OUT%\RUNNING.txt
echo All runs finished: results in %OUT%
goto :eof

:run
rem %1 build, %2 shape, %3-%9 the probe's T D N shortest longest band threads (threads 1 = pinned)
set B=%1
set TAG=%2
if exist %OUT%\%B%_%TAG%.pmc.txt exit /b
if "%9"=="1" (set OMP_NUM_THREADS=1) else (set OMP_NUM_THREADS=)
echo [%TIME%] %B% %TAG%
%XP% -on PROC_THREAD+LOADER+CSWITCH -f %OUT%\k_%B%_%TAG%.etl -Pmc %PMC% CSWITCH > %OUT%\%B%_%TAG%.xperf.txt 2>&1
%HERE%\bin\%B%\rprobe.exe fill %3 %4 %5 %6 %7 %8 %9 1 12 > %OUT%\%B%_%TAG%.probe.txt 2>&1
%XP% -stop >> %OUT%\%B%_%TAG%.xperf.txt 2>&1
%XP% -merge %OUT%\k_%B%_%TAG%.etl %OUT%\t_%B%_%TAG%.etl >> %OUT%\%B%_%TAG%.xperf.txt 2>&1
%XP% -i %OUT%\t_%B%_%TAG%.etl -o %OUT%\t_%B%_%TAG%.txt -a dumper >> %OUT%\%B%_%TAG%.xperf.txt 2>&1
%UV% run --no-project python %HERE%\etw_pmc.py %OUT%\t_%B%_%TAG%.txt rprobe > %OUT%\%B%_%TAG%.pmc.tmp 2>&1
findstr /c:"rprobe.exe" %OUT%\%B%_%TAG%.pmc.tmp >nul && move /y %OUT%\%B%_%TAG%.pmc.tmp %OUT%\%B%_%TAG%.pmc.txt >nul
del /q %OUT%\k_%B%_%TAG%.etl %OUT%\t_%B%_%TAG%.etl %OUT%\t_%B%_%TAG%.txt
exit /b
