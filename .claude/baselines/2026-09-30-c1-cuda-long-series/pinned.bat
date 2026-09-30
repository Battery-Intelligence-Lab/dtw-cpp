@echo off
rem Usage: pinned.bat <exe> <args...> -- runs <exe> pinned to the 8 P-cores (logical CPUs 0,1,10-13,22,23)
set TMP=C:\D\git\wt\tmp\C1\cuda
set TEMP=C:\D\git\wt\tmp\C1\cuda
set PATH=C:\D\git\wt\C1\build\bin;C:\Program Files\LLVM\bin;C:\gurobi1301\win64\bin;%PATH%
start "" /b /wait /affinity 0xC03C03 %*
echo PINNED_EXIT=%ERRORLEVEL%
