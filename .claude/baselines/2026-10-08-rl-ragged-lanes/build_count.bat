@echo off
rem build_count.bat <v>: bin\msvc_<v>_count\rprobe.exe, the lane-call counting build (kit only, never shipped code):
rem the shipped src\<v>\dtwc\core\dtw_lanes.cpp compiled with its MSVC-tree flags (compile_commands.json) plus
rem /FI count\count_lanes.hpp, which counts each dtw_kernel_lanes call and forwards it; rprobe.cpp with -DRL_COUNT;
rem linked as build_msvc.bat links, the counting object before lib\msvc_<v>\dtwc_core.lib, whose own dtw_lanes
rem object is then not pulled in. So the fill is that version's own Problem code, counted at its lane function.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
set V=%1
set T=C:\D\git\wt\RL\build-msvc
set O=bin\msvc_%V%_count
if not exist %O% mkdir %O%
cl /nologo /TP -DDTWC_ENABLE_GUROBI -DDTWC_ENABLE_HIGHS -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -I. -Isrc\%V%\dtwc\. -Isrc\%V%\dtwc\extern -Isrc\%V%\dtwc\mip\. /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /GL /arch:AVX2 /openmp:experimental /we4062 /fp:precise /fp:contract /Gy /FIcount\count_lanes.hpp /Fo%O%\dtw_lanes.obj -c src\%V%\dtwc\core\dtw_lanes.cpp || exit /b 1
cl /nologo /TP -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -DRL_COUNT -I. -Isrc\%V%\dtwc\. /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /GL /arch:AVX2 /openmp:experimental /fp:precise /fp:contract /Gy /Fo%O%\rprobe.obj -c rprobe.cpp || exit /b 1
link /nologo /machine:x64 /INCREMENTAL:NO /subsystem:console /LTCG %O%\rprobe.obj %O%\dtw_lanes.obj lib\msvc_%V%\dtwc_core.lib %T%\GurobiCXX.lib C:\gurobi1301\win64\lib\gurobi130.lib %T%\bin\highs.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib /OUT:%O%\rprobe.exe || exit /b 1
echo built %O%\rprobe.exe
