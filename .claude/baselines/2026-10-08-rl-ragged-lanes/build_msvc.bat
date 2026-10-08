@echo off
rem build_msvc.bat base|A|B|B2: bin\msvc_<v>\rprobe.exe from rprobe.cpp against src\<v>\dtwc (headers) and
rem lib\msvc_<v>\dtwc_core.lib (that version's library, saved from the RL cl 19.50 tree C:\D\git\wt\RL\build-msvc,
rem configured as MV's build-msvc). Compiled with dtwc_cl.cpp's flags from the tree's compile_commands.json (/GL) and
rem linked as dtwc_cl.exe is (/LTCG; HiGHS and Gurobi from the tree and the system: the unit changes neither).
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
set V=%1
set T=C:\D\git\wt\RL\build-msvc
set O=bin\msvc_%V%
if not exist %O% mkdir %O%
cl /nologo /TP -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -Isrc\%V%\dtwc\. /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /GL /arch:AVX2 /openmp:experimental /fp:precise /fp:contract /Gy /Fo%O%\rprobe.obj -c rprobe.cpp || exit /b 1
link /nologo /machine:x64 /INCREMENTAL:NO /subsystem:console /LTCG %O%\rprobe.obj lib\msvc_%V%\dtwc_core.lib %T%\GurobiCXX.lib C:\gurobi1301\win64\lib\gurobi130.lib %T%\bin\highs.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib /OUT:%O%\rprobe.exe || exit /b 1
echo built %O%\rprobe.exe
