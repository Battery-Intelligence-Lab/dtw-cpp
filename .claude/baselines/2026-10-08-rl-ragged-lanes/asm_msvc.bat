@echo off
rem asm_msvc.bat <v> <dtwc dir>: asm\msvc_<v>_dtw_lanes.asm, dtw_lanes.cpp of <dtwc dir> with cl 19.50's flags for that
rem file in the RL MSVC tree (compile_commands.json) without /GL (a /GL object holds no code to list), /FAs.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if not exist asm mkdir asm
if not exist asm\obj mkdir asm\obj
cl /nologo /TP -DDTWC_ENABLE_GUROBI -DDTWC_ENABLE_HIGHS -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_VERSION_STRING=\"2.0.0rc1\" -I%2\. -I%2\extern -I%2\mip\. /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /arch:AVX2 /openmp:experimental /we4062 /fp:precise /fp:contract /Gy /FAs /Faasm\msvc_%1_dtw_lanes.asm /Foasm\obj\msvc_%1_dtw_lanes.obj -c %2\core\dtw_lanes.cpp || exit /b 1
echo asm\msvc_%1_dtw_lanes.asm
