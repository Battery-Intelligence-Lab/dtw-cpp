@echo off
rem vecrep.bat <src dir with dtwc\> <tag>: cl 19.50's /Qvec-report:2 for dtw_lanes.cpp with the MSVC tree's flags
rem minus /GL (a /GL object is code-generated at link time), plus /FAs; writes vec\<tag>.txt and vec\<tag>.asm.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
set CF=/nologo /TP /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /arch:AVX2 /openmp:experimental /we4062 /fp:precise /fp:contract /Gy /diagnostics:column
cl %CF% /Qvec-report:2 /FAs /Favec\%2.asm /Fovec\%2.obj /I%1\dtwc -c %1\dtwc\core\dtw_lanes.cpp > vec\%2.txt 2>&1
echo exit %errorlevel%
