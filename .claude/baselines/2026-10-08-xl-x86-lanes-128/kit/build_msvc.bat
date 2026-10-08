@echo off
rem build_msvc.bat base|cand: bin\msvc_<v>\xprobe.exe from xprobe.cpp and that version's
rem src\<v>\dtwc\core\{dtw_lanes,dtw_dispatch}.cpp, with cl 19.50's flags for those files in
rem build\cuda-verify-0928 (and the XL worktree's build-msvc), /GL included, linked as dtwc_cl.exe is (/LTCG).
rem The project's -D/-I flags are dropped: no header these files include tests a project macro. Also writes
rem asm\msvc_<v>_dtw_lanes.asm: the same flags without /GL (a /GL object holds no code to list) and /FAs.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
set V=%1
set O=bin\msvc_%V%
if not exist %O% mkdir %O%
if not exist asm mkdir asm
set CF=/nologo /TP /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /arch:AVX2 /openmp:experimental /we4062 /fp:precise /fp:contract /Gy /Isrc\%V%\dtwc
cl %CF% /GL /Fo%O%\dtw_lanes.obj -c src\%V%\dtwc\core\dtw_lanes.cpp || exit /b 1
cl %CF% /FAs /Faasm\msvc_%V%_dtw_lanes.asm /Fo%O%\dtw_lanes_nogl.obj -c src\%V%\dtwc\core\dtw_lanes.cpp || exit /b 1
cl %CF% /GL /Fo%O%\dtw_dispatch.obj -c src\%V%\dtwc\core\dtw_dispatch.cpp || exit /b 1
cl %CF% /GL /Fo%O%\xprobe.obj -c xprobe.cpp || exit /b 1
link /nologo /machine:x64 /INCREMENTAL:NO /subsystem:console /LTCG %O%\dtw_lanes.obj %O%\dtw_dispatch.obj %O%\xprobe.obj /OUT:%O%\xprobe.exe || exit /b 1
echo built %O%\xprobe.exe
