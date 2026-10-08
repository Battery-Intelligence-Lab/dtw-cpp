@echo off
rem Relinks bin\msvc_<v>\xprobe.exe's objects with /MAP into bin\msvc_<v>\map\ (same /LTCG link otherwise), so the
rem lanes functions of the linked code have addresses. usage: msvc_map.bat base|cand
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0.."
set O=bin\msvc_%1
if not exist %O%\map mkdir %O%\map
link /nologo /machine:x64 /INCREMENTAL:NO /subsystem:console /LTCG %O%\dtw_lanes.obj %O%\dtw_dispatch.obj %O%\xprobe.obj /OUT:%O%\map\xprobe_map.exe /MAP:%O%\map\xprobe.map || exit /b 1
