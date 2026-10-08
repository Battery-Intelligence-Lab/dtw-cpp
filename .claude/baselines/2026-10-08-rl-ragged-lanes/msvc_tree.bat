@echo off
rem msvc_tree.bat configure|build|ctest [args]: the RL worktree's cl 19.50 tree C:\D\git\wt\RL\build-msvc, under
rem vcvars64, as MV's build-msvc (Ninja, cl, Release, tests, Gurobi, YAML OFF), deps from the CPM cache.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CPM_SOURCE_CACHE=
set CMAKE_GENERATOR=
set TMP=C:\D\git\wt\tmp\rl
set TEMP=C:\D\git\wt\tmp\rl
cd /d C:\D\git\wt\RL
if "%1"=="configure" goto configure
if "%1"=="build" goto build
if "%1"=="ctest" goto ctest
echo usage: msvc_tree.bat configure^|build^|ctest
exit /b 2
:configure
cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_MAKE_PROGRAM=C:/D/git/dtw-cpp/.venv/Scripts/ninja.exe -DCPM_SOURCE_CACHE=C:/D/cpm-cache -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_YAML=OFF -DDTWC_ENABLE_GUROBI=ON -DGUROBI_HOME=C:/gurobi1301/win64 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
exit /b %errorlevel%
:build
cmake --build build-msvc -j 8 %2 %3 %4 %5
exit /b %errorlevel%
:ctest
ctest --test-dir build-msvc -C Release -j1 --output-on-failure %2 %3 %4 %5
exit /b %errorlevel%
