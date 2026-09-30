@echo off
rem Usage: build.bat <worktree> <out.exe> -- links long_fill against <worktree>\build-cuda's dtwc++.lib
rem (cl flags: build-cuda's C++ compile of the library, from compile_commands.json).
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL
set WT=%~1
cl /nologo /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /GL /arch:AVX2 /openmp:experimental /fp:precise /fp:contract /Gy -DDTWC_HAS_CUDA -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_HAS_YAML -I%WT%\dtwc -I%WT%\tests -external:I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include" -external:W0 %~dp0long_fill.cpp /Fo%~dpn2.obj /Fe%2 /link /LTCG /INCREMENTAL:NO %WT%\build-cuda\bin\dtwc++.lib "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64\cudart.lib" psapi.lib shlwapi.lib user32.lib advapi32.lib
echo BUILD_EXIT=%ERRORLEVEL%
