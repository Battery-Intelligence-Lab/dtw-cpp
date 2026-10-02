@echo off
rem Usage: build.bat <worktree> <out.exe> -- links clara_assign_bench against <worktree>\build-cuda's dtwc++.lib
rem (cl flags: build-cuda's compile of tests\unit\test_cuda_correctness.cpp, from compile_commands.json).
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL
set WT=%~1
cl /nologo /TP -DDTWC_HAS_CUDA -DDTWC_HAS_OPENMP -DDTWC_HAS_YAML /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /GL /arch:AVX2 /openmp:experimental /fp:precise /fp:contract /Gy -I%WT%\dtwc -I%WT%\tests -external:IC:\D\cpm-cache\cli11\ac32\include -external:I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include" -external:W0 %~dp0clara_assign_bench.cpp /Fo%~dpn2.obj /Fe%2 /link /LTCG /INCREMENTAL:NO %WT%\build-cuda\bin\dtwc++.lib %WT%\build-cuda\bin\highs.lib "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64\cudart.lib" kernel32.lib user32.lib advapi32.lib shell32.lib
echo BUILD_EXIT=%ERRORLEVEL%
