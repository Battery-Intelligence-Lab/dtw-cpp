@echo off
rem Usage: mem_probe_build.bat <out.exe> -- links the probe against build-cuda's current dtwc++.lib
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL
cd /d C:\Users\engs2321\AppData\Local\Temp\claude\c--D-git-dtw-cpp\79bd8991-b0e9-441b-bc55-95584dcfaf55\scratchpad\w13a
cl /nologo /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /GL /arch:AVX2 /openmp:experimental /fp:precise /fp:contract /Gy -DDTWC_HAS_CUDA -DDTWC_HAS_MMAP -DDTWC_HAS_OPENMP -DDTWC_HAS_YAML -DDTWC_VERSION_STRING=\"2.0.0rc1\" -IC:\D\git\wt\W13a\dtwc -external:I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include" -external:W0 mem_probe.cpp /Fe%1 /link /LTCG /INCREMENTAL:NO C:\D\git\wt\W13a\build-cuda\bin\dtwc++.lib "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64\cudart.lib" psapi.lib shlwapi.lib user32.lib advapi32.lib
echo PROBE_BUILD_EXIT=%ERRORLEVEL%
