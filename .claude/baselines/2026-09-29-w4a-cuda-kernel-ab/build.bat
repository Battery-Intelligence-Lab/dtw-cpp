@echo off
rem W4a driver build. Usage: build.bat <repo> <out dir>   (run make_clone.sh <repo> <out dir> first)
rem nvcc flags: build-cuda's compile of dtwc/cuda/cuda_dtw.cu (compile_commands.json);
rem cl flags: the CUDA tests' (without Catch2 and /GL). smem_edge links the built dtwc++.lib.
set REPO=%~1
set OUT=%~2
set HERE=%~dp0
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"
set ARCH="--generate-code=arch=compute_89,code=[compute_89,sm_89]"
set NVFLAGS=-forward-unknown-to-host-compiler -DDTWC_HAS_CUDA -DDTWC_HAS_OPENMP -DDTWC_HAS_YAML -I%REPO%\dtwc -I%REPO%\tests -I%OUT% -allow-unsupported-compiler -Xcompiler="-O2 -Ob2" -DNDEBUG -std=c++20 %ARCH% -Xcompiler=-MD
cd /d %OUT%
%NVCC% %NVFLAGS% -x cu -c %HERE%ab.cu -o ab.obj || exit /b 1
cl /nologo /TP -DDTWC_HAS_OPENMP -I%REPO%\dtwc /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /arch:AVX2 /openmp:experimental /fp:precise /fp:contract /Gy -c %HERE%oracle.cpp /Fooracle.obj || exit /b 1
%NVCC% -allow-unsupported-compiler %ARCH% -Xcompiler=-MD ab.obj oracle.obj -o ab.exe || exit /b 1
%NVCC% %NVFLAGS% -x cu -c %HERE%smem_edge.cu -o smem_edge.obj || exit /b 1
%NVCC% -allow-unsupported-compiler %ARCH% -Xcompiler=-MD smem_edge.obj %REPO%\build-cuda\bin\dtwc++.lib -o smem_edge.exe || exit /b 1
echo BUILD_OK
