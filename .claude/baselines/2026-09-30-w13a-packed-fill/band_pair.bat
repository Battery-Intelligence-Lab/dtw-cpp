@echo off
rem Usage: band_pair.bat <tag> -- base then head band runs, back to back, 5 pinned repetitions each
set W=C:\Users\engs2321\AppData\Local\Temp\claude\c--D-git-dtw-cpp\79bd8991-b0e9-441b-bc55-95584dcfaf55\scratchpad\w13a
echo BASE_START %TIME%
call %W%\bench_fill.bat %W%\base\bench_cuda_dtw_base.exe %W%\base\band_base_%1.json 5 > %W%\base\band_base_%1.txt 2>&1
echo HEAD_START %TIME%
call %W%\bench_fill.bat %W%\head\bench_cuda_dtw_head.exe %W%\head\band_head_%1.json 5 > %W%\head\band_head_%1.txt 2>&1
echo END %TIME%
