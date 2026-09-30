@echo off
rem Usage: bench_fill.bat <exe> <out.json> <reps> -- the W13a band cases (BM_cuda_fill), pinned to the 8 P-cores
start "" /b /wait /affinity 0xC03C03 %1 "--benchmark_filter=^BM_cuda_fill/" --benchmark_repetitions=%3 --benchmark_report_aggregates_only=false --benchmark_out=%2 --benchmark_out_format=json
echo BENCH_EXIT=%ERRORLEVEL%
