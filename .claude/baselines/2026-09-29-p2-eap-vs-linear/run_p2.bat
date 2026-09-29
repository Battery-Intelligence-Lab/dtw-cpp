@echo off
rem P2 timing, pinned to logical CPU 22 (a P-core): affinity mask 0x400000. 7 interleaved rounds.
cd /d "%~dp0"
start "" /b /wait /affinity 400000 p2_eap_vs_linear.exe C:\D\git\dtw-cpp\data\benchmark\UCRArchive_2018 p2_pairs.csv 7 ECG5000 StarLightCurves Mallat InlineSkate HandOutlines CinCECGTorso Rock:z > p2_run.txt 2>&1
echo exit=%ERRORLEVEL%
start "" /b /wait /affinity 400000 p2_eap_vs_linear.exe C:\D\git\dtw-cpp\data\benchmark\UCRArchive_2018 p2_pairs_rock_raw.csv 7 Rock > p2_run_rock_raw.txt 2>&1
echo exit=%ERRORLEVEL%
