# UCR Archive Benchmark Results

Full DTW k-medoids clustering (Lloyd iteration) on all 128 UCR time series datasets.
Distance matrices computed via brute-force DTW (L1 norm, no banding).
All results from Oxford ARC HTC cluster, 2026-04-09.

## Hardware

| Platform | Processor | Cores/Threads | Memory |
|----------|-----------|---------------|--------|
| CPU (baseline) | Intel Xeon Platinum 8268 @ 2.90 GHz | 16 | 64 GB |
| CPU (Genoa) | AMD EPYC 9645 @ 2.30 GHz | 168 | 672 GB |
| GPU (L40S) | NVIDIA L40S (Ada Lovelace) | 18,176 CUDA | 48 GB GDDR6X |
| GPU (H100) | NVIDIA H100 NVL (Hopper) | 16,896 CUDA | 94 GB HBM3 |
| Apple M2 Max (CPU) | Apple M2 Max (8P + 4E cores) | 12 | 64 GB unified |
| Apple M2 Max (GPU, Metal) | Apple M2 Max 38-core GPU | 38 GPU cores | shares unified |

### Apple Silicon spot-check (2026-04-12, M2 Max)

Not yet a full UCR sweep; measurements come from `benchmarks/bench_dtw_baseline` (CPU thread scaling) and `benchmarks/bench_metal_dtw` (Metal vs CPU at matching sizes).

Thread scaling on `fillDistanceMatrix/100/1000/-1` (4950 pairs × 1000² cells, unbanded):

| Threads | Time (ms) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 14 618 | 1.0× | 100% |
| 8 (P-cores) | 2 124 | 6.88× | 86% |
| 12 (P + E) | 1 599 | 9.14× | 76% |

`OMP_PROC_BIND` / `OMP_PLACES` pinning has no measurable effect on Darwin + Homebrew libomp (pinning sweep at T=12 was within 0.3%).

Metal GPU path on the same workloads (unbanded, FP32):

| Workload | CPU 12-thread (ms) | Metal (ms) | Speedup |
|---|---|---|---|
| 50 × 500 | 106 | 6.9 | **15.4×** |
| 100 × 500 | 404 | 26.2 | **15.4×** |
| 100 × 1000 | 1648 | 139 | **11.9×** |
| 200 × 500 | 1626 | 103 | **15.8×** |
| 50 × 2500 | — | 151 | — |
| **75 × 2500** | **5914** | **342** | **17.3×** |

Long series (threadgroup cap at ~2730 → global-memory kernel kicks in):

| Workload | CPU 12-thread | Metal | Speedup |
|---|---|---|---|
| 10 × 10000 | 3.06 s | 108 ms | **28.3×** |
| 30 × 10000 | 16.06 s | 929 ms | **17.3×** |
| **75 × 10000** | **92.5 s** | **5.80 s** | **15.9×** |

Banded (Sakoe-Chiba) on the hardest workload (75 × 10000):

| Variant | CPU 12-thread | Metal |
|---|---|---|
| unbanded | 92.5 s | 5.80 s |
| band = L/10 = 1000 | 17.5 s | 2.01 s |
| band = 100 (tight) | 1.75 s | 1.40 s |

Achieved throughput: 35–52 × 10⁹ DTW cells/sec; ~180–320 GFLOPS (≈1.3–2.4% of 13.6 TFLOPS FP32 peak). Low FLOP fraction is expected for DTW's memory-bandwidth-bound DP recurrence. Initial anti-diagonal wavefront kernels; register-tiling and warp-shuffle variants are follow-on work.

Wrapper overhead vs native C++ (measured on 8 workloads, 50×500 through 30×10000):

| Language | vs direct Metal | vs `Problem::fillDistanceMatrix` | Details |
|---|---|---|---|
| Python (nanobind) | **−0.5% to +0.9%** | **<1%** | `nb::gil_scoped_release` keeps GIL off during GPU dispatch |
| MATLAB (MEX, R2024b) | +8.5% to +17% | **<1%** | The 8.5–17% number was a like-for-unlike comparison: MATLAB uses `Problem::fillDistanceMatrix` (adds ~15% Problem-wrapper work), whereas the C++ bench timed `compute_distance_matrix_metal` directly. Comparing same path, MATLAB is within 1% of Python. |

MATLAB 10 000-length verification: `10×10000` → 103.81 ms, `30×10000` → 890.75 ms (vs C++ 108 ms / 929 ms). Both bindings meet the ≤10% wrapper-overhead target.

## Summary

| Platform | Total Time | Speedup |
|----------|-----------|---------|
| Xeon 8268 (16 cores) | 43.5 min | 1.0x |
| EPYC 9645 (168 cores) | 3.7 min | 11.8x |
| NVIDIA L40S | 3.9 min | 11.1x |
| NVIDIA H100 NVL | 3.1 min | 14.2x |

All 128 datasets passed on every platform. 0 failures, 0 timeouts.

## Top 20 Datasets by Computation Time

| Dataset | N | Length | k | Xeon 16c (s) | EPYC 168c (s) | L40S (s) | H100 (s) | H100 Speedup |
|---------|--:|-------:|--:|-------------:|--------------:|---------:|---------:|-------------:|
| HandOutlines | 1000 | 2709 | 2 | 769.94 | 45.81 | 39.40 | 15.82 | 48.7x |
| FordB | 3636 | 500 | 2 | 350.14 | 24.77 | 26.07 | 18.95 | 18.5x |
| FordA | 3601 | 500 | 2 | 344.16 | 23.09 | 25.50 | 17.73 | 19.4x |
| NonInvasiveFetalECGThorax1 | 1800 | 750 | 42 | 191.51 | 12.78 | 12.73 | 7.79 | 24.6x |
| NonInvasiveFetalECGThorax2 | 1800 | 750 | 42 | 191.37 | 13.56 | 12.74 | 7.76 | 24.7x |
| ElectricDevices | 8926 | 96 | 7 | 129.28 | 27.09 | 32.39 | 31.00 | 4.2x |
| StarLightCurves | 1000 | 1024 | 3 | 110.01 | 8.46 | 7.26 | 4.64 | 23.7x |
| EthanolLevel | 504 | 1751 | 4 | 81.71 | 6.32 | 4.59 | 2.49 | 32.9x |
| UWaveGestureLibraryAll | 896 | 945 | 8 | 75.26 | 6.00 | 5.07 | 3.00 | 25.1x |
| SemgHandMovementCh2 | 450 | 1500 | 6 | 47.86 | 4.08 | 2.83 | 1.78 | 26.9x |
| SemgHandSubjectCh2 | 450 | 1500 | 5 | 47.80 | 3.88 | 2.80 | 1.67 | 28.6x |
| Crop | 7200 | 46 | 24 | 30.14 | 10.68 | 13.92 | 13.52 | 2.2x |
| MixedShapesRegularTrain | 500 | 1024 | 5 | 27.58 | 2.36 | 2.01 | 1.37 | 20.2x |
| EOGVerticalSignal | 362 | 1250 | 12 | 21.55 | 1.95 | 1.47 | 1.02 | 21.2x |
| EOGHorizontalSignal | 362 | 1250 | 12 | 21.55 | 1.92 | 1.41 | 1.00 | 21.5x |
| SemgHandGenderCh2 | 300 | 1500 | 2 | 21.33 | 2.20 | 1.39 | 0.99 | 21.5x |
| ShapesAll | 600 | 512 | 60 | 10.09 | 1.03 | 1.00 | 0.90 | 11.2x |
| UWaveGestureLibraryZ | 896 | 315 | 8 | 8.83 | 1.04 | 1.12 | 1.26 | 7.0x |
| UWaveGestureLibraryX | 896 | 315 | 8 | 8.82 | 1.03 | 1.17 | 1.08 | 8.1x |
| UWaveGestureLibraryY | 896 | 315 | 8 | 8.82 | 1.10 | 1.27 | 1.31 | 6.7x |

## Full Results

| Dataset | N | Length | k | Xeon 16c (s) | EPYC 168c (s) | L40S (s) | H100 (s) |
|---------|--:|-------:|--:|-------------:|--------------:|---------:|---------:|
| ACSF1 | 100 | 1460 | 10 | 2.31 | 0.48 | 0.46 | 0.43 |
| Adiac | 390 | 176 | 37 | 0.87 | 0.51 | 0.82 | 1.33 |
| AllGestureWiimoteX | 300 | Vary | 10 | 0.25 | 0.11 | 0.34 | 0.40 |
| AllGestureWiimoteY | 300 | Vary | 10 | 0.26 | 0.13 | 0.28 | 0.41 |
| AllGestureWiimoteZ | 300 | Vary | 10 | 0.25 | 0.12 | 0.29 | 0.41 |
| ArrowHead | 36 | 251 | 3 | 0.06 | 0.06 | 0.23 | 0.34 |
| BME | 30 | 128 | 3 | 0.05 | 0.06 | 0.22 | 0.34 |
| Beef | 30 | 470 | 5 | 0.07 | 0.06 | 0.29 | 0.37 |
| BeetleFly | 20 | 512 | 2 | 0.07 | 0.06 | 0.23 | 0.37 |
| BirdChicken | 20 | 512 | 2 | 0.05 | 0.06 | 0.21 | 0.37 |
| CBF | 30 | 128 | 3 | 0.05 | 0.05 | 0.23 | 0.37 |
| Car | 60 | 577 | 4 | 0.19 | 0.11 | 0.24 | 0.38 |
| Chinatown | 20 | 24 | 2 | 0.04 | 0.05 | 0.22 | 0.35 |
| ChlorineConcentration | 467 | 166 | 3 | 0.85 | 0.18 | 0.35 | 0.48 |
| CinCECGTorso | 40 | 1639 | 4 | 0.56 | 0.26 | 0.26 | 0.53 |
| Coffee | 28 | 286 | 2 | 0.05 | 0.05 | 0.27 | 0.38 |
| Computers | 250 | 720 | 2 | 3.56 | 0.48 | 0.47 | 0.51 |
| CricketX | 390 | 300 | 12 | 1.62 | 0.25 | 0.39 | 0.52 |
| CricketY | 390 | 300 | 12 | 1.63 | 0.24 | 0.39 | 0.50 |
| CricketZ | 390 | 300 | 12 | 1.62 | 0.25 | 0.47 | 0.51 |
| Crop | 7200 | 46 | 24 | 30.14 | 10.68 | 13.92 | 13.52 |
| DiatomSizeReduction | 16 | 345 | 4 | 0.05 | 0.05 | 0.30 | 0.37 |
| DistalPhalanxOutlineAgeGroup | 400 | 80 | 3 | 0.24 | 0.11 | 0.27 | 0.41 |
| DistalPhalanxOutlineCorrect | 600 | 80 | 2 | 0.49 | 0.16 | 0.36 | 0.49 |
| DistalPhalanxTW | 400 | 80 | 6 | 0.25 | 0.11 | 0.32 | 0.40 |
| DodgerLoopGame | 20 | 288 | 2 | 0.04 | 0.10 | 0.29 | 0.45 |
| DodgerLoopWeekend | 20 | 288 | 2 | 0.05 | 0.05 | 0.53 | 0.36 |
| ECG200 | 100 | 96 | 2 | 0.07 | 0.06 | 0.23 | 0.36 |
| ECG5000 | 500 | 140 | 5 | 0.75 | 0.18 | 0.65 | 0.55 |
| ECGFiveDays | 23 | 136 | 2 | 0.04 | 0.05 | 0.34 | 0.36 |
| EOGHorizontalSignal | 362 | 1250 | 12 | 21.55 | 1.92 | 1.41 | 1.00 |
| EOGVerticalSignal | 362 | 1250 | 12 | 21.55 | 1.95 | 1.47 | 1.02 |
| Earthquakes | 322 | 512 | 2 | 3.02 | 0.36 | 0.54 | 0.54 |
| ElectricDevices | 8926 | 96 | 7 | 129.28 | 27.09 | 32.39 | 31.00 |
| EthanolLevel | 504 | 1751 | 4 | 81.71 | 6.32 | 4.59 | 2.49 |
| FaceAll | 560 | 131 | 14 | 0.85 | 0.22 | 0.42 | 0.58 |
| FaceFour | 24 | 350 | 4 | 0.06 | 0.05 | 0.23 | 0.36 |
| FacesUCR | 200 | 131 | 14 | 0.16 | 0.08 | 0.24 | 0.39 |
| FiftyWords | 450 | 270 | 50 | 1.75 | 0.25 | 0.49 | 0.53 |
| Fish | 175 | 463 | 7 | 0.78 | 0.17 | 0.35 | 0.40 |
| FordA | 3601 | 500 | 2 | 344.16 | 23.09 | 25.50 | 17.73 |
| FordB | 3636 | 500 | 2 | 350.14 | 24.77 | 26.07 | 18.95 |
| FreezerRegularTrain | 150 | 301 | 2 | 0.29 | 0.10 | 0.32 | 0.39 |
| FreezerSmallTrain | 28 | 301 | 2 | 0.06 | 0.05 | 0.25 | 0.35 |
| Fungi | 18 | 201 | 18 | 0.05 | 0.05 | 0.28 | 0.37 |
| GestureMidAirD1 | 208 | Vary | 26 | 0.20 | 0.10 | 0.25 | 0.40 |
| GestureMidAirD2 | 208 | Vary | 26 | 0.19 | 0.09 | 0.25 | 0.40 |
| GestureMidAirD3 | 208 | Vary | 26 | 0.21 | 0.10 | 0.37 | 0.38 |
| GesturePebbleZ1 | 132 | Vary | 6 | 0.17 | 0.09 | 0.33 | 0.36 |
| GesturePebbleZ2 | 146 | Vary | 6 | 0.18 | 0.09 | 0.27 | 0.37 |
| GunPoint | 50 | 150 | 2 | 0.06 | 0.06 | 0.23 | 0.43 |
| GunPointAgeSpan | 135 | 150 | 2 | 0.11 | 0.07 | 0.30 | 0.36 |
| GunPointMaleVersusFemale | 135 | 150 | 2 | 0.10 | 0.07 | 0.23 | 0.36 |
| GunPointOldVersusYoung | 136 | 150 | 2 | 0.10 | 0.07 | 0.23 | 0.37 |
| Ham | 109 | 431 | 2 | 0.30 | 0.11 | 0.49 | 0.38 |
| HandOutlines | 1000 | 2709 | 2 | 769.94 | 45.81 | 39.40 | 15.82 |
| Haptics | 155 | 1092 | 5 | 3.10 | 0.46 | 0.44 | 0.47 |
| Herring | 64 | 512 | 2 | 0.18 | 0.09 | 0.28 | 0.37 |
| HouseTwenty | 40 | 2000 | 2 | 0.78 | 0.35 | 0.33 | 0.40 |
| InlineSkate | 100 | 1882 | 7 | 3.80 | 0.74 | 0.50 | 0.50 |
| InsectEPGRegularTrain | 62 | 601 | 3 | 0.21 | 0.08 | 0.28 | 0.36 |
| InsectEPGSmallTrain | 17 | 601 | 3 | 0.08 | 0.06 | 0.23 | 0.36 |
| InsectWingbeatSound | 220 | 256 | 11 | 0.43 | 0.14 | 0.26 | 0.40 |
| ItalyPowerDemand | 67 | 24 | 2 | 0.05 | 0.06 | 0.24 | 0.34 |
| LargeKitchenAppliances | 375 | 720 | 3 | 7.76 | 0.84 | 0.76 | 0.69 |
| Lightning2 | 60 | 637 | 2 | 0.22 | 0.09 | 0.31 | 0.37 |
| Lightning7 | 70 | 319 | 7 | 0.11 | 0.06 | 0.25 | 0.36 |
| Mallat | 55 | 1024 | 8 | 0.41 | 0.16 | 0.26 | 0.40 |
| Meat | 60 | 448 | 3 | 0.14 | 0.07 | 0.24 | 0.36 |
| MedicalImages | 381 | 99 | 10 | 0.31 | 0.11 | 0.32 | 0.41 |
| MiddlePhalanxOutlineAgeGroup | 400 | 80 | 3 | 0.23 | 0.10 | 0.27 | 0.47 |
| MiddlePhalanxOutlineCorrect | 600 | 80 | 2 | 0.46 | 0.15 | 0.34 | 0.48 |
| MiddlePhalanxTW | 399 | 80 | 6 | 0.23 | 0.10 | 0.29 | 0.41 |
| MixedShapesRegularTrain | 500 | 1024 | 5 | 27.58 | 2.36 | 2.01 | 1.37 |
| MixedShapesSmallTrain | 100 | 1024 | 5 | 1.18 | 0.28 | 0.31 | 0.40 |
| MoteStrain | 20 | 84 | 2 | 0.04 | 0.05 | 0.26 | 0.36 |
| NonInvasiveFetalECGThorax1 | 1800 | 750 | 42 | 191.51 | 12.78 | 12.73 | 7.79 |
| NonInvasiveFetalECGThorax2 | 1800 | 750 | 42 | 191.37 | 13.56 | 12.74 | 7.76 |
| OSULeaf | 200 | 427 | 6 | 0.85 | 0.17 | 0.33 | 0.41 |
| OliveOil | 30 | 570 | 4 | 0.10 | 0.07 | 0.30 | 0.36 |
| PLAID | 537 | Vary | 11 | 3.35 | 0.45 | 0.56 | 0.68 |
| PhalangesOutlinesCorrect | 1800 | 80 | 2 | 3.71 | 0.97 | 1.29 | 1.38 |
| Phoneme | 214 | 1024 | 39 | 5.15 | 0.77 | 0.64 | 0.59 |
| PickupGestureWiimoteZ | 50 | Vary | 10 | 0.06 | 0.06 | 0.23 | 0.35 |
| PigAirwayPressure | 104 | 2000 | 52 | 4.65 | 0.88 | 0.50 | 0.58 |
| PigArtPressure | 104 | 2000 | 52 | 4.65 | 0.90 | 0.51 | 0.63 |
| PigCVP | 104 | 2000 | 52 | 4.65 | 0.89 | 0.48 | 0.48 |
| Plane | 105 | 144 | 7 | 0.09 | 0.07 | 0.23 | 0.36 |
| PowerCons | 180 | 144 | 2 | 0.14 | 0.08 | 0.25 | 0.38 |
| ProximalPhalanxOutlineAgeGroup | 400 | 80 | 3 | 0.23 | 0.10 | 0.40 | 0.43 |
| ProximalPhalanxOutlineCorrect | 600 | 80 | 2 | 0.44 | 0.16 | 0.65 | 0.53 |
| ProximalPhalanxTW | 400 | 80 | 6 | 0.22 | 0.11 | 0.29 | 0.43 |
| RefrigerationDevices | 375 | 720 | 3 | 7.75 | 0.98 | 0.76 | 0.84 |
| Rock | 20 | 2844 | 4 | 0.58 | 0.34 | 0.33 | 0.36 |
| ScreenType | 375 | 720 | 3 | 7.75 | 0.87 | 0.76 | 0.78 |
| SemgHandGenderCh2 | 300 | 1500 | 2 | 21.33 | 2.20 | 1.39 | 0.99 |
| SemgHandMovementCh2 | 450 | 1500 | 6 | 47.86 | 4.08 | 2.83 | 1.78 |
| SemgHandSubjectCh2 | 450 | 1500 | 5 | 47.80 | 3.88 | 2.80 | 1.67 |
| ShakeGestureWiimoteZ | 50 | Vary | 10 | 0.06 | 0.09 | 0.32 | 0.37 |
| ShapeletSim | 20 | 500 | 2 | 0.07 | 0.06 | 0.30 | 0.38 |
| ShapesAll | 600 | 512 | 60 | 10.09 | 1.03 | 1.00 | 0.90 |
| SmallKitchenAppliances | 375 | 720 | 3 | 7.75 | 0.81 | 0.84 | 0.70 |
| SmoothSubspace | 150 | 15 | 3 | 0.05 | 0.05 | 0.23 | 0.35 |
| SonyAIBORobotSurface1 | 20 | 70 | 2 | 0.04 | 0.05 | 0.22 | 0.33 |
| SonyAIBORobotSurface2 | 27 | 65 | 2 | 0.04 | 0.05 | 0.22 | 0.63 |
| StarLightCurves | 1000 | 1024 | 3 | 110.01 | 8.46 | 7.26 | 4.64 |
| Strawberry | 613 | 235 | 2 | 2.43 | 0.38 | 0.50 | 0.56 |
| SwedishLeaf | 500 | 128 | 15 | 0.64 | 0.18 | 0.41 | 0.48 |
| Symbols | 25 | 398 | 6 | 0.07 | 0.06 | 0.22 | 0.37 |
| SyntheticControl | 300 | 60 | 6 | 0.16 | 0.09 | 0.30 | 0.42 |
| ToeSegmentation1 | 40 | 277 | 2 | 0.07 | 0.05 | 0.24 | 0.35 |
| ToeSegmentation2 | 36 | 343 | 2 | 0.07 | 0.06 | 0.23 | 0.36 |
| Trace | 100 | 275 | 4 | 0.14 | 0.08 | 0.36 | 0.37 |
| TwoLeadECG | 23 | 82 | 2 | 0.03 | 0.06 | 0.30 | 0.35 |
| TwoPatterns | 1000 | 128 | 4 | 2.41 | 0.51 | 0.67 | 0.75 |
| UMD | 36 | 150 | 3 | 0.05 | 0.05 | 0.21 | 0.34 |
| UWaveGestureLibraryAll | 896 | 945 | 8 | 75.26 | 6.00 | 5.07 | 3.00 |
| UWaveGestureLibraryX | 896 | 315 | 8 | 8.82 | 1.03 | 1.17 | 1.08 |
| UWaveGestureLibraryY | 896 | 315 | 8 | 8.82 | 1.10 | 1.27 | 1.31 |
| UWaveGestureLibraryZ | 896 | 315 | 8 | 8.83 | 1.04 | 1.12 | 1.26 |
| Wafer | 1000 | 152 | 2 | 3.09 | 0.54 | 0.77 | 0.79 |
| Wine | 57 | 234 | 2 | 0.07 | 0.06 | 0.24 | 0.36 |
| WordSynonyms | 267 | 270 | 25 | 0.65 | 0.15 | 0.40 | 0.50 |
| Worms | 181 | 900 | 5 | 2.88 | 0.44 | 0.46 | 0.49 |
| WormsTwoClass | 181 | 900 | 2 | 2.88 | 0.44 | 0.43 | 0.50 |
| Yoga | 300 | 426 | 2 | 1.83 | 0.28 | 0.39 | 0.50 |

## Notes

- Method: Lloyd k-medoids iteration (not MIP, not FastPAM)
- DTW: L1 norm, no Sakoe-Chiba band, full warping path
- Data precision: float64
- GPU: CUDA distance matrix computation, CPU-side clustering
- OpenMP dynamic scheduling with runtime-tuned chunk sizes
- Distance matrices saved for CPU/GPU numerical verification
