Medians (range) over 3 round(s), Gcell/s; ratio = clang-cl / MSVC of the medians (above 1: clang-cl faster).

| shape | threads | MSVC Gcell/s | clang-cl Gcell/s | clang-cl/MSVC | shortest timed s | result |
| --- | --- | --- | --- | --- | --- | --- |
| (a) N 2000, L 100, unbanded: lanes fill | 1 | 1.341 (1.332-1.345) | 11.019 (10.883-11.074) | **8.22** | 1.81 | same |
| (a) N 2000, L 100, unbanded: lanes fill | 24 | 31.088 (30.761-31.441) | 123.256 (122.845-127.712) | **3.96** | 0.63 | same |
| (b) N 500, L 1000, band 100: lanes fill | 1 | 1.324 (1.312-1.453) | 10.780 (10.747-10.787) | **8.14** | 2.21 | same |
| (b) N 500, L 1000, band 100: lanes fill | 24 | 31.133 (30.898-31.190) | 134.231 (132.265-134.608) | **4.31** | 0.53 | same |
| (c) N 1000, L 90-110, unbanded: per-pair fill | 1 | 1.470 (1.469-1.474) | 1.628 (1.625-1.633) | **1.11** | 3.07 | same |
| (c) N 1000, L 90-110, unbanded: per-pair fill | 24 | 31.478 (31.363-33.056) | 38.614 (36.306-38.843) | **1.23** | 0.52 | same |
| (d) dtw, one pair, L 1000, unbanded | 1 | 1.387 (1.378-1.399) | 1.407 (1.406-1.415) | **1.01** | 0.50 | same |
| (d) dtw, one pair, L 1000, band 100 | 1 | 0.859 (0.857-0.859) | 0.881 (0.862-0.883) | **1.03** | 0.50 | same |

band: PASS: the clang-cl wheel ships as built (both lanes fills >= 1.2x at 24 threads, no shape below 0.95x)
problems: none
