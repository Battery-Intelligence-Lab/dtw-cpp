# XG report from raw.txt

# wsl: systemd running, waited 0 s for its jobs; jobs now: 0; up 2.92 s
# info gcc_base: info gcc 13.3.0 W f64 8 f32 16 x86_64 1 FMA 1 AVX2 1 omp threads 24
# info gcc_cand: info gcc 13.3.0 W f64 16 f32 32 x86_64 1 FMA 1 AVX2 1 omp threads 24

clock of vCPU 2 (dependent register-add chain, median over the round's probes): round 0 5.380 GHz, round 1 5.354 GHz, round 2 5.355 GHz
- rounds whose clock differs from the rounds' median by more than 3%: none
- probes whose own clock moved by more than 3% over their measurements: 81
  - round 0 gcc_base f64 L1 L 100 band -1: 5.160-5.402 GHz
  - round 0 gcc_cand f64 L1 L 100 band -1: 4.842-5.387 GHz
  - round 0 gcc_base f64 L1 L 100 band 10: 5.186-5.434 GHz
  - round 0 gcc_cand f64 L1 L 100 band 10: 5.257-5.454 GHz
  - round 0 gcc_base f64 L1 L 1000 band -1: 4.778-5.459 GHz
  - round 0 gcc_cand f64 L1 L 1000 band -1: 5.192-5.558 GHz
  - round 0 gcc_base f64 Sq L 100 band -1: 5.152-5.315 GHz
  - round 0 gcc_cand f64 Sq L 100 band -1: 5.361-5.564 GHz
  - round 0 gcc_base f64 Sq L 100 band 10: 5.274-5.498 GHz
  - round 0 gcc_cand f64 Sq L 100 band 10: 5.267-5.525 GHz
  - round 0 gcc_cand f64 Sq L 1000 band -1: 5.234-5.431 GHz
  - round 0 gcc_base f64 Sq L 1000 band 100: 5.224-5.413 GHz
  - round 0 gcc_cand f64 Sq L 1000 band 100: 5.345-5.562 GHz
  - round 0 gcc_base f32 L1 L 100 band -1: 5.374-5.562 GHz
  - round 0 gcc_cand f32 L1 L 100 band -1: 5.273-5.442 GHz
  - round 0 gcc_base f32 L1 L 100 band 10: 5.054-5.476 GHz
  - round 0 gcc_cand f32 L1 L 100 band 10: 5.101-5.531 GHz
  - round 0 gcc_base f32 L1 L 1000 band -1: 5.309-5.562 GHz
  - round 0 gcc_base f32 L1 L 1000 band 100: 4.930-5.519 GHz
  - round 0 gcc_cand f32 L1 L 1000 band 100: 5.265-5.455 GHz
  - round 0 gcc_base f32 Sq L 100 band -1: 5.203-5.555 GHz
  - round 0 gcc_cand f32 Sq L 100 band -1: 5.204-5.413 GHz
  - round 0 gcc_base f32 Sq L 100 band 10: 5.362-5.528 GHz
  - round 0 gcc_cand f32 Sq L 100 band 10: 4.984-5.518 GHz
  - round 0 gcc_base f32 Sq L 1000 band -1: 5.049-5.479 GHz
  - round 0 gcc_base f32 Sq L 1000 band 100: 5.211-5.400 GHz
  - round 0 gcc_cand f32 Sq L 1000 band 100: 5.179-5.538 GHz
  - round 1 gcc_cand f64 L1 L 100 band -1: 4.810-5.516 GHz
  - round 1 gcc_cand f64 L1 L 100 band 10: 4.917-5.527 GHz
  - round 1 gcc_base f64 L1 L 100 band 10: 5.222-5.561 GHz
  - round 1 gcc_cand f64 L1 L 1000 band -1: 5.314-5.566 GHz
  - round 1 gcc_cand f64 L1 L 1000 band 100: 5.296-5.517 GHz
  - round 1 gcc_base f64 L1 L 1000 band 100: 5.226-5.563 GHz
  - round 1 gcc_base f64 Sq L 100 band -1: 5.222-5.480 GHz
  - round 1 gcc_cand f64 Sq L 100 band 10: 5.228-5.502 GHz
  - round 1 gcc_base f64 Sq L 100 band 10: 3.956-5.309 GHz
  - round 1 gcc_cand f64 Sq L 1000 band -1: 4.671-5.230 GHz
  - round 1 gcc_base f64 Sq L 1000 band -1: 5.033-5.255 GHz
  - round 1 gcc_cand f32 L1 L 100 band -1: 5.329-5.517 GHz
  - round 1 gcc_base f32 L1 L 100 band -1: 5.204-5.561 GHz
  - round 1 gcc_cand f32 L1 L 100 band 10: 5.188-5.516 GHz
  - round 1 gcc_base f32 L1 L 100 band 10: 5.159-5.498 GHz
  - round 1 gcc_cand f32 L1 L 1000 band -1: 5.278-5.488 GHz
  - round 1 gcc_base f32 L1 L 1000 band -1: 5.332-5.515 GHz
  - round 1 gcc_cand f32 L1 L 1000 band 100: 5.358-5.536 GHz
  - round 1 gcc_base f32 L1 L 1000 band 100: 4.925-5.358 GHz
  - round 1 gcc_cand f32 Sq L 100 band -1: 5.156-5.476 GHz
  - round 1 gcc_base f32 Sq L 100 band -1: 5.187-5.533 GHz
  - round 1 gcc_cand f32 Sq L 100 band 10: 5.207-5.424 GHz
  - round 1 gcc_base f32 Sq L 100 band 10: 5.241-5.503 GHz
  - round 1 gcc_cand f32 Sq L 1000 band -1: 4.788-5.407 GHz
  - round 1 gcc_base f32 Sq L 1000 band -1: 5.099-5.501 GHz
  - round 1 gcc_cand f32 Sq L 1000 band 100: 5.241-5.461 GHz
  - round 1 gcc_base f32 Sq L 1000 band 100: 5.243-5.496 GHz
  - round 2 gcc_base f64 L1 L 100 band -1: 5.222-5.525 GHz
  - round 2 gcc_cand f64 L1 L 100 band -1: 4.718-5.474 GHz
  - round 2 gcc_base f64 L1 L 100 band 10: 5.236-5.444 GHz
  - round 2 gcc_cand f64 L1 L 100 band 10: 4.826-5.519 GHz
  - round 2 gcc_base f64 L1 L 1000 band -1: 5.187-5.382 GHz
  - round 2 gcc_cand f64 L1 L 1000 band -1: 5.008-5.254 GHz
  - round 2 gcc_cand f64 L1 L 1000 band 100: 5.209-5.562 GHz
  - round 2 gcc_base f64 Sq L 100 band -1: 5.230-5.389 GHz
  - round 2 gcc_base f64 Sq L 100 band 10: 5.293-5.546 GHz
  - round 2 gcc_cand f64 Sq L 100 band 10: 4.751-5.502 GHz
  - round 2 gcc_base f64 Sq L 1000 band -1: 5.239-5.559 GHz
  - round 2 gcc_cand f64 Sq L 1000 band -1: 5.154-5.483 GHz
  - round 2 gcc_base f64 Sq L 1000 band 100: 4.736-5.538 GHz
  - round 2 gcc_cand f64 Sq L 1000 band 100: 4.779-5.415 GHz
  - round 2 gcc_base f32 L1 L 100 band -1: 5.244-5.568 GHz
  - round 2 gcc_cand f32 L1 L 100 band -1: 5.268-5.516 GHz
  - round 2 gcc_base f32 L1 L 100 band 10: 4.991-5.388 GHz
  - round 2 gcc_base f32 L1 L 1000 band -1: 5.239-5.475 GHz
  - round 2 gcc_cand f32 L1 L 1000 band -1: 5.279-5.540 GHz
  - round 2 gcc_base f32 L1 L 1000 band 100: 5.255-5.498 GHz
  - round 2 gcc_cand f32 L1 L 1000 band 100: 5.122-5.522 GHz
  - round 2 gcc_base f32 Sq L 100 band -1: 5.171-5.358 GHz
  - round 2 gcc_cand f32 Sq L 100 band -1: 5.300-5.557 GHz
  - round 2 gcc_base f32 Sq L 100 band 10: 5.254-5.547 GHz
  - round 2 gcc_cand f32 Sq L 100 band 10: 4.963-5.529 GHz
  - round 2 gcc_base f32 Sq L 1000 band 100: 5.248-5.450 GHz
  - round 2 gcc_cand f32 Sq L 1000 band 100: 5.292-5.561 GHz

## gcc: candidate (cand) against base, median over rounds; ratio = base ns / cand ns (single thread) or cand Gcell/s / base Gcell/s (fill), per round

| shape | W base/cand | base ns/cell | cand ns/cell | ratio (median) | range | cycles/row step base/cand | clock GHz base/cand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f32 L1 L 100 unbanded | 16/32 | 0.09196 | 0.04349 | 2.115 | 2.067-2.160 | 7.90 / 7.57 | 5.453 / 5.417 |
| f32 L1 L 100 band 10 | 16/32 | 0.06815 | 0.05013 | 1.368 | 1.311-1.370 | 5.82 / 8.61 | 5.386 / 5.369 |
| f32 L1 L 1000 unbanded | 16/32 | 0.08395 | 0.04246 | 1.941 | 1.936-2.001 | 7.16 / 7.29 | 5.380 / 5.341 |
| f32 L1 L 1000 band 100 | 16/32 | 0.08009 | 0.04712 | 1.700 | 1.662-1.722 | 6.82 / 8.13 | 5.290 / 5.370 |
| f32 Sq L 100 unbanded | 16/32 | 0.11597 | 0.05431 | 2.134 | 2.128-2.163 | 9.88 / 9.34 | 5.326 / 5.357 |
| f32 Sq L 100 band 10 | 16/32 | 0.07354 | 0.05434 | 1.355 | 1.342-1.370 | 6.35 / 9.34 | 5.429 / 5.326 |
| f32 Sq L 1000 unbanded | 16/32 | 0.10722 | 0.05444 | 1.948 | 1.934-1.983 | 9.13 / 9.44 | 5.407 / 5.352 |
| f32 Sq L 1000 band 100 | 16/32 | 0.10060 | 0.05814 | 1.736 | 1.709-1.766 | 8.59 / 9.98 | 5.326 / 5.364 |
| f64 L1 L 100 unbanded | 8/16 | 0.34237 | 0.09345 | 3.634 | 3.572-3.678 | 14.38 / 8.02 | 5.254 / 5.368 |
| f64 L1 L 100 band 10 | 8/16 | 0.34469 | 0.08997 | 3.831 | 3.723-3.892 | 14.77 / 7.77 | 5.376 / 5.384 |
| f64 L1 L 1000 unbanded | 8/16 | 0.33426 | 0.08649 | 3.865 | 3.827-3.909 | 14.08 / 7.33 | 5.303 / 5.460 |
| f64 L1 L 1000 band 100 | 8/16 | 0.34025 | 0.08858 | 3.856 | 3.761-3.859 | 14.35 / 7.62 | 5.359 / 5.365 |
| f64 Sq L 100 unbanded | 8/16 | 0.34461 | 0.11512 | 2.963 | 2.956-3.035 | 14.67 / 10.09 | 5.377 / 5.450 |
| f64 Sq L 100 band 10 | 8/16 | 0.34344 | 0.10248 | 3.337 | 3.313-3.351 | 14.77 / 8.90 | 5.366 / 5.341 |
| f64 Sq L 1000 unbanded | 8/16 | 0.33473 | 0.10809 | 3.088 | 3.070-3.140 | 14.42 / 9.13 | 5.380 / 5.350 |
| f64 Sq L 1000 band 100 | 8/16 | 0.33612 | 0.11046 | 2.974 | 2.920-3.062 | 14.13 / 9.58 | 5.268 / 5.308 |

| fill (f64 L1, OpenMP default threads) | W base/cand | base Gcell/s | cand Gcell/s | ratio (median) | range | matrices |
| --- | --- | --- | --- | --- | --- | --- |
| N 2000 L 100 unbanded, 24 threads | 8/16 | 56.00 | 134.91 | 2.387 | 2.248-2.409 | identical |
| N 500 L 1000 band 100, 24 threads | 8/16 | 52.85 | 135.27 | 2.539 | 2.530-2.575 | identical |
| N 1000 L 100 +-10 ragged unbanded, 24 threads | 8/16 | 24.64 | 23.15 | 0.990 | 0.932-1.003 | identical |

| sweep (check/*.txt) | W base/cand | lane outputs | hash base | hash cand | lanes vs per-pair mismatches base / cand | outside the route bound base / cand | worst / (2n-1) eps max, base / cand (bound 2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f32 L1 | 16/32 | 4,140,000 | `46691aba0dcc9185` | `46691aba0dcc9185` | 0 / 0 | 0 / 0 | 0 / 0 |
| f32 Sq | 16/32 | 4,140,000 | `1ed88e3d48203e14` | `e01d201911ab52c7` DIFFER | 671,722 / 672,783 | 0 / 0 | 0.3278 / 0.3278 |
| f64 L1 | 8/16 | 4,140,000 | `24e36e73a3691ea6` | `24e36e73a3691ea6` | 0 / 0 | 0 / 0 | 0 / 0 |
| f64 Sq | 8/16 | 4,140,000 | `b222c31b78321559` | `a38d65676974ef64` DIFFER | 332 / 673,733 | 0 / 0 | 0.1329 / 0.3095 |

- lane outputs equal at both widths: NO; per-pair route unchanged (4 hashes): yes; fill matrices equal: yes
- the ruling's reading: L1 lanes bit for bit at both widths (with the per-pair route and the fill matrices) -> yes; squared lanes within the route bound at both widths -> yes; clustering identical -> yes (CLI runs compared: 64 (base and candidate); exit 0 in both: 64/64; labels identical: 64/64; medoids identical: 64/64; cpp_conformance passed in 2/2 builds, its regenerated reference equal to the tracked one in 2/2)
- timing band: every single-thread shape >= 1.1x: min 1.355 (16 shapes) -> pass; both equal-length fills >= 1.1x: 2.387, 2.539 -> pass; ragged fill in 0.97-1.03x: 0.990 -> pass
- **verdict gcc under the ruling (timing band + L1 identical + squared within the route bound + clustering identical): PASS**
- beside it, the literal reading (timing band + every lane output as at 64): FALSIFIED

