# F15 libcxx fingerprint row — 2026-09-01

## Scope

Apple Clang + libc++ is a fourth coherent F15 profile. It is not a skip and
not a generator change. Scalar and per-row `[-1,1]` hashes already matched
the registered `libstdcxx` schedule; the continuous `[-10,10]` accelerator
stream differed by a few ULPs, so `profile_for()` returned nullptr.

Registered before the focused rebuild. Dump subject: production
`accelerator_series_set(3, 4, 42)`, then
`symmetric_zero_diagonal_matrix` with `dtwFull_L<double>` and
`dtwBanded<double>(..., 0)`, IEEE SHA-256 of little-endian binary64 bytes
exactly as `tests/unit/unit_test_deterministic_series.cpp`.

## Environment [confirmed]

```text
Apple clang version 21.0.0 (clang-2100.1.1.101)
STL: libc++ (Apple Command Line Tools)
CMAKE_BUILD_TYPE=Release
flags include: -O3 -fno-math-errno -fno-trapping-math -freciprocal-math
  -fassociative-math -fno-signed-zeros -fno-rounding-math -march=native
host: darwin 25.6.0
```

## Dumped fingerprints [confirmed]

```text
scalar_hash=194FB0E76C52FCD84F09960547EEDC6A43788FDCC89F739DF44E3C49AF7B16E0
rows_hash=1F9E6847BA0FFC7943EBCA024827CD6B5A890B8911BFF4F6C59211F3C28892AB
accelerator_hash=1D063EF12CB8680807CEEEF9F8C2F35331D0AA91186F824A16D9F3B2954AEF77
full_hash=7DE312EABCFFB71D857BF97B9CFCE9C08A6F855E7CE25B863342BE46CBC38B73
band0_hash=E81034CE0654315D254D07FA518DDCE7472A542A4EECBD740935E9DFF891022E
full_ne_band0=1
```

Accelerator binary64 masks:

```text
4017B933480DF696,C01953416B53F15F,40166012566A8F3E,3FFEFDF719BF921C,
BFF15561688E12F2,C0200041BE89C29E,BFEA14A986DAD398,C00A9B4B96F5696D,
C01C92166F9FE92A,4008246451B14AA8,C021BE586FDF2D37,4011C288EB7D97C0
```

Scalar and row hashes match `libstdcxx`. Full and band-0 hashes match
`libstdcxx`. Band-0 differs from full, so the fixture remains non-degenerate.
The accelerator stream differs from `libstdcxx` at four ULP positions
(`…921C` vs `…9220`, `…12F2` vs `…12F0`, `…D398` vs `…D3A0`, `…696D` vs
`…696C`). Mixing those generator bits with another row's oracle hashes is a
fail.

Fingerprint-row name: `libcxx` (not a compiler identity).

## Pass band

`unit_test_deterministic_series` must print

```text
F15_TEST_SUPPORT profile=libcxx scalar=ran row_seeded=ran continuous=ran dense=ran source_audit=ran skips=0
```

and Catch2 `All tests passed` with the existing assertion/case floor. CTest
regex accepts `relaxed|precise|libstdcxx|libcxx`.

## Focused execution [confirmed]

```text
F15_TEST_SUPPORT profile=libcxx scalar=ran row_seeded=ran continuous=ran dense=ran source_audit=ran skips=0
All tests passed (165 assertions in 6 test cases)
```

`ctest -R '^unit_test_deterministic_series$'` → Passed.
