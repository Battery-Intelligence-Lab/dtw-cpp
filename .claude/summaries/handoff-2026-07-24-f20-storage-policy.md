# Handoff - 2026-07-24 - F20 Problem storage policy

## Accomplishments

- Read the live F20 plan, frozen API contract, killed ideas, plan archive,
  `LESSONS.md`, F19 handoff, Task 1.4 history, current Data/DataLoader/Problem
  code, existing storage tests, GPU/CLI direct-vector consumers, and
  Python/MATLAB setter/data bindings.
- Confirmed Task 1.4 tested `DataLoader::load_stored()` only. It did not drive
  `Problem::set_storage_policy`.
- Registered the existing owning `Problem::set_data(Data)` operation as the
  governed cross-language boundary.
- Registered the ownership, transaction, Float32, llfio-OFF, view-preservation,
  copy-lifetime, loader-constructor, GPU-loudness, exact-byte, exact-distance,
  binding, mutation, and full-gate bands in
  `.claude/baselines/2026-07-24-f20-storage-policy.md`.
- Captured clean inherited focused baselines at
  `253dfda8e5d150492261582673b47f7859c7d78e`: canonical 8320 assertions /
  4 cases; llfio-OFF 14 assertions / 4 cases / 2 capability skips.

## Decisions

- Problem series policy governs the next owning `set_data`, not existing data
  retroactively.
- `set_view_data` remains an explicit non-owning bypass.
- Series backing remains separate from distance-matrix mmap and both CLI RAM
  controls.
- Problem must retain a shareable owner for mapped series and names; storing
  only the view Data is forbidden because it dangles.
- `Problem(name, loader)` honors and owns the loader's complete stored result.
- Explicit unsupported Mmap requests fail before publication; no silent heap
  substitution. Auto retains the existing loud best-effort behavior.
- Mapped Data must be rejected before CUDA/Metal's current owning-vector upload
  paths; the CLI keeps series Heap-backed until it has a governed series-policy
  option.
- Product attempts are capped at two.

## Exact resume point

Commit the registration/bookkeeping alone. Then add the permanent non-skipping
F20 Problem route to `tests/unit/test_storage_policy.cpp`, run it against the
inherited product to capture the required both-heap failure, and only then
implement shared series routing plus Problem-owned lifetime. Do not touch F21,
F26, F39, distance-matrix selection, or CLI RAM semantics.

## Open risks

- `LoadedData` is move-only with llfio, while `Problem` is broadly returned and
  copied. A shared backing owner must preserve those source semantics.
- MmapDataStore v1 stores Float64 only. F20 must reject explicit Float32 Mmap
  rather than reinterpret it.
- The default Auto threshold is not observable on Windows because the free-RAM
  query returns unavailable; do not claim that branch validated.
- Fresh Python/MATLAB artifacts may be llfio-OFF. Their decisive real-binding
  proof is then the explicit-Mmap error, not a fake mapped success.
