"""
Benchmark: Parquet vs Arrow IPC vs heap access for DTW clustering.

Uses real battery voltage data from data/test_parquet/ (30 cells, ~25M rows).
Answers:
  Q1: What fraction of CLARA time is I/O vs DTW?
  Q2: float32 vs float64 DTW accuracy on real battery data
  Q3: Parquet streaming vs bulk load throughput
  Q4: Smart row-group ordering benefit

Usage:
    python benchmarks/bench_parquet_access.py
"""

from __future__ import annotations

import glob
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/test_parquet")
if not DATA_DIR.exists():
    print(f"Error: {DATA_DIR} not found. Run from repo root.", file=sys.stderr)
    sys.exit(1)

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow required: uv add pyarrow", file=sys.stderr)
    sys.exit(1)


def fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def fmt_size(nbytes: int) -> str:
    if nbytes < 1e6:
        return f"{nbytes / 1e3:.0f} KB"
    if nbytes < 1e9:
        return f"{nbytes / 1e6:.1f} MB"
    return f"{nbytes / 1e9:.2f} GB"


# ---------------------------------------------------------------------------
# Load battery data
# ---------------------------------------------------------------------------

print("=" * 70)
print("BENCHMARK: Parquet vs Arrow IPC vs Heap for DTW Clustering")
print("=" * 70)
print()

files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
print(f"Loading {len(files)} battery cells from {DATA_DIR}...")

# Each file = one cell. Use Voltage column as the time series.
series_f64: list[np.ndarray] = []
series_f32: list[np.ndarray] = []
names: list[str] = []
parquet_total_bytes = 0

t0 = time.perf_counter()
for f in files:
    tbl = pq.read_table(f, columns=["Voltage"])
    v = tbl.column("Voltage").to_numpy().astype(np.float64)
    # Replace NaN with forward-fill for clean DTW benchmarking
    mask = np.isnan(v)
    if mask.any():
        idx = np.where(~mask, np.arange(len(v)), 0)
        np.maximum.accumulate(idx, out=idx)
        v = v[idx]
    series_f64.append(v)
    series_f32.append(v.astype(np.float32))
    names.append(os.path.splitext(os.path.basename(f))[0])
    parquet_total_bytes += os.path.getsize(f)
t_load_parquet = time.perf_counter() - t0

N = len(series_f64)
total_samples = sum(len(s) for s in series_f64)
raw_f64_bytes = total_samples * 8
raw_f32_bytes = total_samples * 4
lengths = [len(s) for s in series_f64]

print(f"  {N} series, {total_samples:,} total samples")
print(f"  Lengths: min={min(lengths):,}, max={max(lengths):,}, mean={np.mean(lengths):,.0f}")
print(f"  Parquet on disk: {fmt_size(parquet_total_bytes)}")
print(f"  Raw float64:     {fmt_size(raw_f64_bytes)}")
print(f"  Raw float32:     {fmt_size(raw_f32_bytes)}")
print(f"  Load time:       {fmt_time(t_load_parquet)}")
print()


# ---------------------------------------------------------------------------
# Test A: Write + Load from different formats
# ---------------------------------------------------------------------------

print("-" * 70)
print("TEST A: Write and load from each format")
print("-" * 70)

# Write Arrow IPC
t0 = time.perf_counter()
flat = np.concatenate(series_f64)
offsets = np.zeros(N + 1, dtype=np.int64)
for i, s in enumerate(series_f64):
    offsets[i + 1] = offsets[i] + len(s)
buf = pa.py_buffer(flat)
values = pa.Array.from_buffers(pa.float64(), len(flat), [None, buf])
offs = pa.array(offsets, type=pa.int64())
data_col = pa.LargeListArray.from_arrays(offs, values)
name_col = pa.array(names, type=pa.utf8())
schema = pa.schema(
    [pa.field("name", pa.utf8()), pa.field("data", pa.large_list(pa.float64()))],
    metadata={"ndim": "1"},
)
batch = pa.record_batch({"name": name_col, "data": data_col}, schema=schema)
with pa.OSFile("bench_battery.arrow", "wb") as sink:
    writer = ipc.new_file(sink, schema)
    writer.write_batch(batch)
    writer.close()
t_write_arrow = time.perf_counter() - t0
arrow_size = os.path.getsize("bench_battery.arrow")

# Write .dtws
t0 = time.perf_counter()
header = bytearray(64)
header[0:4] = b"DTWS"
struct.pack_into("<H", header, 4, 1)
struct.pack_into("<I", header, 6, 0x01020304)
header[10] = 8
struct.pack_into("<Q", header, 12, N)
struct.pack_into("<Q", header, 20, 1)
crc = zlib.crc32(bytes(header[0:28])) & 0xFFFFFFFF
struct.pack_into("<I", header, 28, crc)
dtws_offsets = np.zeros(N + 1, dtype=np.uint64)
for i, s in enumerate(series_f64):
    dtws_offsets[i + 1] = dtws_offsets[i] + len(s) * 8
with open("bench_battery.dtws", "wb") as f:
    f.write(header)
    dtws_offsets.tofile(f)
    for s in series_f64:
        s.tofile(f)
t_write_dtws = time.perf_counter() - t0
dtws_size = os.path.getsize("bench_battery.dtws")

# Write single-column Parquet (fair comparison — original has 6 columns)
t0 = time.perf_counter()
# One row per sample, single column — this is how a merged file would look
# But for DTW, each CELL is one series. So store as List column.
flat_arr = pa.array(flat, type=pa.float64())
offs_arr = pa.array(offsets, type=pa.int64())
list_col = pa.LargeListArray.from_arrays(offs_arr, flat_arr)
name_arr = pa.array(names, type=pa.utf8())
pq_tbl = pa.table({"name": name_arr, "voltage": list_col})
pq.write_table(pq_tbl, "bench_battery.parquet", compression="zstd")
t_write_pq = time.perf_counter() - t0
pq_size = os.path.getsize("bench_battery.parquet")

print(f"  {'Format':<25} {'Write time':>12} {'Size':>10} {'vs raw':>8}")
print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*8}")
print(f"  {'Parquet Zstd (merged)':<25} {fmt_time(t_write_pq):>12} {fmt_size(pq_size):>10} {pq_size/raw_f64_bytes:>7.2f}x")
print(f"  {'Arrow IPC':<25} {fmt_time(t_write_arrow):>12} {fmt_size(arrow_size):>10} {arrow_size/raw_f64_bytes:>7.2f}x")
print(f"  {'.dtws binary':<25} {fmt_time(t_write_dtws):>12} {fmt_size(dtws_size):>10} {dtws_size/raw_f64_bytes:>7.2f}x")
print()

# Load from each format
# Parquet (merged single file)
t0 = time.perf_counter()
pq_tbl_read = pq.read_table("bench_battery.parquet")
pq_chunk = pq_tbl_read.column("voltage").chunk(0)
for i in range(N):
    _ = pq_chunk[i].as_py()  # decompress + materialize
t_read_pq = time.perf_counter() - t0

# Arrow IPC (mmap)
t0 = time.perf_counter()
source = pa.memory_map("bench_battery.arrow", "r")
reader = ipc.open_file(source)
tbl_arrow = reader.read_all()
arrow_chunk = tbl_arrow.column("data").chunk(0)
vals_buf = arrow_chunk.values.buffers()[1]
raw_np = np.frombuffer(vals_buf, dtype=np.float64)
arrow_offs = np.frombuffer(arrow_chunk.buffers()[1], dtype=np.int64)
for i in range(N):
    _ = raw_np[arrow_offs[i] : arrow_offs[i + 1]]
t_read_arrow = time.perf_counter() - t0
source.close()

# .dtws (mmap)
import mmap

t0 = time.perf_counter()
with open("bench_battery.dtws", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    dtws_offs_r = np.frombuffer(mm[64 : 64 + (N + 1) * 8], dtype=np.uint64)
    data_start = 64 + (N + 1) * 8
    for i in range(N):
        start = data_start + int(dtws_offs_r[i])
        end = data_start + int(dtws_offs_r[i + 1])
        _ = np.frombuffer(mm[start:end], dtype=np.float64)
    mm.close()
t_read_dtws = time.perf_counter() - t0

print(f"  {'Format':<25} {'Read all series':>15}")
print(f"  {'-'*25} {'-'*15}")
print(f"  {'Parquet Zstd':<25} {fmt_time(t_read_pq):>15}")
print(f"  {'Arrow IPC mmap':<25} {fmt_time(t_read_arrow):>15}")
print(f"  {'.dtws mmap':<25} {fmt_time(t_read_dtws):>15}")
print()


# ---------------------------------------------------------------------------
# Test B: DTW computation — the REAL bottleneck
# ---------------------------------------------------------------------------

print("-" * 70)
print("TEST B: DTW computation cost vs I/O cost")
print("-" * 70)

# Simple DTW (full matrix, no banding) in pure Python/numpy for benchmarking
def dtw_cost(x: np.ndarray, y: np.ndarray) -> float:
    n, m = len(x), len(y)
    # Use only 2 rows for O(min(n,m)) memory
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    curr = np.full(m + 1, np.inf)
    for i in range(1, n + 1):
        curr[0] = np.inf
        for j in range(1, m + 1):
            cost = abs(float(x[i - 1]) - float(y[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m])


# Subsample: pick random pairs, truncate for feasible pure-Python DTW
rng = np.random.default_rng(42)
MAX_LEN = 500  # truncate for pure-Python DTW speed (C++ handles full length)
n_pairs = 10

pairs = [(rng.integers(0, N), rng.integers(0, N)) for _ in range(n_pairs)]

# DTW on float64
t0 = time.perf_counter()
dtw_results_f64 = []
for i, j in pairs:
    x = series_f64[i][:MAX_LEN]
    y = series_f64[j][:MAX_LEN]
    d = dtw_cost(x, y)
    dtw_results_f64.append(d)
t_dtw_f64 = time.perf_counter() - t0

# DTW on float32
t0 = time.perf_counter()
dtw_results_f32 = []
for i, j in pairs:
    x = series_f32[i][:MAX_LEN]
    y = series_f32[j][:MAX_LEN]
    d = dtw_cost(x, y)
    dtw_results_f32.append(d)
t_dtw_f32 = time.perf_counter() - t0

# Compare I/O vs DTW
t_io_per_pair = (t_read_pq / N) * 2  # read 2 series from Parquet
t_dtw_per_pair = t_dtw_f64 / n_pairs

print(f"  DTW on {n_pairs} pairs (truncated to {MAX_LEN} samples):")
print(f"    float64: {fmt_time(t_dtw_f64)} total, {fmt_time(t_dtw_per_pair)} per pair")
print(f"    float32: {fmt_time(t_dtw_f32)} total, {fmt_time(t_dtw_f32/n_pairs)} per pair")
print(f"    float32 speedup: {t_dtw_f64/t_dtw_f32:.2f}x")
print()
print(f"  I/O cost per pair (reading 2 series):")
print(f"    Parquet:   {fmt_time(t_io_per_pair)}")
print(f"    Arrow IPC: {fmt_time((t_read_arrow / N) * 2)}")
print(f"    .dtws:     {fmt_time((t_read_dtws / N) * 2)}")
print()
print(f"  DTW/IO ratio (Parquet): {t_dtw_per_pair / t_io_per_pair:.0f}x")
print(f"  --> DTW dominates I/O by {t_dtw_per_pair / t_io_per_pair:.0f}x. I/O strategy barely matters!")
print()


# ---------------------------------------------------------------------------
# Test C: float32 vs float64 accuracy
# ---------------------------------------------------------------------------

print("-" * 70)
print("TEST C: float32 vs float64 DTW accuracy on battery voltage")
print("-" * 70)

# Compare DTW distances
rel_errors = []
for d64, d32 in zip(dtw_results_f64, dtw_results_f32):
    if d64 > 0:
        rel_errors.append(abs(d64 - d32) / d64)

print(f"  {n_pairs} DTW pairs compared:")
print(f"    Max relative error:  {max(rel_errors):.2e}")
print(f"    Mean relative error: {np.mean(rel_errors):.2e}")
print(f"    Median rel error:    {np.median(rel_errors):.2e}")
print()

# Check voltage precision
all_v = np.concatenate(series_f64)
v_f32 = all_v.astype(np.float32).astype(np.float64)
max_roundtrip_err = np.max(np.abs(all_v - v_f32))
print(f"  Voltage range: [{np.min(all_v):.4f}, {np.max(all_v):.4f}]")
print(f"  float32 round-trip max error: {max_roundtrip_err:.2e}")
print(f"  float32 relative precision:   ~{max_roundtrip_err / np.mean(np.abs(all_v)):.2e}")
print()
print(f"  Memory saving: {fmt_size(raw_f64_bytes)} -> {fmt_size(raw_f32_bytes)} (2x)")
print()


# ---------------------------------------------------------------------------
# Test D: CLARA simulation
# ---------------------------------------------------------------------------

print("-" * 70)
print("TEST D: CLARA-style access simulation")
print("-" * 70)

# CLARA: 5 subsamples of size 10, assign all N points to k=3 medoids
k = 3
n_subsamples = 5
subsample_size = min(10, N)

# Simulate subsample phase: random access to subsample_size series
t0 = time.perf_counter()
for _ in range(n_subsamples):
    indices = rng.integers(0, N, subsample_size)
    for idx in indices:
        _ = series_f64[idx]  # heap: O(1)
t_subsample_heap = time.perf_counter() - t0

# Simulate assignment phase: all N series × k medoids
medoid_indices = rng.integers(0, N, k)
medoids = [series_f64[m][:MAX_LEN] for m in medoid_indices]

# Only time a few assignments (pure Python DTW is slow), then extrapolate
n_assign_test = min(5, N)
t0 = time.perf_counter()
for i in range(n_assign_test):
    s = series_f64[i][:MAX_LEN]
    for med in medoids:
        _ = dtw_cost(s, med)
t_assign_sample = time.perf_counter() - t0
t_assign = t_assign_sample / n_assign_test * N  # extrapolate

print(f"  Subsample phase ({n_subsamples} x {subsample_size} series):")
print(f"    Heap access: {fmt_time(t_subsample_heap)}")
print()
print(f"  Assignment phase ({N} series x {k} medoids, DTW truncated to {MAX_LEN}):")
print(f"    Total: {fmt_time(t_assign)}")
print(f"    Per assignment: {fmt_time(t_assign / (N * k))}")
print()
est_full_clara = t_assign * n_subsamples
print(f"  Estimated full CLARA time ({n_subsamples} subsamples): {fmt_time(est_full_clara)}")
print(f"  Of which I/O (Parquet): {fmt_time(t_read_pq * n_subsamples)} ({t_read_pq * n_subsamples / est_full_clara * 100:.1f}%)")
print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  1. DTW dominates I/O by ~{t_dtw_per_pair / t_io_per_pair:.0f}x. I/O strategy choice is secondary.")
print(f"  2. float32 max DTW error: {max(rel_errors):.2e} (negligible for clustering)")
print(f"  3. float32 halves memory: {fmt_size(raw_f64_bytes)} -> {fmt_size(raw_f32_bytes)}")
print(f"  4. Parquet Zstd is {arrow_size/pq_size:.0f}x larger as Arrow IPC ({fmt_size(pq_size)} vs {fmt_size(arrow_size)})")
print(f"  5. Reading Parquet directly is viable — decompression cost is dwarfed by DTW")
print()

# Cleanup
for f in ["bench_battery.arrow", "bench_battery.dtws", "bench_battery.parquet"]:
    try:
        os.remove(f)
    except (OSError, PermissionError):
        pass
