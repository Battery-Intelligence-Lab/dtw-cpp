---
description: "Convert time series between CSV, Parquet, Arrow IPC, HDF5, and .dtws formats."
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
---

# DTWC++ Convert

Convert data between formats. `$ARGUMENTS` has input path and output path/format.

## Format support

| Extension | Name | When |
|-----------|------|------|
| `.csv` | CSV | Human-readable, portable |
| `.parquet` | Parquet | Compressed, columnar, best for N > 10k |
| `.arrow`, `.ipc` | Arrow IPC | Zero-copy mmap, fastest load |
| `.h5`, `.hdf5` | HDF5 | With metadata |
| `.dtws` | DTWC binary | Internal distance matrix cache |

## Step 1: Detect formats

Derive from extensions. Report sizes:
```bash
ls -lh "INPUT_PATH"
```

## Step 2: Choose tool

**`dtwc-convert` CLI** (if installed via Python package):
- CSV ↔ Parquet ↔ Arrow IPC ↔ HDF5
- Fastest for large datasets (zero-copy where possible)

**Python I/O module** (`dtwcpp.io`):
- Flexible, programmable
- Good for subsetting / filtering / preprocessing during conversion

**Check availability:**
```bash
which dtwc-convert && echo "CLI available"
python3 -c "import dtwcpp.io" 2>/dev/null && echo "Python available"
```

## Step 3a: CLI path (preferred for large data)

```bash
dtwc-convert INPUT_PATH OUTPUT_PATH [--column COL] [--name-column NAME_COL]
```

## Step 3b: Python path

```python
import dtwcpp as dc
from pathlib import Path

inp = Path("INPUT_PATH")
out = Path("OUTPUT_PATH")

# Load from source
ext_in = inp.suffix.lower()
if ext_in == ".csv":
    data = dc.load_dataset_csv(str(inp))
elif ext_in == ".parquet":
    data = dc.load_dataset_parquet(str(inp))
elif ext_in in (".h5", ".hdf5"):
    data = dc.load_dataset_hdf5(str(inp))
elif ext_in in (".arrow", ".ipc"):
    data = dc.load_dataset_arrow_ipc(str(inp))
else:
    raise ValueError(f"Unsupported input: {ext_in}")

# Save to target
ext_out = out.suffix.lower()
if ext_out == ".csv":
    dc.save_dataset_csv(data, str(out))
elif ext_out == ".parquet":
    dc.save_dataset_parquet(data, str(out))
elif ext_out in (".h5", ".hdf5"):
    dc.save_dataset_hdf5(data, str(out))
elif ext_out in (".arrow", ".ipc"):
    dc.save_dataset_arrow_ipc(data, str(out))
else:
    raise ValueError(f"Unsupported output: {ext_out}")

print(f"Converted {data.size} series from {ext_in} to {ext_out}")
```

## Step 4: Verify round-trip

```python
# Re-load and check shape matches
data2 = dc.load_dataset_parquet(str(out))  # or whichever format
assert data2.size == data.size, f"Size mismatch: {data.size} → {data2.size}"
# Check first series matches
import numpy as np
x1, x2 = np.asarray(data[0]), np.asarray(data2[0])
assert np.allclose(x1, x2), "Data mismatch after conversion"
print(f"Round-trip verified: {data2.size} series, first series matches")
```

## Step 5: Report

```bash
ls -lh "OUTPUT_PATH"
```

Show size ratio (compression effect).

## Tips

- CSV → Parquet typically 5-20× smaller (depending on dtype)
- Parquet → Arrow IPC: sub-second for 10k series, zero-copy reload
- `.dtws` is an internal distance matrix format, not time series
- For very wide dataframes, use `--column` to select only the series column

## Related

- `/cluster` — use converted data
- `/help data-formats` — format reference
