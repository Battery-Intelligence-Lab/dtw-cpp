---
description: "Convert time series between CSV, Parquet, Arrow IPC, and HDF5 formats."
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
| `.arrow`, `.ipc` | Arrow IPC | Memory-mapped, fastest load |
| `.h5`, `.hdf5` | HDF5 | With metadata |
| `.dtwm` | DTWC binary | Distance matrix of `--checkpoint`, not time series |

## Step 1: Detect formats

Derive from extensions. Report sizes:
```bash
ls -lh "INPUT_PATH"
```

## Step 2: Choose tool

**`dtwc-convert` CLI** (if installed via Python package):
- CSV / Parquet / HDF5 → Arrow IPC (`.arrow`, `.ipc`, `.feather`)
- Fastest for large datasets (dtwc_cl memory-maps Arrow IPC)

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
dtwc-convert INPUT_PATH -o OUTPUT_PATH [--columns COL ...] [--name-column NAME_COL]
```

## Step 3b: Python path

```python
import numpy as np
import dtwcpp as dc
from pathlib import Path

inp = Path("INPUT_PATH")
out = Path("OUTPUT_PATH")

# Load from source: an (N, L) array
ext_in = inp.suffix.lower()
if ext_in == ".csv":
    data, _ = dc.load_dataset_csv(str(inp))
elif ext_in == ".parquet":
    data, _ = dc.load_dataset_parquet(str(inp))
elif ext_in in (".h5", ".hdf5"):
    data = dc.load_dataset_hdf5(str(inp))["series"]
elif ext_in in (".arrow", ".ipc"):
    data = np.array(dc.load(str(inp)).as_data().p_vec)
else:
    raise ValueError(f"Unsupported input: {ext_in}")

# Save to target (Arrow IPC: dtwc-convert, Step 3a)
ext_out = out.suffix.lower()
if ext_out == ".csv":
    dc.save_dataset_csv(data, str(out))
elif ext_out == ".parquet":
    dc.save_dataset_parquet(data, str(out))
elif ext_out in (".h5", ".hdf5"):
    dc.save_dataset_hdf5(data, str(out))
else:
    raise ValueError(f"Unsupported output: {ext_out}")

print(f"Converted {len(data)} series from {ext_in} to {ext_out}")
```

## Step 4: Verify round-trip

```python
# Re-load and check shape matches
data2, _ = dc.load_dataset_parquet(str(out))  # or whichever format
assert len(data2) == len(data), f"Size mismatch: {len(data)} → {len(data2)}"
# Check first series matches
import numpy as np
x1, x2 = np.asarray(data[0]), np.asarray(data2[0])
assert np.allclose(x1, x2), "Data mismatch after conversion"
print(f"Round-trip verified: {len(data2)} series, first series matches")
```

## Step 5: Report

```bash
ls -lh "OUTPUT_PATH"
```

Show size ratio (compression effect).

## Tips

- CSV → Parquet typically 5-20× smaller (depending on dtype)
- Parquet → Arrow IPC: sub-second for 10k series, memory-mapped reload
- `.dtwm` is a distance matrix file (`--checkpoint`), not time series
- For very wide dataframes, use `dtwc-convert --columns COL` to select only the series column

## Related

- `/cluster` — use converted data
- `/help data-formats` — format reference
