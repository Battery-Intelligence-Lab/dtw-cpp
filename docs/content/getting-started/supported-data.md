---
title: Supported data formats
weight: 3
---


# Supported data formats

DTW-C++ supports importing data from several data structures. The core
`Data`/`Problem` APIs support both univariate and multivariate series, but the
row-oriented file examples on this page are still mostly univariate. Input data
can either be read from multiple files (one per series), or from a single
file, where each time series is represented row-wise or through one of the
columnar formats below.

## Reading data from disk

You can specify either a _file path_ or a _folder path_ for your data. The software accommodates `*.csv` and `*.tsv` file extensions.

### Specifying a file path

A _file path_ points to a single file containing all of your data, represented in variable-length rows. The values might be separated by commas, tabs, or spaces. In this scenario, time-series names are allocated sequentially from 1 to N, row by row.

A blank line is not a series: blank lines at the end of the file are ignored, and a blank line followed by more data is an error naming its row. Write a missing value as `nan`; an empty field is an error. Numbers are read the same way on every platform and in every locale: `.` is the decimal separator, a UTF-8 byte-order mark at the start of the file is skipped, and a Ctrl-Z byte is a non-numeric field like any other, not the end of the file. `dtwc_cl`, C++ `dtwc::read_data`, Python `dtwcpp.load()` and MATLAB `dtwc.load()` read text with this one reader.

**Example:** A file with 5 time-series of varying lengths:

|       |       |       |       |       |       |       |       |       |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 43.87 | 48.98 | 27.60 | 49.84 | 75.13 | 95.93 |       |       |       |
| 38.16 | 44.56 |       |       |       |       |       |       |       |
| 76.55 | 64.63 | 65.51 | 6.10  | 29.11 | 13.86 | 81.43 | 25.11 |       |
| 79.52 | 70.94 | 16.26 | 58.53 | 69.91 | 14.93 | 24.35 | 61.60 | 12.71 |
| 18.69 | 75.47 | 11.90 | 22.38 |       |       |       |       |       |

### Specifying a folder path

A _folder path_ can contain multiple individual files, each representing a _single_ time series. Each line of these files holds exactly one value, after the columns skipped with `--skip-cols`; a line with more fields is an error, not a line whose extra fields are dropped. The values may be separated by commas, tabs, or spaces. In this case, time-series names are derived from the individual file names. Hidden files (names starting with `.`, such as `.DS_Store` or `.gitkeep`) are not series and are skipped. Blank lines follow the single-file rule above: ignored at the end, an error before more data.

**Example:** A `file.csv` is available, with two columns separated by a comma; the first column is just an index, and the second column is the actual data. The index column must be skipped; read without doing so, the file is rejected. Files written by pandas start with a `,0` header row. To read the data points in such files, pass `--skip-rows 1 --skip-cols 1` to `dtwc_cl` (`start_row(1).start_column(1)` on a C++ `DataLoader`, `skip_rows=1, skip_cols=1` in Python).

|   | , | 0     |
|---|---|-------|
| 0 | , | 43.87 |
| 1 | , | 48.98 |
| 2 | , | 27.60 |
| 3 | , | 49.84 |
| 4 | , | 75.13 |
| 5 | , | 95.93 |

## Parquet

Parquet provides columnar, compressed storage. It requires
`-DDTWC_ENABLE_ARROW=ON` at build time (which pulls in Apache Arrow). The eager
and row-group readers use one shared schema rule: the selected top-level column
must be Float32, Float64, `List<Float32/Float64>`, or
`LargeList<Float32/Float64>`. `--column` selects it explicitly; when omitted,
the first eligible top-level column is used.

### Single file — one scalar column as one series

For a scalar Float32/Float64 column, all rows form one time series. The rows are
not independent clustering points.

CLI:

```bash
dtwc_cl -i data.parquet --column Voltage -k 5
```

C++:

```cpp
problem.set_data(dtwc::read_data("data.parquet", 0, 0, '\0', "Voltage"));
```

### Directory of Parquet files

Directory input eagerly concatenates the selected column from each
`.parquet`/`.pq` file, in the order a folder of CSV files is read (sorted, hidden
files skipped). With scalar columns this is one series per file, named from the
filename. List columns contribute one series per list row, named `series_<i>`
and numbered on across the files, so the files of a folder never repeat a
`series_<i>` name.

CLI:

```bash
dtwc_cl -i /path/to/parquet_folder/ --column Voltage -k 5
```

### List columns (list-per-row encoding)

A Parquet file may store all series in one List/LargeList Float32/Float64
column. Each list cell is one variable-length series and receives the stable
name `series_0`, `series_1`, and so on. This layout is produced by
`dtwc-convert`.

```bash
dtwc_cl -i data.parquet --column series -k 5
```

Python reads this layout, from a file or a folder of them, with the installed
pyarrow (the `dtwcpp[parquet]` extra; the wheel links no Arrow C++): each row of
the first list column is a series, named by the first string column, else
`series_<i>`. Without pyarrow, reading raises `ImportError` naming the extra.

```python
data = dtwcpp.load("data.parquet").as_data()
```

### Metadata-first RAM-limited streaming

`--ram-limit` is checked from Parquet schema and row-group metadata before the
selected payload is materialised. When the conservative decode/materialisation
estimate exceeds the cap, the CLI can stream only a single list-per-row file
through non-full FastCLARA:

```bash
dtwc_cl -i data.parquet --column series -k 5 --method clara \
  --sample-size 500 --ram-limit 2GiB
```

Scalar-column input, directories, non-CLARA methods, a sample resolving to all
N series, and non-full CLARA with CUDA fail loudly while over budget. Parquet
row groups are indivisible; rewrite the file with smaller row groups if one
cannot fit beside retained sample/medoid data. Float32 streaming remains
Float32 through sample, medoid, and assignment payloads. The cap governs series
decoding/materialisation rather than total process RSS.

The streamed route keeps a settings-only `Problem` and does not build a parent
distance matrix. It writes labels, medoids, and the binary clustering-result
checkpoint; dense distance-matrix and silhouette CSVs are omitted. For the
same seed and settings, those three emitted artifacts are byte-identical to the
resident list-column route.

---

## Arrow IPC (Feather v2)

Arrow IPC (`.arrow` / `.ipc` / `.feather`) is **memory-mapped**: the file's buffers are read in place, with no decoding step, and each series is copied once into memory. Preferred for repeated clustering runs on the same dataset.

Requires `-DDTWC_ENABLE_ARROW=ON`.

CLI:

```bash
dtwc_cl -i data.arrow -k 10
```

C++:

```cpp
problem.set_data(dtwc::read_data("data.arrow"));
```

Python reads it with the installed pyarrow (the `dtwcpp[parquet]` extra), the
same columns and names:

```python
data = dtwcpp.load("data.arrow").as_data()
```

**Schema:** a `data` column of `List` or `LargeList` (more than 2 billion values) of `Float32`/`Float64` holds one series per row, across every record batch; an optional `name` column of `Utf8` or `LargeUtf8` (Polars' default) names the series (without one, or for a null name, a series is `series_<i>`), and a `name` column of any other type is an error. The schema metadata `ndim` gives the features per time step (default 1). A null series or value is an error. Create Arrow IPC files with the `dtwc-convert` tool — see [Data formats and conversion](../../guides/data-formats/).

---

## Reading data directly

Series you have already read, with numpy, pandas or plain Python, go in as they
are. `dtwcpp.cluster`, `dtwcpp.DTWClustering.fit`, `dtwcpp.load` and
`Problem.set_data` take a 2-D array (one series per row), a list of 1-D arrays or
lists (series of any lengths) and a pandas DataFrame (one series per row, named by
its index; other inputs are named by their ordinals):

```python
import numpy as np
import dtwcpp

X = np.loadtxt("series.csv", delimiter=",")              # 2-D: one series per row
result = dtwcpp.cluster(X, k=3)
ragged = [np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0])]  # lengths may differ
labels = dtwcpp.DTWClustering(n_clusters=2).fit(ragged).labels_
# a pandas DataFrame: dtwcpp.cluster(df, k=3), series named by df.index

prob = dtwcpp.Problem("mine")
prob.set_data(X)                                           # names "0", "1", ...
```

If you are using DTW-C++ directly (e.g., as a library within your software), you might prefer to read data independently or use pre-generated data. DTW-C++ employs the `Data` class to encapsulate a `std::vector<std::vector<data_type>>` data object and `std::vector<std::string>` for their corresponding names. The following example code snippet demonstrates how to input data into a Problem object.

```cpp
// Your data generation routine:
std::vector<std::vector<double>> myData = generate_some_data();
std::vector<std::string> myNames = generate_some_names();
//-------------------------------

// Create a Problem object:
dtwc::Problem myProblem{'problem_name'};
auto myDataObject = dtwc::Data(std::move(myData), std::move(myNames));
myProblem.set_data(myDataObject);
/* Other settings */
```
