# @author Volkan Kumtepeli
"""Write the Arrow IPC reader fixtures beside this script, with pyarrow 23.0.1.

The C++ reader (tests/unit/io/test_arrow_ipc.cpp), Python (tests/python/test_io.py) and MATLAB
(tests/matlab/test_tier1_route_parity.m) read these files and expect the same series and names, or the same error:

- ipc_list.arrow: data list<double>, name utf8 with a null: [0 0.5 0.25], [1.5 2], [3], named a, series_1, c;
- ipc_large_list.arrow: the same as large_list<double> and large_utf8;
- ipc_float32.arrow: the same values as list<float>, named x, y, z;
- ipc_no_name.arrow: the same values, no name column: series_0, series_1, series_2;
- ipc_ndim2.arrow: schema metadata ndim 2: [0 1 2 3] and [4 5], two and one time steps;
- ipc_batches.arrow: three record batches, [0 1], [2] | [3] | [4 5], [6], [7 8 9], named a, b | null | d, null, f;
- ipc_stream.arrow: ipc_list.arrow's series and names in the stream format, as two record batches;
- ipc_null_series.arrow, ipc_null_value.arrow: a null series at row 1; a null value inside the series at row 1;
- feather_v1.feather: Feather version 1, which is not Arrow IPC;
- ipc_scalar_data.arrow, ipc_int64_names.arrow, ipc_ndim_2x.arrow: a Float64 'data' column, an Int64 'name' column,
  ndim "2x";
- ipc_dictionary.arrow: ipc_no_name.arrow's series beside a dictionary-encoded column, which nanoarrow cannot decode;
- ipc_lz4.arrow, ipc_zstd.arrow: ipc_no_name.arrow's series, LZ4- and ZSTD-compressed, which the reader decodes;
- ipc_lz4_batches.arrow, ipc_zstd_batches.arrow: ipc_batches.arrow's series and names, in its three batches, alike;
- ipc_v4_union.arrow: metadata version V4 and LZ4, a sparse union column before ipc_no_name.arrow's series, which the
  reader refuses (nanoarrow's single-column decode misplaces the buffers after a pre-1.0 union).

FastCLARA's stream of an Arrow IPC file (tests/unit/io/test_ipc_stream.cpp, tests/integration/test_cli_ipc_stream.cmake):

- clara_256.arrow, clara_256_lz4.arrow, clara_256_zstd.arrow: 256 random walks of 4 steps of k/8 (an LCG's), in eight
  record batches of 32, named r<i> but every seventh null; uncompressed, LZ4 and ZSTD; clara_256.csv their CSV twin;
- clara_256_f32.arrow: the same as list<float>, LZ4; clara_256_ndim2.arrow: the same, LZ4, as time steps of two
  features (metadata ndim 2), every third series with its last time step repeated: two time steps or three, so that
  a band of 1 holds in time steps and not in values;
- clara_verdict_rows.arrow, clara_verdict_union.arrow, clara_verdict_nan.arrow, clara_verdict_band.arrow: 64 walks of 8
  steps, no name column, LZ4, four record batches of 16, poisoned as each test case names (CASES below; the rows sampled
  by a CLARA of sample_size 8, one sample, seed 3, are 10 15 22 23 30 37 52 55). With --parquet-twins <dir>, their
  Parquet twins (column 'data', a row group a record batch), whose stream gave the cases' texts;
- clara_verdict_ndim.arrow: the same walks as four time steps of two features (ndim 2), row 41's cut to 7 values.

The stream's threads' scratch within --ram-limit (tests/unit/core/test_thread_scratch.cpp):

- clara_long_late.arrow: 2,048 walks of 8 steps, row 1,500's of 1,500, LZ4, record batches of 64: a series longer than
  any the test's samples hold, read late in the pass;
- clara_ragged.arrow: 512 walks of 6 to 10 steps, uncompressed, record batches of 32: unequal lengths under a band of
  4, streamed as Float32 (a copy: no codec's scratch beside a record batch).

Every value is exact in Float32. Run: uv run --no-project --with pyarrow==23.0.1 python make_ipc_fixtures.py
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SERIES = [[0.0, 0.5, 0.25], [1.5, 2.0], [3.0]]


def write(name, table, stream=False, compression=None, metadata_version=pa.ipc.MetadataVersion.V5):
    options = pa.ipc.IpcWriteOptions(compression=compression, metadata_version=metadata_version)
    with pa.OSFile(str(HERE / name), "wb") as sink:
        open_writer = pa.ipc.new_stream if stream else pa.ipc.new_file
        with open_writer(sink, table.schema, options=options) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)


def table(columns, metadata=None, batches=None):
    """A table of `columns`, as one record batch or as `batches` (lists of rows, one list a batch)."""
    t = pa.table(columns)
    if metadata:
        t = t.replace_schema_metadata(metadata)
    if batches:
        offsets = [0]
        for size in batches:
            offsets.append(offsets[-1] + size)
        t = pa.Table.from_batches([t.slice(a, b - a).combine_chunks().to_batches()[0]
                                   for a, b in zip(offsets, offsets[1:])], t.schema)
    return t


f64 = pa.list_(pa.float64())
write("ipc_list.arrow", table({"data": pa.array(SERIES, f64), "name": pa.array(["a", None, "c"], pa.string())}))
write("ipc_large_list.arrow", table({"data": pa.array(SERIES, pa.large_list(pa.float64())),
                                     "name": pa.array(["a", None, "c"], pa.large_string())}))
write("ipc_float32.arrow", table({"data": pa.array(SERIES, pa.list_(pa.float32())),
                                  "name": pa.array(["x", "y", "z"], pa.string())}))
write("ipc_no_name.arrow", table({"data": pa.array(SERIES, f64)}))
write("ipc_ndim2.arrow", table({"data": pa.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0]], f64)}, {"ndim": "2"}))
write("ipc_batches.arrow", table({
    "data": pa.array([[0.0, 1.0], [2.0], [3.0], [4.0, 5.0], [6.0], [7.0, 8.0, 9.0]], f64),
    "name": pa.array(["a", "b", None, "d", None, "f"], pa.string())}, batches=[2, 1, 3]))
write("ipc_stream.arrow", table({"data": pa.array(SERIES, f64), "name": pa.array(["a", None, "c"], pa.string())},
                                batches=[1, 2]), stream=True)
write("ipc_null_series.arrow", table({"data": pa.array([[0.0, 1.0], None, [2.0]], f64)}))
write("ipc_null_value.arrow", table({"data": pa.array([[0.0, 1.0], [2.0, None]], f64)}))
feather.write_feather(pa.table({"data": pa.array([0.0, 1.0], pa.float64())}), str(HERE / "feather_v1.feather"),
                      version=1)
write("ipc_scalar_data.arrow", table({"data": pa.array([0.0, 1.0], pa.float64())}))
write("ipc_int64_names.arrow", table({"data": pa.array(SERIES, f64), "name": pa.array([7, 8, 9], pa.int64())}))
write("ipc_ndim_2x.arrow", table({"data": pa.array(SERIES, f64)}, {"ndim": "2x"}))
write("ipc_dictionary.arrow", table({"data": pa.array(SERIES, f64),
                                     "label": pa.array(["u", "v", "u"]).dictionary_encode()}))
for codec in ("lz4", "zstd"):
    write(f"ipc_{codec}.arrow", table({"data": pa.array(SERIES, f64)}), compression=codec)
    write(f"ipc_{codec}_batches.arrow", table({
        "data": pa.array([[0.0, 1.0], [2.0], [3.0], [4.0, 5.0], [6.0], [7.0, 8.0, 9.0]], f64),
        "name": pa.array(["a", "b", None, "d", None, "f"], pa.string())}, batches=[2, 1, 3]), compression=codec)
union = pa.UnionArray.from_sparse(pa.array([0, 0, 0], pa.int8()), [pa.array([7, 8, 9], pa.int32())], field_names=["i"])
write("ipc_v4_union.arrow", table({"u": union, "data": pa.array(SERIES, f64)}), compression="lz4",
      metadata_version=pa.ipc.MetadataVersion.V4)


def walks(rows, steps, seed):
    """`rows` random walks of `steps` steps of k/8, k in [-8, 8], from an LCG seeded with `seed`: exact in Float32."""
    state, out = seed, []
    for _ in range(rows):
        x, walk = 0.0, []
        for _ in range(steps):
            state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
            x += ((state >> 33) % 17 - 8) / 8
            walk.append(x)
        out.append(walk)
    return out


CLARA = walks(256, 4, 7)
clara_names = pa.array([None if i % 7 == 3 else f"r{i}" for i in range(len(CLARA))], pa.string())
for name, codec, values in (("clara_256.arrow", None, f64), ("clara_256_lz4.arrow", "lz4", f64),
                            ("clara_256_zstd.arrow", "zstd", f64), ("clara_256_f32.arrow", "lz4", pa.list_(pa.float32()))):
    write(name, table({"data": pa.array(CLARA, values), "name": clara_names}, batches=[32] * 8), compression=codec)
NDIM2 = [walk + walk[2:] if i % 3 == 0 else walk for i, walk in enumerate(CLARA)]
write("clara_256_ndim2.arrow", table({"data": pa.array(NDIM2, f64), "name": clara_names}, {"ndim": "2"},
                                     batches=[32] * 8), compression="lz4")
with open(HERE / "clara_256.csv", "w", newline="\n") as csv:
    csv.writelines(",".join(repr(value) for value in walk) + "\n" for walk in CLARA)

# Each case: its rows' changes (a row's position 1 set to a value, a row entirely NaN, a row 30 values of 0.5).
NAN, INF = float("nan"), float("inf")
CASES = {
    "rows": {18: (1, NAN), 33: (1, INF)},             # unsampled rows of batches 1 and 2: the lower is refused
    "union": {52: (1, -INF), 20: (1, NAN)},           # the sampled row 52 is refused before the lower 20
    "nan": {25: "entirely NaN", 45: (1, INF)},        # Interpolate: row 25 is refused, before the infinite 45
    "band": {40: "30 values", 60: "30 values"},       # band 2: the first shortest row (0) and the first longest (40)
}
for case, changes in CASES.items():
    rows = walks(64, 8, 3)
    for row, change in changes.items():
        if change == "entirely NaN":
            rows[row] = [NAN] * 8
        elif change == "30 values":
            rows[row] = [0.5] * 30
        else:
            rows[row][change[0]] = change[1]
    verdict = table({"data": pa.array(rows, f64)}, batches=[16] * 4)
    write(f"clara_verdict_{case}.arrow", verdict, compression="lz4")
    if len(sys.argv) == 3 and sys.argv[1] == "--parquet-twins":
        pq.write_table(verdict, str(Path(sys.argv[2]) / f"clara_verdict_{case}.parquet"), row_group_size=16)
cut = walks(64, 8, 3)
cut[41] = cut[41][:7]
write("clara_verdict_ndim.arrow", table({"data": pa.array(cut, f64)}, {"ndim": "2"}, batches=[16] * 4),
      compression="lz4")

# The stream's scratch fixtures (seeds 11 and 12, 13; row i of clara_ragged has 6 + i % 5 steps).
LONG_LATE = walks(2048, 8, 11)
LONG_LATE[1500] = walks(1, 1500, 12)[0]
write("clara_long_late.arrow", table({"data": pa.array(LONG_LATE, f64)}, batches=[64] * 32), compression="lz4")
RAGGED = [walk[: 6 + i % 5] for i, walk in enumerate(walks(512, 10, 13))]
write("clara_ragged.arrow", table({"data": pa.array(RAGGED, f64)}, batches=[32] * 16))
