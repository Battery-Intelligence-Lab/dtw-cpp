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

Every value is exact in Float32. Run: uv run --no-project --with pyarrow==23.0.1 python make_ipc_fixtures.py
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

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
