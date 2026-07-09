/**
 * @file arrow_c_data.hpp
 * @brief Zero-dependency ingest from the Arrow C Data interface into dtwc::Data.
 *
 * @details Consumes an (ArrowSchema, ArrowArray) pair — the two C structs behind
 * the Arrow PyCapsule protocol `__arrow_c_array__`, exported by polars, DuckDB,
 * pyarrow, pandas, and anything else implementing the interface. Reading is done
 * with the vendored, dependency-free nanoarrow (dtwc/extern/nanoarrow), so no
 * Arrow C++ / pyarrow is required at build or run time.
 *
 * Accepted layouts (each becomes univariate series, ndim = 1):
 *   - list<floating> / large_list<floating>: one series per list element.
 *   - struct< name: utf8/large_utf8 (optional), <list-of-floating> >: the list
 *     child supplies the series, the utf8 child (if present) supplies names.
 *
 * Values are copied once into owning storage (`Data::p_vec`); the borrowed Arrow
 * buffers are never aliased, so the caller may release the ArrowArray as soon as
 * this returns. "Zero-copy" here means the buffer-direct read that avoids both a
 * pyarrow dependency and the per-element Python-object materialisation of the old
 * `list(row) for row in X` path — not that the doubles are aliased in place.
 *
 * The two structs are BORROWED: this function reads them and does NOT call their
 * `release` callbacks. Ownership/release is the caller's responsibility (the
 * Python binding releases after this returns).
 *
 * @author Claude Opus 4.8
 * @date 09 Jul 2026
 */

#pragma once

#include "../Data.hpp"

#include <string>
#include <vector>

// Forward declarations of the Arrow C Data interface structs — keeps the heavy
// nanoarrow header out of this public surface. The full definitions (from
// nanoarrow.h, guarded by ARROW_C_DATA_INTERFACE) are ABI-compatible with these.
extern "C" {
struct ArrowSchema;
struct ArrowArray;
struct ArrowArrayStream;
}

namespace dtwc::io {

/**
 * @brief Build a Data object from an Arrow C Data interface (schema, array) pair.
 *
 * @param schema  Borrowed ArrowSchema* describing the array's type.
 * @param array   Borrowed ArrowArray* holding the buffers.
 * @param names   Optional series names. If empty, names default to the utf8 child
 *                (struct layout) or to "series_<i>". If non-empty, its size must
 *                equal the number of series.
 * @return Data (float64, ndim = 1) owning a copy of every series.
 * @throws dtwc::InvalidInput on unsupported schema, null elements, or a
 *         names-count mismatch.
 */
Data data_from_arrow(const ArrowSchema *schema, const ArrowArray *array,
                     std::vector<std::string> names = {});

/**
 * @brief Build a Data object from an Arrow C stream (the `__arrow_c_stream__`
 *        PyCapsule protocol, as exported by polars and pandas).
 *
 * Consumes every batch, concatenating the list elements into one series set
 * (names default to "series_<global i>"). The stream is FULLY CONSUMED and its
 * release callback is called before returning (including on error), so the
 * caller must not touch it afterwards.
 *
 * @param stream Borrowed-then-consumed ArrowArrayStream*.
 * @throws dtwc::InvalidInput on stream error, unsupported schema, or nulls.
 */
Data data_from_arrow_stream(ArrowArrayStream *stream);

/**
 * @brief Release a borrowed Arrow schema/array pair (calls their C release
 *        callbacks if still live). Safe to call with nulls. Lets a caller that
 *        holds only the forward-declared pointers (e.g. the Python binding, which
 *        must not pull in nanoarrow) hand ownership back after data_from_arrow.
 */
void release_arrow(ArrowSchema *schema, ArrowArray *array) noexcept;

} // namespace dtwc::io
