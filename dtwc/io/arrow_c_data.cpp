/**
 * @file arrow_c_data.cpp
 * @brief Implementation of Arrow C Data interface ingest (see arrow_c_data.hpp).
 *
 * @author Claude Opus 4.8
 * @date 09 Jul 2026
 */

#include "arrow_c_data.hpp"

#include "../error.hpp"
#include "nanoarrow/nanoarrow.h"

#include <string>
#include <vector>

namespace dtwc::io {

namespace {

/// RAII guard so an ArrowArrayView is reset on every exit path (including throw).
struct ViewGuard {
  ArrowArrayView *v;
  ~ViewGuard() { ArrowArrayViewReset(v); }
};

[[noreturn]] void fail(const std::string &msg) { throw dtwc::InvalidInput(msg); }

bool is_floating(ArrowType t) { return t == NANOARROW_TYPE_DOUBLE || t == NANOARROW_TYPE_FLOAT; }
bool is_list(ArrowType t) { return t == NANOARROW_TYPE_LIST || t == NANOARROW_TYPE_LARGE_LIST; }
bool is_string(ArrowType t)
{
  return t == NANOARROW_TYPE_STRING || t == NANOARROW_TYPE_LARGE_STRING;
}

} // namespace

Data data_from_arrow(const ArrowSchema *schema, const ArrowArray *array,
                     std::vector<std::string> names)
{
  if (schema == nullptr || array == nullptr)
    fail("data_from_arrow: null Arrow schema/array pointer.");

  ArrowError error;
  ArrowArrayView view;
  if (ArrowArrayViewInitFromSchema(&view, schema, &error) != NANOARROW_OK)
    fail(std::string("data_from_arrow: unsupported Arrow schema: ") + ArrowErrorMessage(&error));
  ViewGuard guard{ &view };

  if (ArrowArrayViewSetArray(&view, array, &error) != NANOARROW_OK)
    fail(std::string("data_from_arrow: array does not match schema: ") + ArrowErrorMessage(&error));

  // Resolve the list-of-floating view that carries the series, and (optionally) a
  // utf8 view that carries names. Accept either a bare list column or a struct
  // column with one list child (+ an optional string child), matching what the
  // repo's Arrow-IPC writer produces (name: utf8, data: large_list<float64>).
  const ArrowArrayView *list_view = nullptr;
  const ArrowArrayView *name_view = nullptr;

  if (is_list(view.storage_type)) {
    list_view = &view;
  } else if (view.storage_type == NANOARROW_TYPE_STRUCT) {
    for (int64_t c = 0; c < view.n_children; ++c) {
      const ArrowArrayView *child = view.children[c];
      if (list_view == nullptr && is_list(child->storage_type))
        list_view = child;
      else if (name_view == nullptr && is_string(child->storage_type))
        name_view = child;
    }
    if (list_view == nullptr)
      fail("data_from_arrow: struct column has no list<floating> child to read series from.");
  } else {
    fail("data_from_arrow: expected a list/large_list of floating point, or a struct "
         "containing one; got an unsupported top-level Arrow type.");
  }

  if (list_view->n_children < 1 || !is_floating(list_view->children[0]->storage_type))
    fail("data_from_arrow: list elements must be float32 or float64.");

  const ArrowArrayView *values = list_view->children[0];
  const int64_t n = list_view->length;

  std::vector<std::vector<data_t>> series;
  series.reserve(static_cast<size_t>(n));

  for (int64_t i = 0; i < n; ++i) {
    if (ArrowArrayViewIsNull(list_view, i))
      fail("data_from_arrow: null series at row " + std::to_string(i) +
           " (drop or fill nulls before clustering).");

    const int64_t start = ArrowArrayViewListChildOffset(list_view, i);
    const int64_t end = ArrowArrayViewListChildOffset(list_view, i + 1);

    std::vector<data_t> s;
    s.reserve(static_cast<size_t>(end - start));
    for (int64_t j = start; j < end; ++j) {
      if (ArrowArrayViewIsNull(values, j))
        fail("data_from_arrow: null value inside series at row " + std::to_string(i) +
             " (drop or fill nulls before clustering).");
      s.push_back(static_cast<data_t>(ArrowArrayViewGetDoubleUnsafe(values, j)));
    }
    series.push_back(std::move(s));
  }

  // Names: caller-supplied wins; else the struct's utf8 child; else series_<i>.
  if (!names.empty()) {
    if (static_cast<int64_t>(names.size()) != n)
      fail("data_from_arrow: names count (" + std::to_string(names.size()) +
           ") does not match series count (" + std::to_string(n) + ").");
  } else if (name_view != nullptr && name_view->length == n) {
    names.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      if (ArrowArrayViewIsNull(name_view, i)) {
        names.emplace_back("series_" + std::to_string(i));
      } else {
        const ArrowStringView sv = ArrowArrayViewGetStringUnsafe(name_view, i);
        names.emplace_back(sv.data, static_cast<size_t>(sv.size_bytes));
      }
    }
  } else {
    names.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      names.emplace_back("series_" + std::to_string(i));
  }

  return Data(std::move(series), std::move(names), 1);
}

Data data_from_arrow_stream(ArrowArrayStream *stream)
{
  if (stream == nullptr || stream->release == nullptr)
    fail("data_from_arrow_stream: null or already-released Arrow stream.");

  // Guarantees stream->release runs on every exit path (success or throw).
  struct StreamGuard {
    ArrowArrayStream *s;
    ~StreamGuard() { if (s->release != nullptr) s->release(s); }
  } sguard{ stream };

  ArrowSchema schema;
  if (stream->get_schema(stream, &schema) != NANOARROW_OK) {
    const char *e = stream->get_last_error != nullptr ? stream->get_last_error(stream) : nullptr;
    fail(std::string("data_from_arrow_stream: get_schema failed: ") + (e ? e : "unknown"));
  }
  struct SchemaGuard {
    ArrowSchema *s;
    ~SchemaGuard() { if (s->release != nullptr) s->release(s); }
  } scguard{ &schema };

  std::vector<std::vector<data_t>> all;
  while (true) {
    ArrowArray batch;
    batch.release = nullptr;
    if (stream->get_next(stream, &batch) != NANOARROW_OK) {
      const char *e = stream->get_last_error != nullptr ? stream->get_last_error(stream) : nullptr;
      fail(std::string("data_from_arrow_stream: get_next failed: ") + (e ? e : "unknown"));
    }
    if (batch.release == nullptr)
      break; // end of stream

    try {
      Data d = data_from_arrow(&schema, &batch, {});
      for (auto &s : d.p_vec)
        all.push_back(std::move(s));
    } catch (...) {
      release_arrow(nullptr, &batch);
      throw;
    }
    release_arrow(nullptr, &batch);
  }

  std::vector<std::string> names;
  names.reserve(all.size());
  for (size_t i = 0; i < all.size(); ++i)
    names.emplace_back("series_" + std::to_string(i));

  return Data(std::move(all), std::move(names), 1);
}

void release_arrow(ArrowSchema *schema, ArrowArray *array) noexcept
{
  if (array != nullptr && array->release != nullptr)
    array->release(array);
  if (schema != nullptr && schema->release != nullptr)
    schema->release(schema);
}

} // namespace dtwc::io
