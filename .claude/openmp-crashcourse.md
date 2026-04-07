# OpenMP Crash Course for DTWC++

Minimum you need to know to read and write the parallel code in this repo.

---

## 1. The basics

OpenMP is a pragma-based parallelism API. You sprinkle `#pragma omp` directives on existing serial code. The compiler (MSVC `/openmp`, GCC/Clang `-fopenmp`) handles thread creation.

```cpp
#ifdef _OPENMP
#include <omp.h>
#endif
```

Everything is opt-in. If OpenMP isn't enabled, the pragmas are ignored and code runs serial.

---

## 2. parallel for — the workhorse

```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    result[i] = expensive(i);  // each iteration is independent
}
```

- Spawns a team of threads, splits iterations across them.
- **Requirement**: iterations must be independent (no data races).
- Loop variable `i` is automatically private.
- Variables declared outside the loop are **shared** by default.

### schedule clause

Controls how iterations are divided:

| Schedule | Behaviour | When to use |
|----------|-----------|-------------|
| `static` | Equal chunks, assigned at compile time | Uniform work per iteration |
| `dynamic` | Grab next chunk when idle | Variable work per iteration |
| `dynamic,1` | One iteration at a time | Highly variable work |
| `guided` | Decreasing chunk sizes | Large loops, moderate variance |

```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < N; ++i) { ... }
```

In this repo, DTW pair computation uses `schedule(dynamic)` because pair (i,j) cost varies with series length.

---

## 3. Variable scoping

| Clause | Meaning | Default? |
|--------|---------|----------|
| `shared(x)` | All threads see the same `x` | Yes, for variables declared before the parallel region |
| `private(x)` | Each thread gets its own uninitialised copy | No |
| `firstprivate(x)` | Each thread gets a copy initialised from the original | No |
| `reduction(+:sum)` | Each thread has a private copy; merged with `+` at the end | No |

```cpp
double total = 0.0;
#pragma omp parallel for reduction(+:total)
for (int i = 0; i < N; ++i) {
    total += data[i];  // no race — each thread has its own total
}
// total now holds the global sum
```

### thread_local vs private

- `thread_local static std::vector<T> buf;` — persists across calls, one per OS thread. Used in this repo for DTW rolling buffers to avoid repeated allocation.
- `private(buf)` — fresh uninitialised copy per parallel region entry.

This repo prefers `thread_local` for hot-path buffers (warping functions).

---

## 4. critical — mutual exclusion

```cpp
#pragma omp critical
{
    shared_map[key] = value;  // only one thread at a time
}
```

- Named form: `#pragma omp critical(name)` — different names = different locks.
- **Expensive.** Serialises the block. Use only for infrequent operations.
- This repo uses it for the lazy-init double-check pattern in `distByInd`.

### Double-check pattern

```cpp
if (need_init) {                    // fast path: no lock
    #pragma omp critical(init_lock)
    {
        if (need_init) {            // re-check under lock
            do_init();
        }
    }
}
```

---

## 5. atomic — lightweight single-variable updates

```cpp
#pragma omp atomic
counter++;
```

- Faster than `critical` for simple read-modify-write on one variable.
- Works for `++`, `--`, `+=`, `-=`, `*=`, etc.
- Read variant: `#pragma omp atomic read` / `write`.

```cpp
double global_min = INF;

#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    double local = compute(i);
    #pragma omp atomic
    if (local < global_min) global_min = local;  // WRONG — atomic doesn't work on if
}
```

For min/max, use `reduction` instead:

```cpp
#pragma omp parallel for reduction(min:global_min)
```

---

## 6. barrier, single, master

```cpp
#pragma omp parallel
{
    setup();           // all threads run this

    #pragma omp barrier  // wait for everyone

    #pragma omp single   // exactly one thread runs this (implicit barrier after)
    {
        merge_results();
    }

    // all threads continue here after the single block
    postprocess();
}
```

- `barrier` — all threads wait until everyone reaches this point.
- `single` — one thread executes, others wait. Has implicit barrier at end.
- `master` — only thread 0 executes, no implicit barrier (others skip it and continue).
- `#pragma omp single nowait` — removes the implicit barrier.

---

## 7. parallel regions vs parallel for

```cpp
// parallel for — shorthand for a parallel region containing a for
#pragma omp parallel for
for (int i = 0; i < N; ++i) { ... }

// equivalent expanded form:
#pragma omp parallel
{
    #pragma omp for
    for (int i = 0; i < N; ++i) { ... }
}
```

The expanded form is useful when you need setup/teardown around the loop:

```cpp
#pragma omp parallel
{
    thread_local_setup();
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < N; ++i) { compute(i); }
    thread_local_teardown();
}
```

---

## 8. Nested parallelism and num_threads

```cpp
#pragma omp parallel for num_threads(4)
for (int i = 0; i < N; ++i) { ... }
```

- `num_threads(N)` — override default thread count.
- `omp_get_num_threads()` — how many threads are in the current team.
- `omp_get_thread_num()` — which thread am I (0-based).
- `omp_get_max_threads()` — max threads that would be used.

Nested parallelism (parallel inside parallel) is disabled by default. Don't use it unless you know what you're doing.

---

## 9. Common pitfalls

### Race on shared container

```cpp
// BUG: push_back is not thread-safe
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    results.push_back(compute(i));  // data race!
}

// FIX: pre-allocate and index
results.resize(N);
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    results[i] = compute(i);       // disjoint indices — safe
}
```

### False sharing

```cpp
// SLOW: adjacent elements in different cache lines get bounced between cores
int counts[num_threads];  // 4 bytes apart = same cache line
#pragma omp parallel
{
    counts[omp_get_thread_num()]++;  // false sharing!
}

// FIX: pad to cache line size, or use reduction
```

### Iterator loops

OpenMP requires random-access iterators (or plain `int`/`size_t` in C++17):

```cpp
// Won't compile with OpenMP:
for (auto it = map.begin(); it != map.end(); ++it) { ... }

// Works:
std::vector<Key> keys(map.size());
std::copy(map.begin(), map.end(), keys.begin());
#pragma omp parallel for
for (int i = 0; i < (int)keys.size(); ++i) { ... }
```

### Exceptions in parallel regions

If a thread throws and it's not caught within the parallel region, behaviour is **undefined** (usually a crash). Always catch inside:

```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    try {
        risky(i);
    } catch (...) {
        // handle or store for later
    }
}
```

---

## 10. Patterns used in this repo

### Disjoint pair fill (distance matrix)

```cpp
#pragma omp parallel for schedule(dynamic) collapse(2)
for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
        distMat.set(i, j, dtw(series[i], series[j]));
```

- `collapse(2)` — treats the nested loop as a single iteration space for better load balance.
- Each (i,j) pair is unique — no lock needed on `set()`.
- `dtw()` uses `thread_local` buffers — no allocation contention.

### thread_local rolling buffer

```cpp
template <typename data_t>
data_t dtwFull_L(const data_t* x, size_t nx, const data_t* y, size_t ny) {
    thread_local static std::vector<data_t> col;
    col.resize(m_short);  // reuses existing capacity across calls
    // ... fill col ...
}
```

Each OS thread gets its own `col`. No lock needed. Capacity grows monotonically — no repeated allocations after warmup.

### run() abstraction

This repo wraps parallel for in a helper:

```cpp
// parallelisation.hpp
template <typename Fn>
void run(Fn&& task, size_t N) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N; ++i)
        task(i);
}
```

Callers just pass a lambda. The parallelism is invisible at the call site.

---

## Quick reference card

| Pragma | Purpose |
|--------|---------|
| `parallel for` | Parallel loop |
| `parallel for reduction(+:x)` | Parallel loop with sum reduction |
| `parallel for schedule(dynamic)` | Dynamic load balancing |
| `critical` / `critical(name)` | Mutual exclusion block |
| `atomic` | Single-variable atomic update |
| `barrier` | Wait for all threads |
| `single` | One thread executes (+ barrier) |
| `parallel for collapse(N)` | Flatten N nested loops |
| `parallel for num_threads(K)` | Limit thread count |
| `parallel for private(x)` | Per-thread copy of x |
| `parallel for firstprivate(x)` | Per-thread copy, initialised from original |
