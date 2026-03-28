# Lemire O(n) Envelope Algorithm — Future Optimization

## Current: O(n * band) sliding window scan
The current `compute_envelopes()` scans `[p-band, p+band]` for each position p.
For band=50, n=8000: 800K comparisons. Cache-friendly (contiguous reads).
Computed once per series, reused O(N) times. Negligible vs O(N^2 * n) DTW.

**Keep current approach.** Band is small by definition.

## Future: Lemire's O(n) streaming min/max (if band ever grows large)

Reference: Daniel Lemire, "Streaming Maximum-Minimum Filter Using No More
than Three Comparisons per Element," Nordic J. Computing, 13(4), 2006.

### How it works

Maintains two monotone deques (one for max, one for min):

1. **Max deque**: stores indices in decreasing order of series values.
   - Back: newest index. Front: index of current window maximum.
   - When adding element i: pop all back elements with `series[back] <= series[i]`
     (they can never be the max for any future window), then push i.
   - When the window slides past front: pop front.

2. **Min deque**: same but reversed comparison.

Each element enters and leaves each deque exactly once → O(n) total, O(1) amortized per position.

### Implementation notes for DTWC++

- **Don't use `std::deque`** — it's a linked structure with pointer chasing. Use a flat array
  with head/tail indices as a circular buffer.
- **Buffer size**: at most `2*band+1` elements live simultaneously (window width).
- **thread_local buffers**: reuse across calls to avoid allocation.
- **Caution**: the flat circular buffer approach requires modular arithmetic for index wrapping.
  An alternative is a simple array with head/tail that resets when empty.

### When to consider switching

Only if profiling shows envelope computation is a bottleneck, which would require:
- Very large band (band > 1000) AND
- Many series (N > 100K, so envelope is computed 100K times)

For the current target (band=50, n=8K), the naive scan takes ~50 microseconds per series.
Even for N=100K series, that's 5 seconds total — negligible vs the O(N^2) DTW pairs.