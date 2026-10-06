"""v_skew2.hpp / v_fmin_skew2.hpp: dtw_kernel_linear computes two columns (j, j+1) per pass of the
inner loop, so two dependency chains run side by side; each cell gets the same three neighbours in
the same roles as the shipped kernel, so a Cell's result is unchanged bit for bit."""
import re, sys
src = open('/Users/engs2321/git/dtw-cpp/dtwc/core/dtw_kernel.hpp').read()
start = src.index('template <typename T, typename Cost, typename Cell>\nT dtw_kernel_linear(')
end = src.index('// ===========================================================================\n// Kernel 2')
new = '''template <typename T, typename Cost, typename Cell>
T dtw_kernel_linear(std::size_t n_short, std::size_t n_long,
                    Cost cost_in, Cell cell, T early_abandon = T(-1))
{
  const Cost cost = cost_in; // in registers: see the file comment
  constexpr T maxValue = std::numeric_limits<T>::max();
  if (n_short == 0 || n_long == 0) return maxValue;

  thread_local static std::vector<T> short_buf;
  short_buf.resize(n_short);
  T *short_side = short_buf.data(); // hoisted out of the loops

  short_side[0] = cell.seed(cost(0, 0), 0, 0);
  for (std::size_t i = 1; i < n_short; ++i)
    short_side[i] = cell.combine(maxValue, short_side[i - 1], maxValue,
                                 cost(i, 0), i, 0);

  const bool do_early_abandon = (early_abandon >= T(0));

  std::size_t j = 1;
  // Two columns per pass: column a = j, column b = j + 1. Cell (i, b) takes
  // diag = dp[i-1, j] (the previous a), up = dp[i, j] (this a), left = dp[i-1, j+1].
  for (; j + 1 < n_long; j += 2) {
    T diag_a = short_side[0];
    T left_a = cell.combine(maxValue, maxValue, short_side[0], cost(0, j), 0, j);
    T left_b = cell.combine(maxValue, maxValue, left_a, cost(0, j + 1), 0, j + 1);
    short_side[0] = left_b;
    T min_a = do_early_abandon ? left_a : T(0);
    T min_b = do_early_abandon ? left_b : T(0);
    for (std::size_t i = 1; i < n_short; ++i) {
      const T old_up = short_side[i]; // dp[i, j-1]
      const T a = cell.combine(diag_a, old_up, left_a, cost(i, j), i, j);
      const T b = cell.combine(left_a, a, left_b, cost(i, j + 1), i, j + 1);
      diag_a = old_up;
      left_a = a;
      left_b = b;
      short_side[i] = b;
      if (do_early_abandon) {
        min_a = std::min(min_a, a);
        min_b = std::min(min_b, b);
      }
    }
    if (do_early_abandon && (min_a > early_abandon || min_b > early_abandon)) return maxValue;
  }
  for (; j < n_long; ++j) {
    T diag = short_side[0];
    T left = cell.combine(maxValue, maxValue, short_side[0], cost(0, j), 0, j);
    short_side[0] = left;
    T row_min = do_early_abandon ? left : T(0);
    for (std::size_t i = 1; i < n_short; ++i) {
      const T old_up = short_side[i];
      left = cell.combine(diag, old_up, left, cost(i, j), i, j);
      diag = old_up;
      short_side[i] = left;
      if (do_early_abandon) row_min = std::min(row_min, left);
    }
    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return short_side[n_short - 1];
}

'''
out = src[:start] + new + src[end:]
for name, fmin in (('v_skew2', False), ('v_fmin_skew2', True)):
    o = out.replace('namespace dtwc::core {', f'namespace {name} {{').replace('} // namespace dtwc::core', f'}} // namespace {name}')
    if fmin:
        o = o.replace('return std::min(std::min(diag, up), left) + cost;', 'return std::fmin(std::fmin(diag, up), left) + cost;')
        o = o.replace('return std::min(std::min(diag, up + penalty), left + penalty) + cost;', 'return std::fmin(std::fmin(diag, up + penalty), left + penalty) + cost;')
    open(name + '.hpp', 'w').write(o)
print('ok')
