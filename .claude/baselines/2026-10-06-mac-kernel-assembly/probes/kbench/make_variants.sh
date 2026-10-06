#!/bin/sh
# Recreates the scratch kernel variants from the shipped header (read only). Each copy gets its own
# namespace; the kernels' internal calls are qualified so ADL on dtwc::core Cost types cannot reach the
# shipped kernels. v_skew2 / v_fmin_skew2 come from make_skew.py (run after this).
set -e
cd "$(dirname "$0")"
K=/Users/engs2321/git/dtw-cpp/dtwc/core/dtw_kernel.hpp
FMIN='s/return std::min(std::min(diag, up), left) + cost;/return std::fmin(std::fmin(diag, up), left) + cost;/;s/return std::min(std::min(diag, up + penalty), left + penalty) + cost;/return std::fmin(std::fmin(diag, up + penalty), left + penalty) + cost;/'
W16='s|inline constexpr std::size_t dtw_lanes = 64 / sizeof(T);|inline constexpr std::size_t dtw_lanes = 128 / sizeof(T);|'
W32='s|inline constexpr std::size_t dtw_lanes = 64 / sizeof(T);|inline constexpr std::size_t dtw_lanes = 256 / sizeof(T);|'
mk() { # name, extra sed program
  sed -e "s/^namespace dtwc::core {/namespace $1 {/" -e "s|^} // namespace dtwc::core|} // namespace $1|" \
      -e "s/return dtw_kernel_linear<T>(/return ::$1::dtw_kernel_linear<T>(/g" \
      -e "s/return dtw_kernel_banded<T>(/return ::$1::dtw_kernel_banded<T>(/g" -e "$2" $K > $1.hpp
}
mk v_fmin "$FMIN"
mk v_w16 "$W16"
mk v_w32 "$W32"
mk v_fmin_w16 "$FMIN;$W16"
uv run --no-project python make_skew.py
for f in v_skew2 v_fmin_skew2; do
  sed -i '' -e "s/return dtw_kernel_linear<T>(/return ::$f::dtw_kernel_linear<T>(/g" -e "s/return dtw_kernel_banded<T>(/return ::$f::dtw_kernel_banded<T>(/g" $f.hpp
done
echo variants made
