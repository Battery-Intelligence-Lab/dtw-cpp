#!/bin/sh
# ctests.sh <base|cand>, in WSL: in that tree, serial ctest of cpp_conformance (labels and medoids digit-identical,
# scores within 1e-12 relative, against the tracked reference), unit_test_dtw_kernel_lanes and unit_test_one_batch_pam
# (both compare the lanes with the per-pair route through the route bound), each test's Catch2 summary shown; whether
# test_codegen_no_calls is registered; then cpp_conformance with DTWC_CONFORMANCE_REGEN=1 and the regenerated
# reference diffed against b3b3a57e's (clust/conformance_reference.b3b3a57e.txt), which is restored afterwards (a
# regenerating run fails by design: "This run rewrote the pinned oracle and verified nothing"; ctest exit 8).
# TMPDIR per tree; taskset -c 0-7 and 8 OpenMP threads (the machine is shared).
set -u
V=$1
HERE=$(cd "$(dirname "$0")" && pwd)
S=$HOME/xg/repo; [ "$V" = cand ] && S=$HOME/xg/repo-cand
T=$S/build-$V
export TMPDIR=$HOME/xg/tmp-$V OMP_NUM_THREADS=8
mkdir -p "$TMPDIR"
cd "$T" || exit 1
echo "== $V: $(ctest -N | tail -1); test_codegen_no_calls registered: $(ctest -N | grep -c test_codegen_no_calls)"
taskset -c 0-7 ctest -R '^(cpp_conformance|unit_test_dtw_kernel_lanes|unit_test_one_batch_pam)$' -j1 -V > "$HERE/ctest-$V.log" 2>&1
echo "ctest exit $?"
grep -E "(All tests passed|test cases:|assertions:).*|Test +#[0-9]+: .*(Passed|Failed|\*\*\*)|tests passed, .* failed" \
  "$HERE/ctest-$V.log" | sed -E 's/^[0-9]+: //'
REF=$S/tests/conformance/conformance_reference.txt
DTWC_CONFORMANCE_REGEN=1 taskset -c 0-7 ctest -R '^cpp_conformance$' -j1 > "$HERE/ctest-$V-regen.log" 2>&1
echo "regen ctest exit $?; regenerated reference vs b3b3a57e's: $(diff --strip-trailing-cr "$REF" \
  "$HERE/conformance_reference.b3b3a57e.txt" | grep -c '^[<>]') differing lines"
diff --strip-trailing-cr "$REF" "$HERE/conformance_reference.b3b3a57e.txt"
cp "$HERE/conformance_reference.b3b3a57e.txt" "$REF"
