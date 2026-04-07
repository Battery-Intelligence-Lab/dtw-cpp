#!/usr/bin/env bash
# stress_test_cli.sh — End-to-end stress test for dtwc_cl CLI
# Exercises every method x variant x metric combination, checkpoint round-trip,
# distance matrix reload, banded DTW, SoftDTW sentinel, and edge cases.
set -uo pipefail

# ---- Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$REPO/build/bin/dtwc_cl.exe"
[[ -f "$CLI" ]] || CLI="$REPO/build/bin/dtwc_cl"
DUMMY_DIR="$REPO/data/dummy"
COFFEE_TRAIN="$REPO/data/benchmark/UCRArchive_2018/Coffee/Coffee_TRAIN.tsv"
OUT_BASE="$REPO/build/stress_test_output"

PASS=0; FAIL=0; SKIP=0; TOTAL=0
FAILURES=""

rm -rf "$OUT_BASE"
mkdir -p "$OUT_BASE"

# ---- Helpers ----
run_test() {
    local name="$1"; shift
    local out_dir="$OUT_BASE/$name"
    mkdir -p "$out_dir"
    "$CLI" "$@" --output "$out_dir" --name test -v \
        >"$out_dir/stdout.txt" 2>"$out_dir/stderr.txt"
    echo $?
}

record_pass() { ((PASS++)); ((TOTAL++)); echo "  PASS: $1"; }
record_fail() { ((FAIL++)); ((TOTAL++)); FAILURES+="  FAIL: $1 — $2\n"; echo "  FAIL: $1 — $2"; }
record_skip() { ((SKIP++)); ((TOTAL++)); echo "  SKIP: $1"; }

validate() {
    local name="$1" ec="$2" n_series="$3" k="$4" variant="${5:-standard}"
    local dir="$OUT_BASE/$name"

    # Exit code
    if [[ "$ec" -ne 0 ]]; then
        record_fail "$name" "exit code $ec"
        return
    fi

    # Labels file
    if [[ ! -f "$dir/test_labels.csv" ]]; then
        record_fail "$name" "missing test_labels.csv"
        return
    fi
    local n_labels
    n_labels=$(tail -n +2 "$dir/test_labels.csv" | wc -l | tr -d ' ')
    if [[ "$n_labels" -ne "$n_series" ]]; then
        record_fail "$name" "labels has $n_labels rows, expected $n_series"
        return
    fi

    # Medoids file
    if [[ ! -f "$dir/test_medoids.csv" ]]; then
        record_fail "$name" "missing test_medoids.csv"
        return
    fi
    local n_medoids
    n_medoids=$(tail -n +2 "$dir/test_medoids.csv" | wc -l | tr -d ' ')
    if [[ "$n_medoids" -ne "$k" ]]; then
        record_fail "$name" "medoids has $n_medoids rows, expected $k"
        return
    fi

    # Cost is finite
    local cost
    cost=$(grep "Total cost:" "$dir/stdout.txt" | grep -oE '[-0-9.e+]+' | tail -1)
    if [[ -z "$cost" ]]; then
        record_fail "$name" "no Total cost in output"
        return
    fi
    # Check for nan/inf
    if echo "$cost" | grep -qiE 'nan|inf'; then
        record_fail "$name" "cost is $cost"
        return
    fi

    # Silhouettes (only for k > 1 and non-CLARA methods that produce full dist matrix)
    if [[ "$k" -gt 1 && ! "$name" =~ "clara" ]]; then
        if [[ ! -f "$dir/test_silhouettes.csv" ]]; then
            record_fail "$name" "missing test_silhouettes.csv"
            return
        fi
    fi

    record_pass "$name"
}

# ======================================================================
echo "============================================"
echo "  DTWC++ CLI Stress Test"
echo "============================================"
echo ""

# ---- Phase 1: Smoke tests (dummy data, 25 series, k=3) ----
echo "--- Phase 1: Smoke tests (dummy data) ---"

METHODS=(pam clara kmedoids hierarchical)
VARIANTS=(standard ddtw wdtw adtw softdtw)

for method in "${METHODS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        name="p1_${method}_${variant}"
        # Skip SoftDTW on dummy data with CLARA/kmedoids (5000-point series too slow)
        if [[ "$variant" == "softdtw" && ("$method" == "clara" || "$method" == "kmedoids") ]]; then
            record_skip "$name (softdtw+${method} on dummy too slow)"
            continue
        fi

        extra=()
        [[ "$variant" == "wdtw" ]]    && extra+=(--wdtw-g 0.05)
        [[ "$variant" == "adtw" ]]    && extra+=(--adtw-penalty 1.0)
        [[ "$variant" == "softdtw" ]] && extra+=(--sdtw-gamma 1.0)
        [[ "$method" == "clara" ]]    && extra+=(--seed 42 --sample-size 15 --n-samples 3)
        [[ "$method" == "hierarchical" ]] && extra+=(--linkage average)

        ec=$(run_test "$name" \
            --input "$DUMMY_DIR" --clusters 3 \
            --method "$method" --variant "$variant" \
            --metric l1 --band -1 \
            --skip-rows 1 --skip-cols 1 \
            "${extra[@]}")
        validate "$name" "$ec" 25 3 "$variant"
    done
done

# Metric variation: SquaredL2
name="p1_pam_standard_sqeucl"
ec=$(run_test "$name" \
    --input "$DUMMY_DIR" --clusters 3 --method pam --variant standard \
    --metric squared_euclidean --band -1 --skip-rows 1 --skip-cols 1)
validate "$name" "$ec" 25 3

# Band variation: band=5
name="p1_pam_standard_band5"
ec=$(run_test "$name" \
    --input "$DUMMY_DIR" --clusters 3 --method pam --variant standard \
    --metric l1 --band 5 --skip-rows 1 --skip-cols 1)
validate "$name" "$ec" 25 3

echo ""

# ---- Phase 2: Correctness (Coffee UCR, 28 series, k=2) ----
echo "--- Phase 2: Correctness (Coffee UCR) ---"

if [[ ! -f "$COFFEE_TRAIN" ]]; then
    echo "  Coffee dataset not found at $COFFEE_TRAIN — skipping Phase 2"
    for variant in "${VARIANTS[@]}"; do record_skip "p2_${variant}"; done
else
    for variant in "${VARIANTS[@]}"; do
        name="p2_pam_${variant}"
        extra=()
        [[ "$variant" == "wdtw" ]]    && extra+=(--wdtw-g 0.05)
        [[ "$variant" == "adtw" ]]    && extra+=(--adtw-penalty 1.0)
        [[ "$variant" == "softdtw" ]] && extra+=(--sdtw-gamma 1.0)

        ec=$(run_test "$name" \
            --input "$COFFEE_TRAIN" --clusters 2 --method pam \
            --variant "$variant" --skip-cols 1 \
            "${extra[@]}")
        validate "$name" "$ec" 28 2 "$variant"
    done

    # ARI check for standard variant
    if [[ -f "$OUT_BASE/p2_pam_standard/test_labels.csv" ]]; then
        # Extract ground truth (column 1 of TSV, tab-separated)
        cut -f1 "$COFFEE_TRAIN" > "$OUT_BASE/coffee_gt.txt"
        # Compare using Python if available
        if command -v python3 &>/dev/null || command -v python &>/dev/null; then
            PY=$(command -v python3 || command -v python)
            ari=$($PY -c "
import csv, sys
pred = {}
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        pred[int(row['name'])] = int(row['cluster'])
gt = [int(float(x)) for x in open(sys.argv[2]).read().split()]
n = len(gt)
pairs_same_both = sum(1 for i in range(n) for j in range(i+1,n)
                      if (gt[i]==gt[j]) == (pred[i+1]==pred[j+1]))
ri = pairs_same_both / (n*(n-1)//2)
print(f'{ri:.4f}')
" "$OUT_BASE/p2_pam_standard/test_labels.csv" "$OUT_BASE/coffee_gt.txt" 2>/dev/null)
            if [[ -n "$ari" ]]; then
                echo "  Rand Index (standard DTW vs ground truth): $ari"
                if (( $(echo "$ari > 0.5" | bc -l 2>/dev/null || echo 1) )); then
                    record_pass "p2_ari_check"
                else
                    record_fail "p2_ari_check" "Rand Index $ari <= 0.5"
                fi
            else
                echo "  (ARI computation failed, skipping)"
                record_skip "p2_ari_check"
            fi
        else
            echo "  (Python not available, skipping ARI check)"
            record_skip "p2_ari_check"
        fi
    fi
fi

echo ""

# ---- Phase 3: Stress tests ----
echo "--- Phase 3: Stress tests ---"

# 3a: Checkpoint round-trip
if [[ -f "$COFFEE_TRAIN" ]]; then
    name="p3_ckpt_run1"
    ec=$(run_test "$name" \
        --input "$COFFEE_TRAIN" --clusters 2 --method pam --skip-cols 1 \
        --checkpoint "$OUT_BASE/$name/ckpt")
    validate "$name" "$ec" 28 2

    if [[ "$ec" -eq 0 && -d "$OUT_BASE/$name/ckpt" ]]; then
        name="p3_ckpt_run2"
        ec=$(run_test "$name" \
            --input "$COFFEE_TRAIN" --clusters 2 --method pam --skip-cols 1 \
            --checkpoint "$OUT_BASE/p3_ckpt_run1/ckpt")
        validate "$name" "$ec" 28 2

        # Verify labels match
        if diff -q "$OUT_BASE/p3_ckpt_run1/test_labels.csv" \
                   "$OUT_BASE/p3_ckpt_run2/test_labels.csv" >/dev/null 2>&1; then
            record_pass "p3_ckpt_labels_match"
        else
            record_fail "p3_ckpt_labels_match" "checkpoint round-trip labels differ"
        fi
    fi

    # 3b: Precomputed distance matrix reload
    if [[ -f "$OUT_BASE/p3_ckpt_run1/test_distance_matrix.csv" ]]; then
        name="p3_distmat_reload"
        ec=$(run_test "$name" \
            --input "$COFFEE_TRAIN" --clusters 2 --method pam --skip-cols 1 \
            --dist-matrix "$OUT_BASE/p3_ckpt_run1/test_distance_matrix.csv")
        validate "$name" "$ec" 28 2

        if diff -q "$OUT_BASE/p3_ckpt_run1/test_labels.csv" \
                   "$OUT_BASE/$name/test_labels.csv" >/dev/null 2>&1; then
            record_pass "p3_distmat_labels_match"
        else
            record_fail "p3_distmat_labels_match" "dist-matrix reload labels differ"
        fi
    fi

    # 3c: Banded DTW
    for band in 10 50 200; do
        name="p3_band_${band}"
        ec=$(run_test "$name" \
            --input "$COFFEE_TRAIN" --clusters 2 --method pam --skip-cols 1 \
            --band "$band")
        validate "$name" "$ec" 28 2
    done

    # 3d: SoftDTW sentinel (verify no NaN in distance matrix)
    name="p3_softdtw_sentinel"
    ec=$(run_test "$name" \
        --input "$COFFEE_TRAIN" --clusters 2 --method pam --skip-cols 1 \
        --variant softdtw --sdtw-gamma 1.0)
    validate "$name" "$ec" 28 2 "softdtw"
    if [[ -f "$OUT_BASE/$name/test_distance_matrix.csv" ]]; then
        if grep -qiE 'nan|inf' "$OUT_BASE/$name/test_distance_matrix.csv"; then
            record_fail "p3_softdtw_matrix_clean" "distance matrix contains nan/inf"
        else
            record_pass "p3_softdtw_matrix_clean"
        fi
    fi

    # 3e: k=1 (single cluster)
    name="p3_k1"
    ec=$(run_test "$name" \
        --input "$COFFEE_TRAIN" --clusters 1 --method pam --skip-cols 1)
    if [[ "$ec" -eq 0 ]]; then
        # All labels should be 0
        bad=$(tail -n +2 "$OUT_BASE/$name/test_labels.csv" | cut -d',' -f2 | grep -v '^0$' | wc -l | tr -d ' ')
        if [[ "$bad" -eq 0 ]]; then
            record_pass "p3_k1"
        else
            record_fail "p3_k1" "$bad labels are not 0"
        fi
    else
        record_fail "p3_k1" "exit code $ec"
    fi

    # 3f: k=N (each point its own cluster)
    name="p3_kN"
    ec=$(run_test "$name" \
        --input "$COFFEE_TRAIN" --clusters 28 --method pam --skip-cols 1)
    validate "$name" "$ec" 28 28

    # 3g: ADTW banded (exercises early-abandon fallback)
    name="p3_adtw_banded"
    ec=$(run_test "$name" \
        --input "$COFFEE_TRAIN" --clusters 2 --method pam --skip-cols 1 \
        --variant adtw --adtw-penalty 2.0 --band 10)
    validate "$name" "$ec" 28 2 "adtw"

    # 3h: WDTW g sweep
    for g in 0.01 0.05 0.5; do
        name="p3_wdtw_g${g}"
        ec=$(run_test "$name" \
            --input "$COFFEE_TRAIN" --clusters 2 --method pam --skip-cols 1 \
            --variant wdtw --wdtw-g "$g")
        validate "$name" "$ec" 28 2 "wdtw"
    done
else
    echo "  Coffee dataset not found — skipping Phase 3"
fi

echo ""

# ---- Summary ----
echo "============================================"
echo "  STRESS TEST SUMMARY"
echo "============================================"
echo "  PASS:  $PASS"
echo "  FAIL:  $FAIL"
echo "  SKIP:  $SKIP"
echo "  TOTAL: $TOTAL"
echo "============================================"

if [[ "$FAIL" -gt 0 ]]; then
    echo ""
    echo "FAILURES:"
    echo -e "$FAILURES"
    exit 1
fi

echo ""
echo "All tests passed."
exit 0
