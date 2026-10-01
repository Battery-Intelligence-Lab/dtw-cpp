#!/usr/bin/env bash
# W7d: dtwc_cl -m pam on data/dummy for every variant, metric and missing-data
# strategy the CLI offers (float64 and float32, unbanded and band 4300 -- the
# smallest feasible band on data/dummy is 4257), keeping every output: the
# distance-matrix, labels, medoids and silhouette CSVs, stdout without the
# timing/path lines, stderr and the exit code.
# Usage: w7d_cli_runs.sh <dtwc_cl.exe> <out_dir>
set -u
BIN="$1"
OUT="$2"
DATA=C:/D/git/wt/W7d/data/dummy
rm -rf "$OUT"; mkdir -p "$OUT"
run() {
  local name="$1"; shift
  local dir="$OUT/$name"
  mkdir -p "$dir/files"
  "$BIN" -i "$DATA" --skip-rows 1 --skip-cols 1 -k 3 -m pam -o "$dir/files" "$@" \
    > "$dir/stdout.raw" 2> "$dir/stderr.txt"
  echo $? > "$dir/exit.txt"
  grep -v "Time:\|Output:\|\[.*s\]\|min:sec" "$dir/stdout.raw" > "$dir/stdout.txt"
  rm -f "$dir/stdout.raw"
}
run std_l1
run std_sq           --metric squared_euclidean
run std_l1_b4300     --band 4300
run std_sq_b4300     --band 4300 --metric squared_euclidean
run ddtw_l1          --variant ddtw
run ddtw_sq          --variant ddtw --metric squared_euclidean
run ddtw_b4300       --variant ddtw --band 4300
run wdtw             --variant wdtw --wdtw-g 0.1
run wdtw_b4300       --variant wdtw --band 4300
run adtw             --variant adtw
run adtw_b4300       --variant adtw --band 4300
OMP_NUM_THREADS=6 run softdtw --variant softdtw
run msm              --variant msm
run twe              --variant twe
run zero_l1          --missing-strategy zero_cost
run zero_sq          --missing-strategy zero_cost --metric squared_euclidean
run zero_b4300       --missing-strategy zero_cost --band 4300
run arow_l1          --missing-strategy arow
run arow_sq          --missing-strategy arow --metric squared_euclidean
run arow_b4300       --missing-strategy arow --band 4300
run interp_l1        --missing-strategy interpolate
run interp_sq        --missing-strategy interpolate --metric squared_euclidean
run f32_std          --dtype float32
run f32_std_sq_b4300 --dtype float32 --metric squared_euclidean --band 4300
run f32_ddtw         --dtype float32 --variant ddtw
run f32_wdtw_b4300   --dtype float32 --variant wdtw --band 4300
run f32_adtw         --dtype float32 --variant adtw
run f32_msm          --dtype float32 --variant msm
run f32_twe          --dtype float32 --variant twe
run f32_zero_sq      --dtype float32 --missing-strategy zero_cost --metric squared_euclidean
run f32_arow         --dtype float32 --missing-strategy arow
