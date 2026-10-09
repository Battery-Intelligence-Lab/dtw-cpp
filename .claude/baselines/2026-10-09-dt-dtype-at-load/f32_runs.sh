#!/usr/bin/env bash
# Unit DT: dtwc_cl --dtype float32 over each reader (CSV folder, batch CSV, Parquet list and
# rows layouts, Arrow IPC) and kmedoids / pam / clara (fixed seed), plus float64 controls and
# two verbose runs; every output kept (files, stdout without the timing/path lines, stderr,
# exit code). Usage: f32_runs.sh <dtwc_cl.exe> <out_dir> <gen_inputs.py's dir> <the repo's data/dummy>
# @author Volkan Kumtepeli
# @author Claude Opus 5.5
set -u
BIN="$1"
OUT="$2"
IN="$3"
DUMMY="$4"
rm -rf "$OUT"; mkdir -p "$OUT"
run() {
  local name="$1"; shift
  local dir="$OUT/$name"
  mkdir -p "$dir/files"
  "$BIN" -o "$dir/files" "$@" > "$dir/stdout.raw" 2> "$dir/stderr.txt"
  echo $? > "$dir/exit.txt"
  grep -v "Time:\|Output:\|\[.*s\]\|min:sec" "$dir/stdout.raw" | sed "s#$OUT#OUT#g" > "$dir/stdout.txt"
  rm -f "$dir/stdout.raw"
}
for m in kmedoids pam clara; do
  run folder_$m  -i "$DUMMY" --skip-rows 1 --skip-cols 1 -k 3 -m $m --seed 123 --dtype float32
  run batch_$m   -i "$IN/batch.csv" -k 3 -m $m --seed 123 --dtype float32
  run list_$m    -i "$IN/list.parquet" -k 3 -m $m --seed 123 --dtype float32
  run rows_$m    -i "$IN/rows.parquet" -k 3 -m $m --seed 123 --dtype float32
  run ipc_$m     -i "$IN/series.arrow" -k 3 -m $m --seed 123 --dtype float32
done
run list_skip_pam  -i "$IN/list.parquet" -k 3 -m pam --skip-rows 7 --dtype f32
run rows_skip_pam  -i "$IN/rows.parquet" -k 3 -m pam --skip-rows 7 --skip-cols 1 --dtype fp32
for f in batch.csv list.parquet series.arrow; do
  run f64_${f%%.*}_pam -i "$IN/$f" -k 3 -m pam
done
run verbose_batch_f32  -i "$IN/batch.csv" -k 3 -m pam --dtype float32 -v
run verbose_list_f32   -i "$IN/list.parquet" -k 3 -m pam --dtype float32 -v
