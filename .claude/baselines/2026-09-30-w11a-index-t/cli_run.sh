#!/usr/bin/env bash
# Run dtwc_cl on data/dummy for each method; keep every output file plus the
# printed summary without its wall-clock line.  usage: cli_run.sh <dtwc_cl> <out_dir>
set -u
CL=$1
OUT=$2
REPO=${REPO:-C:/D/git/wt/W11a}
rm -rf "$OUT"; mkdir -p "$OUT"
run() {
  local tag=$1; shift
  "$CL" -i "$REPO/data/dummy" --skip-rows 1 --skip-cols 1 -o "$OUT/$tag" --name "$tag" "$@" \
    > "$OUT/$tag.stdout" 2>&1
  echo "$tag exit=$?" >> "$OUT/exit_codes.txt"
  grep -v "Time:\|Output:" "$OUT/$tag.stdout" > "$OUT/$tag.summary"
  rm "$OUT/$tag.stdout"
}
run pam -k 3 -m pam
run pam5 -k 5 -m pam --n-init 3
run clara -k 3 -m clara --sample-size 10 --n-samples 3
run clara4 -k 4 -m clara --sample-size 12 --n-samples 4 --seed 7
run onebatch -k 3 -m onebatch --batch-size 8
run kmedoids -k 3 -m kmedoids --n-init 2
run hierarchical -k 3 -m hierarchical --linkage complete
run tadpole -k 3 -m tadpole
run lrcore -k 3 -m lrcore
run mip -k 3 -m mip
cat "$OUT/exit_codes.txt"
