#!/bin/sh
# The bitwise sweep (pprobe check 3) for each version, placement p0; outputs in check_<version>.txt.
cd "$(dirname "$0")" || exit 1
for v in ${@:-base k1 head}; do
  start=$(date +%s)
  "./pprobe_${v}_p0" check 3 > "check_$v.txt" 2>&1
  echo "check $v exit $? in $(( $(date +%s) - start )) s: $(tail -1 "check_$v.txt")"
done
