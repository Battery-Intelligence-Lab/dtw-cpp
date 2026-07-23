#!/usr/bin/env bash
set -euo pipefail

readonly min_assertions=348
readonly min_cases=11

if [[ $# -ne 1 ]]; then
  echo "::error::F9: usage: $0 <test_io_readers log|->"
  exit 2
fi

readonly log_path=$1
if [[ "$log_path" == "-" ]]; then
  readonly log_text=$(cat)
elif [[ -r "$log_path" ]]; then
  readonly log_text=$(<"$log_path")
else
  echo "::error::F9: unreadable log: $log_path"
  exit 2
fi

if grep -q 'DTWC_HAS_ARROW not defined' <<<"$log_text"; then
  echo "::error::F9: test_io_readers SKIPPED - Arrow did not compile into dtwc++."
  exit 1
fi

if grep -Eqi 'skip(ped)?' <<<"$log_text"; then
  echo "::error::F9: test_io_readers reported a skip - the suite must run in full."
  exit 1
fi

# Require one unambiguous Catch2 summary and capture both dimensions. A case
# floor alone can pass when assertions inside those cases silently compile out.
readonly summary=$(
  sed -n \
    's/.*All tests passed (\([0-9][0-9]*\) assertions in \([0-9][0-9]*\) test cases).*/\1 \2/p' \
    <<<"$log_text"
)

if [[ ! "$summary" =~ ^([0-9]+)\ ([0-9]+)$ ]]; then
  echo "::error::F9: expected exactly one parseable Catch2 success summary."
  exit 1
fi

readonly assertions=${BASH_REMATCH[1]}
readonly cases=${BASH_REMATCH[2]}

if (( 10#$assertions < min_assertions )); then
  echo "::error::F9: only $assertions assertions ran, expected >= $min_assertions."
  exit 1
fi

if (( 10#$cases < min_cases )); then
  echo "::error::F9: only $cases test cases ran, expected >= $min_cases (4 Arrow + 7 Parquet)."
  exit 1
fi

echo "F9 gate PASS: test_io_readers executed $assertions assertions in $cases cases, no skips."
