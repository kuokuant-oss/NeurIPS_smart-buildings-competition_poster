#!/bin/bash
# usage: go_p2pr.sh <workers> <what...>
R="$(cd "$(dirname "$0")/../.." && pwd)"; S="$R/data/final"; P="$R/pipeline/final"; mkdir -p "$S"
W="$1"; shift
cd "$R"
for what in "$@"; do
  nice -n 10 python3 -B pipeline/launch/launch.py "$P/pr_p2.py" "$S" "$W" "$what" > "$S/pr_p2_$what.out" 2>&1
  grep -E "SMOKE|RESULT|ERROR|DONE|Traceback|Error" "$S/pr_p2_$what.out" | cut -c1-400; tail -n 2 "$S/pr_p2_$what.out" | cut -c1-300
done
exit 0
