#!/bin/bash
# usage: go_v2.sh run <workers> <limit> <names...>   |   go_v2.sh trace <parallel>
R="$(cd "$(dirname "$0")/../.." && pwd)"; S="$R/data/final"; P="$R/pipeline/final"; mkdir -p "$S"
export SB_TANK=balance SB_RETURN_AIR=flow SB_FASTSIM_CACHE="${SB_FASTSIM_CACHE:-$S/rom-cache}"
cd "$R"
if [ "$1" = "run" ]; then
  shift; W="$1"; L="$2"; shift; shift
  tagf="$S/v2_run_$(echo "$@" | tr ' ' '_' | cut -c1-60).out"
  nice -n 10 python3 -B pipeline/launch/launch.py "$P/v2_run.py" "$S" "$W" "$L" "$@" > "$tagf" 2>&1
  grep -c -E "^(org_|g36|gt_|pr_|gm_)" "$tagf"; grep -E "ERROR|Traceback|DONE" "$tagf" | cut -c1-300 | head -8; tail -n 3 "$tagf" | cut -c1-200
else
  LIST=$(mktemp)
  python3 - "$S" > "$LIST" <<'E'
import json, sys, os
S = sys.argv[1]
for r in json.load(open(S + "/val2/manifest.json"))["scenarios"]:
    for m in ("pr_opt2", "gt_x8"):
        if not os.path.isfile(f"{S}/trace_p2/{m}__{r['building']}__{r['weather']}.npz"): print(m, r["dir"])
E
  echo "todo $(wc -l < $LIST)"
  cat "$LIST" | xargs -P "$2" -L 1 bash -c 'nice -n 10 python3 -B pipeline/launch/launch.py "'"$P"'/trace_p2.py" "'"$S"'" "$0" "$1" 2>&1 | tail -n 1 | cut -c1-110' > "$S/v2_trace.out" 2>&1
  grep -c DONE "$S/v2_trace.out"; grep -v DONE "$S/v2_trace.out" | head -5
  echo ALLDONE
fi
exit 0
