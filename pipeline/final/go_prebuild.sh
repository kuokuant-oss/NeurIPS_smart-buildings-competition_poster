#!/bin/bash
R="$(cd "$(dirname "$0")/../.." && pwd)"; S="$R/data/final"; P="$R/pipeline/final"; mkdir -p "$S"
cd "$R" && nice -n 10 python3 -B pipeline/launch/launch.py "$P/prebuild_v2.py" "$S" "$1" > "$S/prebuild_v2.out" 2>&1
grep -c " ok " "$S/prebuild_v2.out"; grep -E "ERROR|Traceback|DONE" "$S/prebuild_v2.out" | cut -c1-300 | head; tail -n 2 "$S/prebuild_v2.out" | cut -c1-200
exit 0
