#!/bin/bash
R="$(cd "$(dirname "$0")/../.." && pwd)"; S="$R/data/final"; P="$R/pipeline/final"; mkdir -p "$S"
cd "$R" && python3 -B pipeline/launch/launch.py "$P/gen_val2.py" "$S" "$1" > "$S/gen_val2.out" 2>&1
grep -v -i minneap "$S/gen_val2.out" | grep -E "seeds already|DONE|Traceback|Error|rror:|npy|weather_csv|start_timestamp|convection|metrics_path|not builds" | cut -c1-260 | tail -n 30
tail -n 4 "$S/gen_val2.out" | cut -c1-260
cd "$R" && git status --short 2>/dev/null | head -5
exit 0
