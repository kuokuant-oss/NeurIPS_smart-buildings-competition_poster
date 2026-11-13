#!/bin/bash
# per-step traces of both controllers on all 90 local validation scenarios with the corrected simulator (writes only <scratch>/trace_p2)
R="$(cd "$(dirname "$0")/../.." && pwd)"; S="$R/data/final"; P="$R/pipeline/final"; mkdir -p "$S"
cd "$R"
LIST=$(mktemp)
for d in data/raw/kaggle/smart_buildings/training_scenarios/*/* data/processed/newstations/*/*; do
  [ -f "$d/config.gin" ] || continue
  case "$d" in *inneap*) continue;; esac
  for m in pr_opt2 gt_x8; do
    key="${m}__$(basename $(dirname $d))__$(basename $d)"
    [ -f "$S/trace_p2/$key.npz" ] || echo "$m "$R"/$d" >> "$LIST"
  done
done
echo "todo $(wc -l < $LIST)"
cat "$LIST" | xargs -P 20 -L 1 bash -c 'nice -n 10 python3 -B pipeline/launch/launch.py "'"$S"'/trace_p2.py" "'"$S"'" "$0" "$1" 2>&1 | tail -n 1 | cut -c1-110'
echo ALLDONE
