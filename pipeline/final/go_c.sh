#!/bin/bash
# usage: go_c.sh <scene regex> <workers> <names...>
R="$(cd "$(dirname "$0")/../.." && pwd)"; S="$R/data/final"; P="$R/pipeline/final"; mkdir -p "$S"
case "$1" in
  cold) export SCENE_RE='fairbanks|beijing_jan|london_jan|seattle_jan' ;;
  all) export SCENE_RE='.' ;;
  off) export SCENE_RE='/(atlanta|denver|fairbanks|phoenix|singapore)_[0-9]+$' ;;
  new) export SCENE_RE='_(jan|jul)$' ;;
  *) export SCENE_RE="$1" ;;
esac
shift
W="$1"; shift
cd "$R" && nice -n 10 python3 -B pipeline/launch/launch.py "$P/run_c.py" "$S" "$W" "$@"
