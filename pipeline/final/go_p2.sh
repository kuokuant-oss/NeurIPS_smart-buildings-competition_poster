#!/bin/bash
# Corrected-simulator runs (starter-kit PR #2 tank energy balance in memory + PR #3 airflow-weighted return air).
# usage: go_p2.sh <cold|all|regex> <workers> <names...>   -> scratch runs_p2/
export SB_TANK=balance SB_RETURN_AIR=flow RUNS_DIR=runs_p2
exec bash "$(dirname "$0")/go_c.sh" "$@"
