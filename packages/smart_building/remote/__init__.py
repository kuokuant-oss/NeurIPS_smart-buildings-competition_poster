"""Remote execution of long runs (docs/plan/baseline-v1-remote.md).

The remote side only runs tasks and writes raw per-task results; every
statistic and decision is recomputed locally from the fetched files.

Modules: paths (where things are), scenarios (overlay root), tasks (what one
task does), runner (the shard loop), jobs (job definitions), package (the
uploaded run.py).
"""
