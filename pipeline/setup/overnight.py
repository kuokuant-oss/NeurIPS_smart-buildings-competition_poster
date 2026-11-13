"""Overnight autotune with checkpoints (2026-10-29 20:1x; user: staged checks, "中間就要先驗一遍"). From the short run's
end state at_short1_013 (TRAIN 100 -21.07%, INNER 29 -20.17% against the original rule): stage after stage, one
autotune v2 sweep (pipeline/setup/autotune.py) over per-band moves, then the end state on INNER 29 (fast simulator); the next
stage starts only if that end state's INNER 29 mean cost is lower than the stage's start. Stops at the first stage that
keeps nothing or fails its check. Results: data/runs/b2rl/autotune/<tag>/, ROUND lines in data/runs/b2rl/search/run.log.
Host python3 through pipeline/launch/launch.py (the project environment)."""
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import autotune  # noqa: E402
import search  # noqa: E402

CURVE = ROOT / "data/runs/b2rl/eval/curve"
SP_FIXED = 1807.1323159420097        # the fixed model's sp_max_pa, written into every band (the start stays identical)
MOVES = [
    {"parameter": "sat_cost_guard_k", "bands": "mild", "step": 0.25, "low": 0.0, "high": 1.0, "reach": 1},
    {"parameter": "sat_cost_guard_k", "bands": "hot", "step": 0.25, "low": 0.0, "high": 1.5, "reach": 1},
    {"parameter": "heating_max_ratio", "bands": "cold", "step": 0.04, "low": 0.1, "high": 0.5, "reach": 1},
    {"parameter": "heating_max_ratio", "bands": "hot", "step": 0.04, "low": 0.1, "high": 0.5, "reach": 1},
    {"parameter": "heating_max_ratio", "bands": "mild", "step": 0.04, "low": 0.1, "high": 0.5, "reach": 1},
    {"parameter": "unoccupied_lead_min", "bands": "cold", "step": 60, "low": 0, "high": 360, "reach": 1},
    {"parameter": "unoccupied_lead_min", "bands": "mild", "step": 60, "low": 0, "high": 360, "reach": 1},
    {"parameter": "unoccupied_lead_min", "bands": "hot", "step": 60, "low": 0, "high": 360, "reach": 1},
    {"parameter": "t_opt_max_s", "bands": "cold", "step": 1800, "low": 3600, "high": 21600, "reach": 1},
    {"parameter": "occupied_margin_k", "bands": "hot", "step": 0.05, "low": 0.0, "high": 1.0, "reach": 1},
    {"parameter": "sp_max_pa", "bands": "all", "step": 400.0, "low": 200.0, "high": 2000.0, "reach": 1},
]


def inner_mean(name):
    out = CURVE / f"search-{name}-inner-rom.json"
    if not out.exists():
        env = {k: v for k, v in os.environ.items() if k != "SB_FASTSIM"}
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run(["nice", "-n", "5", "python3", "-B", "scripts/b2rl.py", "eval", "--policy",
                        f"data/runs/b2rl/search/models/model-{name}.json", "--set", "inner", "--workers", "16",
                        "--fast", "rom", "--out", str(out.relative_to(ROOT))], cwd=ROOT, env=env, check=True,
                       capture_output=True)
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    return st.mean(r["cost"] for r in rows.values())


def main():
    rule = json.loads((CURVE / "rule-inner-rom.json").read_text(encoding="utf-8"))["rows"]
    rule_inner = st.mean(r["cost"] for r in rule.values())
    if len(sys.argv) > 1:              # restart from another end state: <round json> <name> <tag prefix> <first stage>
        start_round, start_name, prefix, first = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
        start_rows, start_set = None, {}
        spec = json.loads((ROOT / start_round).read_text(encoding="utf-8"))[start_name]
        search.build_schedule(start_name, spec)                        # its model, for the INNER check
    else:
        start_round, start_name, prefix, first = "data/runs/b2rl/search/autotune-short1-end.json", "at_short1_013", "night", 1
        start_rows = {"search": "at_short1_012", "holdout": "at_short1_013"}   # the 14 K state: search / holdout rows
        start_set = {band: {"sp_max_pa": SP_FIXED} for band in autotune.BANDS}
    for stage in range(first, first + 3):
        tag = f"{prefix}{stage}"
        plan = {"tag": tag, "start": {"round": start_round, "name": start_name, "set": start_set},
                "moves": MOVES, "sweeps": 1, **({"start_rows": start_rows} if start_rows else {})}
        (ROOT / f"pipeline/setup/autotune-{tag}.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
        summary = autotune.Tuner(plan, 16).run()
        if not summary["kept"]:
            search.log(f"ROUND overnight {tag}: nothing kept; stopping")
            return
        names = summary["rows_names"]                     # where the end state's search and holdout rows are
        end_name = names["holdout"]                        # (both names' models are the end spec)
        before, after = inner_mean(start_name), inner_mean(end_name)
        verdict = "passed" if after < before else "failed"
        search.log(f"ROUND overnight {tag} check {verdict}: INNER 29 {start_name} {100 * (before / rule_inner - 1):+.2f}% "
                   f"-> {end_name} {100 * (after / rule_inner - 1):+.2f}% vs the original rule (TRAIN 100 "
                   f"{100 * summary['end_vs_rule_train100']:+.2f}%)")
        if verdict == "failed":
            return
        end_round = ROOT / f"data/runs/b2rl/search/autotune-{tag}-end.json"
        end_round.write_text(json.dumps({end_name: summary["spec"]}, indent=1) + "\n", encoding="utf-8")
        start_round, start_name, start_set = str(end_round.relative_to(ROOT)), end_name, {}
        start_rows = names


if __name__ == "__main__":
    main()
