"""Which of the fixed-parameter rule's six parameters carry its gain (user 2026-10-28 18:0x: "好，先拆解再搜尋"; the
fixed rule passed both gates and was submitted as sb-fixed-u100 at 18:03). One parameter at a time on the line through
the rule's value (0x) and the fixed rule's value (1x): back to the rule's value (revert_<k>, the other five fixed) and
twice the fixed rule's offset from the rule (double_<k>, clipped to SLOW_PARAMETER_RANGES). Each variant is a constant-
action model (pipeline/setup/ceiling.py's construction) and runs whole 30-day episodes on the fast simulator (b2rl.py eval
--fast rom) over configs/b2rl/eval-sets.json "train_breakdown40": 8 per station of train_ceiling100 (TRAIN without
train_val25's buildings), random.Random(20261102). The rule and the fixed rule rows of those scenarios come from the
ceiling run (data/runs/b2rl/ceiling/rule.json, fixed.json; same simulator and loop).

Summary (data/runs/b2rl/breakdown/summary.json): per variant the mean per-scenario relative cost against the rule
(score) and against the fixed rule (paired: mean, SE, better/worse), and the cost parts' change against the fixed rule;
per parameter the effect of reverting it (score(revert) - score(fixed): > 0 means its change helps) and of doubling it
(< 0 means more of it helps). The direct search that follows uses these directions.

python3 -B pipeline/launch/launch.py pipeline/setup/breakdown.py [workers, default 8]   (log data/runs/b2rl/breakdown/run.log)"""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import statistics as st
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
OUT = ROOT / "data/runs/b2rl/breakdown"
CEILING = ROOT / "data/runs/b2rl/ceiling"
FIXED = ROOT / "data/runs/b2rl/offset-check/model-fixed-u100.json"
EVAL_SETS = ROOT / "configs/b2rl/eval-sets.json"
SET, SOURCE_SET = "train_breakdown40", "train_ceiling100"
PER_STATION, SEED = 8, 20261102
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MIN_AVAILABLE_MB = 8000


def log(text):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now():%m-%d %H:%M:%S} {text}\n")
    print(text, flush=True)


def available_mb():
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def station(spec):
    return Path(spec).name.split("_")[0]


def choose_scenarios():
    sets = json.loads(EVAL_SETS.read_text(encoding="utf-8"))
    source = sets[SOURCE_SET]
    rng, chosen = random.Random(SEED), []
    for name in sorted({station(s) for s in source}):
        chosen += sorted(rng.sample(sorted(s for s in source if station(s) == name), PER_STATION))
    if SET in sets and sets[SET] != chosen:
        raise SystemExit(f"{SET} in {EVAL_SETS.relative_to(ROOT)} differs from this selection")
    if SET not in sets:
        sets[SET] = chosen
        sets["_note"] += (f" {SET} = pipeline/setup/breakdown.py's set (2026-10-28, the fixed rule's one-parameter-at-a-time "
                          f"breakdown): {PER_STATION} per station of {SOURCE_SET}, random.Random({SEED}).")
        EVAL_SETS.write_text(json.dumps(sets, indent=1) + "\n", encoding="utf-8")
    return chosen


def inverse_action(p, d, low, high):
    """layout v2's two-sided mapping inverted (pipeline/setup/offset_check.py)."""
    if p >= d:
        return 0.0 if high == d else (p - d) / (high - d)
    return 0.0 if d == low else (p - d) / (d - low)


def build_models():
    from smart_building.policies.b2rl import LAYOUTS, ParameterMap, load_model
    from smart_building.policies.g36 import SLOW_PARAMETER_RANGES
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(FIXED)
    names = LAYOUTS[model["layout"]]["parameters"]
    defaults = json.loads((CEILING / "parameter-sets.json").read_text(encoding="utf-8"))["defaults"]
    ranges = {k: SLOW_PARAMETER_RANGES[k] for k in names}
    parameter_map = ParameterMap(defaults, ranges, mapping="two_sided")
    fixed = parameter_map.values(model["layers"][-1]["bias"])
    values = {}
    for k in names:
        values[f"revert_{k}"] = {**fixed, k: defaults[k]}
        doubled = min(max(defaults[k] + 2.0 * (fixed[k] - defaults[k]), ranges[k][0]), ranges[k][1])
        values[f"double_{k}"] = {**fixed, k: int(round(doubled)) if k == "unoccupied_lead_min" else doubled}
    paths = {}
    for name, v in values.items():
        bias = [inverse_action(v[k], defaults[k], *ranges[k]) for k in names]
        back = parameter_map.values(bias)
        if any(abs(back[k] - v[k]) > (0.51 if k == "unoccupied_lead_min" else 1e-9) for k in names):
            raise SystemExit(f"{name}: the constant action does not give its parameters: {back} vs {v}")
        built = dict(model)
        last = model["layers"][-1]
        built["layers"] = model["layers"][:-1] + [{"weight": [[0.0] * len(last["weight"][0])] * len(names),
                                                   "bias": bias}]
        built["source"] = {**model.get("source", {}), "breakdown": f"constant action: {name}"}
        path = OUT / f"model-{name}.json"
        path.write_text(json.dumps(built) + "\n", encoding="utf-8")
        paths[name] = str(path.relative_to(ROOT))
    info = {"defaults": defaults, "fixed": fixed, "values": values, "models": paths}
    (OUT / "parameter-sets.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    return paths, info, names


def run_set(name, policy):
    out = OUT / f"{name}.json"
    while available_mb() < MIN_AVAILABLE_MB:
        time.sleep(60)
    log(f"{name}: eval starts ({available_mb()} MB available, {WORKERS} workers)")
    env = {k: v for k, v in os.environ.items() if k != "SB_FASTSIM"}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(["nice", "-n", "5", "python3", "-B", "scripts/b2rl.py", "eval", "--policy", policy,
                             "--set", SET, "--workers", str(WORKERS), "--fast", "rom", "--out",
                             str(out.relative_to(ROOT))], cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        log(f"{name}: eval failed: {(result.stdout + result.stderr).strip()[-600:]}")
        raise SystemExit(1)
    log(f"{name}: eval done: {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''}")
    return json.loads(out.read_text(encoding="utf-8"))["rows"]


def ceiling_rows(name, specs):
    """The ceiling run's rows of these scenarios (waits until that set has all of them)."""
    path = CEILING / f"{name}.json"
    while True:
        if path.is_file():
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            if all(s in rows for s in specs):
                return {s: rows[s] for s in specs}
        time.sleep(60)


def summarize(specs, rows, info, names):
    rule, fixed = rows["rule"], rows["fixed"]
    n = len(specs)

    def rel(a, b, s, key="cost"):
        return a[s][key] / b[s][key] - 1.0

    per = {}
    for name, r in rows.items():
        to_rule = [rel(r, rule, s) for s in specs]
        to_fixed = [rel(r, fixed, s) for s in specs]
        steps = {s: r[s]["steps"] for s in specs}
        parts = {p: st.mean((r[s][p] - fixed[s][p]) / steps[s] for s in specs) * -1.0
                 for p in ("comfort_part", "energy_part", "carbon_part")}
        per[name] = {"score": st.mean(to_rule), "vs_fixed": st.mean(to_fixed),
                     "vs_fixed_se": (st.stdev(to_fixed) / math.sqrt(n)) if name != "fixed" else 0.0,
                     "better_than_fixed": sum(r[s]["cost"] < fixed[s]["cost"] for s in specs),
                     "worse_than_fixed": sum(r[s]["cost"] > fixed[s]["cost"] for s in specs),
                     "parts_vs_fixed_per_step": parts}
    effects = {k: {"rule_value": info["defaults"][k], "fixed_value": info["fixed"][k],
                   "double_value": info["values"][f"double_{k}"][k],
                   "revert_points": 100 * (per[f"revert_{k}"]["score"] - per["fixed"]["score"]),
                   "double_points": 100 * (per[f"double_{k}"]["score"] - per["fixed"]["score"])} for k in names}
    summary = {"set": SET, "n": n, "per_variant": per, "effects": effects}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    log(f"SUMMARY fixed {100 * per['fixed']['score']:+.2f}% vs rule on {SET} ({n} scenarios)")
    for k, e in effects.items():
        log(f"  {k:20s} rule {e['rule_value']:.4g} fixed {e['fixed_value']:.4g} double {e['double_value']:.4g}: "
            f"revert {e['revert_points']:+.2f} points, double {e['double_points']:+.2f} points "
            f"(revert {per[f'revert_{k}']['better_than_fixed']}/{per[f'revert_{k}']['worse_than_fixed']} better/worse "
            f"than fixed; double {per[f'double_{k}']['better_than_fixed']}/{per[f'double_{k}']['worse_than_fixed']})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = choose_scenarios()
    log(f"{SET}: {len(specs)} scenarios, {len({Path(s).parent.name for s in specs})} buildings")
    paths, info, names = build_models()
    log("variants: " + json.dumps({k: {p: round(x, 4) for p, x in v.items()} for k, v in info["values"].items()}))
    rows = {}
    for name in [f"revert_{k}" for k in names] + [f"double_{k}" for k in names]:
        rows[name] = run_set(name, paths[name])
    rows["rule"], rows["fixed"] = ceiling_rows("rule", specs), ceiling_rows("fixed", specs)
    summarize(specs, rows, info, names)


if __name__ == "__main__":
    main()
