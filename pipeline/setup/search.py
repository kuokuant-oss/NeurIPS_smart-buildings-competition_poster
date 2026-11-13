"""Direct search for the fixed rule's parameters (user 2026-10-28 18:0x: "好，先拆解再搜尋"), after the breakdown
(pipeline/setup/breakdown.py, data/runs/b2rl/breakdown/summary.json): on train_breakdown40, reverting the comfort margin
to the rule's 0 loses 1.60 of the fixed rule's 1.96 points and doubling it to 0.145 K loses 0.08; reverting the
heating airflow ratio to 0.5 loses 0.54 and doubling its change (0.406) gains 0.23; the discharge offset is best near
its fixed value (11 K: +0.26, 12.13 K: +0.22); reverting the optimal start cap to the rule's 3 h gains 0.11; the
unoccupied lead and the static pressure cap do nothing (|effect| <= 0.03).

A round is a JSON file of candidates {name: {parameter: value}} (missing parameters = the fixed rule's); every candidate
is a constant-action model (pipeline/setup/ceiling.py's construction) run on the round's set with whole 30-day episodes on
the fast simulator (b2rl.py eval --fast rom; resumable). The summary prints each candidate's mean per-scenario relative
cost against the rule and, paired, against the fixed rule (rule and fixed rows of train_ceiling100 come from the ceiling
run). Search on train_breakdown40; the best are confirmed on the other 60 of train_ceiling100 (--set
train_ceiling_rest60), then INNER 29 (rom), then the Kaggle gates.

python3 -B pipeline/launch/launch.py pipeline/setup/search.py <round.json> [--set train_breakdown40] [--workers 8]
(log data/runs/b2rl/search/run.log; rows data/runs/b2rl/search/<set>/<candidate>.json)"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
OUT = ROOT / "data/runs/b2rl/search"
CEILING = ROOT / "data/runs/b2rl/ceiling"
FIXED = ROOT / "data/runs/b2rl/offset-check/model-fixed-u100.json"
EVAL_SETS = ROOT / "configs/b2rl/eval-sets.json"
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


def ensure_rest60():
    """train_ceiling_rest60 = train_ceiling100 without train_breakdown40 (the search's held-out check)."""
    sets = json.loads(EVAL_SETS.read_text(encoding="utf-8"))
    rest = [s for s in sets["train_ceiling100"] if s not in set(sets["train_breakdown40"])]
    if "train_ceiling_rest60" in sets:
        if sets["train_ceiling_rest60"] != rest:
            raise SystemExit("train_ceiling_rest60 differs from train_ceiling100 minus train_breakdown40")
        return
    sets["train_ceiling_rest60"] = rest
    sets["_note"] += (" train_ceiling_rest60 = train_ceiling100 without train_breakdown40 (2026-10-28, the held-out "
                      "check of pipeline/setup/search.py's parameter search on train_breakdown40).")
    EVAL_SETS.write_text(json.dumps(sets, indent=1) + "\n", encoding="utf-8")


def inverse_action(p, d, low, high):
    if p >= d:
        return 0.0 if high == d else (p - d) / (high - d)
    return 0.0 if d == low else (p - d) / (d - low)


def with_rule_config(built, rule_config):
    """A candidate's "rule_config" (whitelisted G36 switches on top of G36PolicyB2R2, policies/b2rl.py) goes into the
    model; load_model checks it."""
    from smart_building.policies.b2rl import load_model
    if rule_config:
        built["rule_config"] = dict(rule_config)
        load_model(built)
    return built


def build_model(name, overrides):
    from smart_building.policies.b2rl import LAYOUTS, ParameterMap, load_model
    from smart_building.policies.g36 import SLOW_PARAMETER_RANGES
    overrides = dict(overrides)
    rule_config = overrides.pop("rule_config", None)
    model = load_model(FIXED)
    names = LAYOUTS[model["layout"]]["parameters"]
    defaults = json.loads((CEILING / "parameter-sets.json").read_text(encoding="utf-8"))["defaults"]
    ranges = {k: SLOW_PARAMETER_RANGES[k] for k in names}
    parameter_map = ParameterMap(defaults, ranges, mapping="two_sided")
    values = {**parameter_map.values(model["layers"][-1]["bias"]), **overrides}
    unknown = set(overrides) - set(names)
    if unknown:
        raise SystemExit(f"{name}: unknown parameters {sorted(unknown)}")
    for k, v in values.items():
        if not ranges[k][0] <= v <= ranges[k][1]:
            raise SystemExit(f"{name}: {k} = {v} outside {ranges[k]}")
    bias = [inverse_action(values[k], defaults[k], *ranges[k]) for k in names]
    back = parameter_map.values(bias)
    if any(abs(back[k] - values[k]) > (0.51 if k == "unoccupied_lead_min" else 1e-9) for k in names):
        raise SystemExit(f"{name}: the constant action does not give its parameters: {back} vs {values}")
    built = dict(model)
    last = model["layers"][-1]
    built["layers"] = model["layers"][:-1] + [{"weight": [[0.0] * len(last["weight"][0])] * len(names),
                                               "bias": bias}]
    built["source"] = {**model.get("source", {}), "search": f"constant action: {name} {json.dumps(values)}"}
    with_rule_config(built, rule_config)
    path = OUT / "models" / f"model-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(built) + "\n", encoding="utf-8")
    return path, {**back, **({"rule_config": rule_config} if rule_config else {})}


OUTSIDE_24H = 6            # policies/b2rl.py OBSERVATION_FEATURES: "outside_air_24h" = 24 h mean outside air (deg C) / 20


def build_schedule(name, spec):
    """Side line 1 (user 2026-10-28 22:4x, "依外氣溫度切換"): parameters scheduled on the 24 h mean outside air, as a
    hand-built G36PolicyB2RL network (no new policy code): {"cold": {...}, "mild": {...}, "hot": {...},
    "breaks_c": [c1, c2, h1, h2]} gives the cold set up to c1, a linear change to the mild set by c2, the mild set up
    to h1 and a linear change to the hot set by h2 (deg C; missing parameters = the fixed rule's). The change is linear
    in the action, so a parameter's path is piecewise linear with at most one more kink at the rule's own value. Four
    ReLU units on that feature; zero weight on every other feature; identity input normalisation.
    "layout": "v3" (2026-10-29) also schedules the A12 comfort guard width; a parameter the fixed model does not have
    starts from the rule's own value after the candidate's rule_config, which is what HourlyRule takes as a = 0."""
    from smart_building.policies.b2rl import LAYOUTS, Actor, ParameterMap, layout_features, load_model
    from smart_building.policies.g36 import (C3_COMBINATIONS, SLOW_PARAMETER_RANGES, G36PolicyB2R2,
                                             c3_configured_config)
    model = load_model(FIXED)
    base_names = LAYOUTS[model["layout"]]["parameters"]
    defaults = json.loads((CEILING / "parameter-sets.json").read_text(encoding="utf-8"))["defaults"]
    fixed = ParameterMap(defaults, {k: SLOW_PARAMETER_RANGES[k] for k in base_names},
                         mapping="two_sided").values(model["layers"][-1]["bias"])
    layout = spec.get("layout", model["layout"])
    names = LAYOUTS[layout]["parameters"]
    rule_cfg = c3_configured_config({**C3_COMBINATIONS[G36PolicyB2R2.combination], **(spec.get("rule_config") or {})})
    extra = {k: float(getattr(rule_cfg, k)) for k in names if k not in base_names}
    defaults, fixed = {**defaults, **extra}, {**fixed, **extra}
    ranges = {k: SLOW_PARAMETER_RANGES[k] for k in names}
    parameter_map = ParameterMap(defaults, ranges, mapping="two_sided")
    model = {**model, "layout": layout, "features": list(layout_features(layout)), "parameters": list(names)}
    if "knots_c" in spec:
        built = build_knots(name, spec, model, names, ranges, defaults, fixed, parameter_map)
        with_rule_config(built, spec.get("rule_config"))
        path = OUT / "models" / f"model-{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(built) + "\n", encoding="utf-8")
        return path, {"schedule": spec}
    c1, c2, h1, h2 = (float(x) / 20.0 for x in spec["breaks_c"])
    if not c1 < c2 <= h1 < h2:
        raise SystemExit(f"{name}: breaks must rise: {spec['breaks_c']}")
    action = {}
    for region in ("cold", "mild", "hot"):
        values = {**fixed, **spec.get(region, {})}
        for k, v in values.items():
            if not ranges[k][0] <= v <= ranges[k][1]:
                raise SystemExit(f"{name}: {region} {k} = {v} outside {ranges[k]}")
        action[region] = np.array([inverse_action(values[k], defaults[k], *ranges[k]) for k in names])
    n = len(model["features"])
    hidden = [[0.0] * n for _ in range(4)]
    for row in hidden:
        row[OUTSIDE_24H] = 1.0
    up, down = (action["mild"] - action["cold"]) / (c2 - c1), (action["hot"] - action["mild"]) / (h2 - h1)
    out = np.stack([up, -up, down, -down], axis=1)
    built = {**model, "obs_mean": [0.0] * n, "obs_var": [1.0] * n, "obs_epsilon": 0.0, "obs_clip": 1e9,
             "layers": [{"weight": hidden, "bias": [-c1, -c2, -h1, -h2]},
                        {"weight": out.tolist(), "bias": action["cold"].tolist()}],
             "log_std": [0.0] * len(names),
             "source": {**model.get("source", {}), "search": f"outside-air schedule: {name} {json.dumps(spec)}"}}
    actor = Actor(built)                                   # check: every region gives its own parameters
    for region, t in (("cold", c1 * 20 - 5), ("mild", (c2 + h1) * 10), ("hot", h2 * 20 + 5)):
        features = np.zeros(n)
        features[OUTSIDE_24H] = t / 20.0
        got, want = parameter_map.values(actor(features)), {**fixed, **spec.get(region, {})}
        if any(abs(got[k] - want[k]) > (0.51 if k == "unoccupied_lead_min" else 1e-6) for k in names):
            raise SystemExit(f"{name}: {region} at {t} C gives {got}, not {want}")
    with_rule_config(built, spec.get("rule_config"))
    path = OUT / "models" / f"model-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(built) + "\n", encoding="utf-8")
    return path, {"schedule": spec}


def build_knots(name, spec, model, names, ranges, defaults, fixed, parameter_map):
    """The general outside-air schedule (2026-10-30, stage 1 "更細的外氣分段"): {"knots_c": [t1 < ... < tK] (deg C,
    24 h mean outside air), "levels": [the band of each knot], "<band>": {...}} gives each knot its band's parameters,
    a linear change in the action between knots and the end values outside them; K ReLU units, f(x) = a1 +
    sum_k c_k relu(x - t_k). knots [0, 5, 20, 25] with levels [cold, mild, mild, hot] is build_schedule's three-band
    form; a band may own several knots."""
    from smart_building.policies.b2rl import Actor
    knots = [float(x) / 20.0 for x in spec["knots_c"]]
    levels = list(spec["levels"])
    if len(knots) != len(levels) or len(knots) < 2 or any(b <= a for a, b in zip(knots, knots[1:])):
        raise SystemExit(f"{name}: knots must rise and match the levels: {spec['knots_c']} {levels}")
    action = {}
    for band in dict.fromkeys(levels):
        values = {**fixed, **spec.get(band, {})}
        for k, v in values.items():
            if not ranges[k][0] <= v <= ranges[k][1]:
                raise SystemExit(f"{name}: {band} {k} = {v} outside {ranges[k]}")
        action[band] = np.array([inverse_action(values[k], defaults[k], *ranges[k]) for k in names])
    a = [action[band] for band in levels]
    slopes = [(a[k + 1] - a[k]) / (knots[k + 1] - knots[k]) for k in range(len(knots) - 1)]
    coef = [slopes[0]] + [slopes[k] - slopes[k - 1] for k in range(1, len(slopes))] + [-slopes[-1]]
    n = len(model["features"])
    hidden = [[0.0] * n for _ in knots]
    for row in hidden:
        row[OUTSIDE_24H] = 1.0
    built = {**model, "obs_mean": [0.0] * n, "obs_var": [1.0] * n, "obs_epsilon": 0.0, "obs_clip": 1e9,
             "layers": [{"weight": hidden, "bias": [-t for t in knots]},
                        {"weight": np.stack(coef, axis=1).tolist(), "bias": a[0].tolist()}],
             "log_std": [0.0] * len(names),
             "source": {**model.get("source", {}), "search": f"outside-air schedule: {name} {json.dumps(spec)}"}}
    actor = Actor(built)                                   # check: every knot (and beyond the ends) gives its band's
    checks = list(zip(knots, levels)) + [(knots[0] - 0.25, levels[0]), (knots[-1] + 0.25, levels[-1])]
    for t, band in checks:
        features = np.zeros(n)
        features[OUTSIDE_24H] = t
        got, want = parameter_map.values(actor(features)), {**fixed, **spec.get(band, {})}
        if any(abs(got[k] - want[k]) > (0.51 if k == "unoccupied_lead_min" else 1e-6) for k in names):
            raise SystemExit(f"{name}: {band} at {20 * t} C gives {got}, not {want}")
    return built


def run_candidate(name, model_path, subset, workers):
    out = OUT / subset / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    while available_mb() < MIN_AVAILABLE_MB:
        time.sleep(60)
    log(f"{subset} {name}: eval starts ({available_mb()} MB available, {workers} workers)")
    env = {k: v for k, v in os.environ.items() if k != "SB_FASTSIM"}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(["nice", "-n", "5", "python3", "-B", "scripts/b2rl.py", "eval", "--policy",
                             str(model_path.relative_to(ROOT)), "--set", subset, "--workers", str(workers), "--fast",
                             "rom", "--out", str(out.relative_to(ROOT))], cwd=ROOT, env=env, capture_output=True,
                            text=True)
    if result.returncode:
        log(f"{subset} {name}: eval failed: {(result.stdout + result.stderr).strip()[-600:]}")
        raise SystemExit(1)
    return json.loads(out.read_text(encoding="utf-8"))["rows"]


def reference_rows(name, specs):
    rows = json.loads((CEILING / f"{name}.json").read_text(encoding="utf-8"))["rows"]
    return {s: rows[s] for s in specs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round")
    parser.add_argument("--set", default="train_breakdown40")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    ensure_rest60()
    specs = json.loads(EVAL_SETS.read_text(encoding="utf-8"))[args.set]
    candidates = json.loads((ROOT / args.round).read_text(encoding="utf-8"))
    rule, fixed = reference_rows("rule", specs), reference_rows("fixed", specs)
    base = st.mean(fixed[s]["cost"] / rule[s]["cost"] - 1 for s in specs)
    results = {}
    for name, overrides in candidates.items():
        if name.startswith("_"):
            continue
        path, values = (build_schedule(name, overrides) if "breaks_c" in overrides or "knots_c" in overrides
                        else build_model(name, overrides))
        rows = run_candidate(name, path, args.set, args.workers)
        to_rule = [rows[s]["cost"] / rule[s]["cost"] - 1 for s in specs]
        to_fixed = [rows[s]["cost"] / fixed[s]["cost"] - 1 for s in specs]
        results[name] = {"parameters": values, "score": st.mean(to_rule),
                         "vs_fixed": st.mean(to_fixed), "vs_fixed_se": st.stdev(to_fixed) / math.sqrt(len(specs)),
                         "better_than_fixed": sum(rows[s]["cost"] < fixed[s]["cost"] for s in specs),
                         "worse_than_fixed": sum(rows[s]["cost"] > fixed[s]["cost"] for s in specs)}
        r = results[name]
        log(f"{args.set} {name}: {100 * r['score']:+.2f}% vs rule ({100 * (r['score'] - base):+.2f} points vs fixed "
            f"{100 * base:+.2f}%; paired {100 * r['vs_fixed']:+.2f}% SE {100 * r['vs_fixed_se']:.2f}%, "
            f"{r['better_than_fixed']}/{r['worse_than_fixed']} better/worse than fixed) {json.dumps(overrides)}")
    summary_path = OUT / args.set / f"summary-{Path(args.round).stem}.json"
    summary_path.write_text(json.dumps({"set": args.set, "round": args.round, "fixed_score": base,
                                        "results": results}, indent=1) + "\n", encoding="utf-8")
    best = min(results, key=lambda k: results[k]["score"])
    log(f"ROUND {Path(args.round).stem} on {args.set} done: best {best} {100 * results[best]['score']:+.2f}% vs rule "
        f"(fixed {100 * base:+.2f}%)")


if __name__ == "__main__":
    main()
