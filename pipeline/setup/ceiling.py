"""How much "classify the building, then use that class's parameters" could gain at most (user 2026-10-28 17:0x:
"先量上限"; docs/reference/notes/20-building-classification-for-control.md §6 item 1 and §5.2 steps 1-2).

Five constant parameter sets run whole 30-day TRAIN episodes on the fast simulator (b2rl.py eval --fast rom, the same
gateway loop as every other evaluation):
  rule          the rule (G36PolicyB2R2)
  fixed         the fixed-parameter rule (data/runs/b2rl/offset-check/model-fixed-u100.json: the v2 actor's mean
                parameters at update 100 over train_val25, the version at the Kaggle gates)
  cold, warm    the same actor's mean parameters over its train_val25 decisions whose mean outside air of the last 24
                decisions is below / at least COLD_BELOW_C (data/runs/b2rl/offset-check/decisions-*.jsonl)
  fixed_sp2000  fixed with the static pressure cap back to the rule's 2000 Pa
Each is a constant-action model (the offset check's construction: zero output weights, the bias at the action whose
parameters are the set's values). Scenarios: 20 per station from the TRAIN pool (split-v1-r1) without the buildings of
train_val25 (they gave the parameters), random.Random(20261031), one building after another so the 20 spread over as
many buildings as possible; stored as configs/b2rl/eval-sets.json "train_ceiling100".

Summary (data/runs/b2rl/ceiling/summary.json): per set the mean per-scenario relative cost against the rule and the
mean cost; the single best set (SBS: lowest mean relative cost) and the per-scenario best (VBS: each scenario's lowest
of the five), their gap in percentage points (notes/20: under about 0.5 points the classification route stops), the
same for the pair {rule, fixed}, the best set per class for classes the hidden evaluation could not use directly
(station, building) and for one it could (the month's mean outside air below / at least COLD_BELOW_C, from the
station's weather file and the scenario's start). Everything in-sample: an upper bound, not a selector's result.

python3 -B pipeline/launch/launch.py pipeline/setup/ceiling.py [workers, default 8]   (log data/runs/b2rl/ceiling/run.log)"""
import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import random
import statistics as st
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
OUT = ROOT / "data/runs/b2rl/ceiling"
OFFSET = ROOT / "data/runs/b2rl/offset-check"
FIXED = OFFSET / "model-fixed-u100.json"
DECISIONS = OFFSET / "decisions-v2-rom-s42-model-u100.jsonl"
SUMMARY_U100 = OFFSET / "summary-v2-rom-s42-model-u100.json"
EVAL_SETS = ROOT / "configs/b2rl/eval-sets.json"
SET = "train_ceiling100"
PER_STATION = 20
SEED = 20261031
COLD_BELOW_C = 15.0
SETS = ("rule", "fixed", "cold", "warm", "fixed_sp2000")
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MIN_AVAILABLE_MB = 9000


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
    """20 per station from the TRAIN pool without train_val25's buildings, spread over buildings."""
    from smart_building.learned.b2rl_env import TRAIN_POOL, scenario_dirs
    sets = json.loads(EVAL_SETS.read_text(encoding="utf-8"))
    used = {Path(s).parent.name for s in sets["train_val25"]}
    specs = [f"train/{Path(d).parent.name}/{Path(d).name}" for d in scenario_dirs(TRAIN_POOL)]
    specs = [s for s in specs if Path(s).parent.name not in used]
    rng, chosen = random.Random(SEED), []
    for name in sorted({station(s) for s in specs}):
        by_building = {}
        for s in sorted(x for x in specs if station(x) == name):
            by_building.setdefault(Path(s).parent.name, []).append(s)
        buildings = sorted(by_building)
        rng.shuffle(buildings)
        for b in buildings:
            rng.shuffle(by_building[b])
        picked = []
        while len(picked) < PER_STATION and any(by_building.values()):
            for b in buildings:
                if by_building[b] and len(picked) < PER_STATION:
                    picked.append(by_building[b].pop())
        chosen += sorted(picked)
    if SET in sets and sets[SET] != chosen:
        raise SystemExit(f"{SET} in {EVAL_SETS.relative_to(ROOT)} differs from this selection")
    if SET not in sets:
        sets[SET] = chosen
        sets["_note"] += (f" {SET} = pipeline/setup/ceiling.py's set (2026-10-28, the class-parameter ceiling of notes/20 "
                          f"§6 item 1): {PER_STATION} TRAIN scenarios per station without train_val25's buildings, "
                          f"random.Random({SEED}), spread over buildings.")
        EVAL_SETS.write_text(json.dumps(sets, indent=1) + "\n", encoding="utf-8")
    return chosen


def segment_means():
    """The actor's mean parameters over its decisions in cold / warm hours (mean outside air of the last 24)."""
    rows = [json.loads(line) for line in DECISIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_spec = {}
    for r in rows:
        by_spec.setdefault(r["spec"], []).append(r)
    cold, warm = [], []
    for records in by_spec.values():
        outside = [r["outside"] for r in records]
        for i, r in enumerate(records):
            window = outside[max(0, i - 23):i + 1]
            (cold if sum(window) / len(window) < COLD_BELOW_C else warm).append(r)
    names = [k for k in rows[0] if k not in ("spec", "minute", "weekday", "outside", "occupants")]
    mean = lambda group: {k: float(st.mean(r[k] for r in group)) for k in names}
    return mean(cold), mean(warm), len(cold), len(warm)


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
    defaults = {k: v["default"] for k, v in json.loads(SUMMARY_U100.read_text(encoding="utf-8"))["parameters"].items()}
    ranges = {k: SLOW_PARAMETER_RANGES[k] for k in names}
    parameter_map = ParameterMap(defaults, ranges, mapping="two_sided")
    fixed = parameter_map.values(model["layers"][-1]["bias"])
    cold, warm, n_cold, n_warm = segment_means()
    values = {"fixed": fixed, "cold": cold, "warm": warm, "fixed_sp2000": {**fixed, "sp_max_pa": defaults["sp_max_pa"]}}
    paths = {"rule": "rule", "fixed": str(FIXED.relative_to(ROOT))}
    for name in ("cold", "warm", "fixed_sp2000"):
        bias = [inverse_action(values[name][k], defaults[k], *ranges[k]) for k in names]
        back = parameter_map.values(bias)
        if any(abs(back[k] - values[name][k]) > (0.51 if k == "unoccupied_lead_min" else 1e-9) for k in names):
            raise SystemExit(f"{name}: the constant action does not give its parameters: {back} vs {values[name]}")
        built = dict(model)
        last = model["layers"][-1]
        built["layers"] = model["layers"][:-1] + [{"weight": [[0.0] * len(last["weight"][0])] * len(names),
                                                   "bias": bias}]
        built["source"] = {**model.get("source", {}), "ceiling": f"constant action: the {name} parameter set"}
        path = OUT / f"model-{name}.json"
        path.write_text(json.dumps(built) + "\n", encoding="utf-8")
        paths[name] = str(path.relative_to(ROOT))
    info = {"defaults": defaults, "values": values, "decisions_cold": n_cold, "decisions_warm": n_warm,
            "cold_below_c": COLD_BELOW_C, "models": paths}
    (OUT / "parameter-sets.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    return paths, info


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


def outside_means(specs):
    """Each scenario's mean outside air over its 30 days (and first day), from the station's weather file."""
    cache, out = {}, {}
    from smart_building.learned.b2rl_env import spec_dir
    for spec in specs:
        gin = (Path(spec_dir(spec)) / "config.gin").read_text(encoding="utf-8")
        csv_path = next(line.split("=", 1)[1].strip().strip("'\"") for line in gin.splitlines()
                        if line.startswith("weather_csv_path"))
        start = datetime.strptime(next(line.split("=", 1)[1].strip().strip("'\"") for line in gin.splitlines()
                                       if line.startswith("start_timestamp")), "%Y-%m-%d %H:%M")
        if csv_path not in cache:
            with (ROOT / "data/raw/kaggle" / csv_path).open(encoding="utf-8", newline="") as stream:
                cache[csv_path] = [(datetime.strptime(r["Time"], "%Y%m%d-%H%M"), float(r["TempC"]))
                                   for r in csv.DictReader(stream)]
        rows = cache[csv_path]
        month = [t for d, t in rows if start <= d < start + timedelta(days=30)]
        day = [t for d, t in rows if start <= d < start + timedelta(days=1)]
        out[spec] = {"month": st.mean(month), "day1": st.mean(day)}
    return out


def best_by(groups, rel):
    """Per group the set with the lowest mean relative cost; the mean over scenarios with each group's choice."""
    choice, values = {}, []
    for group, specs in groups.items():
        means = {k: st.mean(rel[k][s] for s in specs) for k in SETS}
        choice[group] = min(means, key=means.get)
        values += [rel[choice[group]][s] for s in specs]
    return st.mean(values), choice


def summarize(specs, rows, info):
    rule = rows["rule"]
    rel = {k: {s: rows[k][s]["cost"] / rule[s]["cost"] - 1.0 for s in specs} for k in SETS}
    per_set = {}
    for k in SETS:
        diff = [rows[k][s]["cost"] - rule[s]["cost"] for s in specs]
        per_set[k] = {"score": st.mean(rel[k].values()), "mean_cost": st.mean(rows[k][s]["cost"] for s in specs),
                      "mean_diff": st.mean(diff), "better": sum(d < 0 for d in diff), "worse": sum(d > 0 for d in diff),
                      "best_in": sum(min(SETS, key=lambda x: rel[x][s]) == k for s in specs)}
    sbs = min(SETS, key=lambda k: per_set[k]["score"])
    vbs = st.mean(min(rel[k][s] for k in SETS) for s in specs)
    vbs_pair = st.mean(min(rel["rule"][s], rel["fixed"][s]) for s in specs)
    oa = outside_means(specs)
    groups = {"station": {}, "building": {}, "month_outside": {}, "day1_outside": {}}
    for s in specs:
        groups["station"].setdefault(station(s), []).append(s)
        groups["building"].setdefault(Path(s).parent.name, []).append(s)
        groups["month_outside"].setdefault("cold" if oa[s]["month"] < COLD_BELOW_C else "warm", []).append(s)
        groups["day1_outside"].setdefault("cold" if oa[s]["day1"] < COLD_BELOW_C else "warm", []).append(s)
    by_class = {}
    for kind, g in groups.items():
        value, choice = best_by(g, rel)
        by_class[kind] = {"score": value, "gain_over_sbs_points": 100 * (per_set[sbs]["score"] - value),
                          "classes": len(g), **({"choice": choice} if kind != "building" else {})}
    per_station = {name: {k: st.mean(rel[k][s] for s in g) for k in SETS} for name, g in groups["station"].items()}
    summary = {"set": SET, "n": len(specs), "parameter_sets": info["values"], "per_set": per_set, "sbs": sbs,
               "sbs_score": per_set[sbs]["score"], "vbs_score": vbs,
               "gap_points": 100 * (per_set[sbs]["score"] - vbs),
               "vbs_rule_fixed_score": vbs_pair, "gap_rule_fixed_points": 100 * (per_set["fixed"]["score"] - vbs_pair),
               "by_class": by_class, "per_station": per_station,
               "outside": {s: oa[s] for s in specs}, "relative": rel}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    log(f"SBS {sbs} {100 * per_set[sbs]['score']:+.2f}%, VBS {100 * vbs:+.2f}%, gap {summary['gap_points']:.2f} points; "
        f"rule-or-fixed VBS {100 * vbs_pair:+.2f}% (gap {summary['gap_rule_fixed_points']:.2f})")
    for k in SETS:
        v = per_set[k]
        log(f"  {k:13s} {100 * v['score']:+6.2f}%  mean cost {v['mean_cost']:.5f}  better {v['better']:3d} worse "
            f"{v['worse']:3d}  best in {v['best_in']:3d}")
    for kind, v in by_class.items():
        log(f"  best per {kind} ({v['classes']} classes): {100 * v['score']:+.2f}% (gain over SBS "
            f"{v['gain_over_sbs_points']:.2f} points) {v.get('choice', '')}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = choose_scenarios()
    log(f"{SET}: {len(specs)} scenarios, {len({Path(s).parent.name for s in specs})} buildings")
    paths, info = build_models()
    log("parameter sets: " + json.dumps(info["values"]))
    rows = {name: run_set(name, paths[name]) for name in SETS}
    summarize(specs, rows, info)


if __name__ == "__main__":
    main()
