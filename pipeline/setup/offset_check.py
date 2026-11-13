"""Diagnostic 0a of docs/reference/notes/18-building-rl-what-made-it-learn.md §10 (user 2026-10-28 13:4x: "先修驗證方式
和做「是不是只學到固定偏移」的檢查"): did the v2 actor learn hour-to-hour choices, or one fixed set of parameters?

1. The actor (default: the v2 run's best checkpoint, update 100) runs whole episodes of train_val25 on the fast
   simulator with its mean action; every hourly decision's parameters, hour, weekday, outside air and occupants are
   recorded (data/runs/b2rl/offset-check/decisions-<name>.jsonl).
2. Per parameter: mean, standard deviation over the hours, share of hours at the rule's own value, and the
   correlation with hour of day (cosine and sine), outside air and occupants.
3. A constant-action model: the same network with a zero output layer and the bias at the action whose parameters
   are those means (layout v2's two-sided mapping inverted; the rule's defaults are the same in every building), so
   the rule runs with fixed parameters all month (model-fixed-<name>.json).
4. That fixed rule and the actor on INNER 29 whole episodes (b2rl.py eval, fast simulator), each paired with the rule
   (data/runs/b2rl/eval/curve/rule-inner-rom.json); summary-<name>.json.
The means come from TRAIN scenarios, so the INNER comparison is out of sample. Run with the project's venv:

python3 -B pipeline/launch/launch.py pipeline/setup/offset_check.py [model.json] [workers]"""
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
OUT = ROOT / "data/runs/b2rl/offset-check"
MODEL = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/runs/b2rl/v2-rom-s42/validation/model-u100.json")
# (an absolute path stays as it is; the workers change their working folder to the simulator's)
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
RULE_INNER = ROOT / "data/runs/b2rl/eval/curve/rule-inner-rom.json"


def episode(spec):
    """One whole train_val25 episode of the actor with every decision recorded (spawn worker)."""
    os.environ["SB_FASTSIM"] = "rom"
    from smart_building.learned.b2rl_env import setup_simulator, spec_dir
    setup_simulator(ROOT)
    from smart_building.evaluation.gateway_episode import run_gateway_episode
    from smart_building.policies.b2rl import b2rl_policy_class
    records, defaults = [], {}
    base = b2rl_policy_class(MODEL)

    class Recording(base):
        def get_action(self, obs):
            before = self.diagnostics["decisions"]
            action = super().get_action(obs)
            if self.slow is not None and self.diagnostics["decisions"] != before:
                o = self.slow.tracker.current
                if not defaults:
                    defaults.update(zip(self.slow.map.names, (float(x) for x in self.slow.map.default)))
                records.append({"minute": int(o.minute_of_day), "weekday": int(o.weekday),
                                "outside": float(np.mean(o.oat)) - 273.15, "occupants": float(o.occupants),
                                **{k: float(v) for k, v in self.slow.parameters.items()}})
            return action

    row = run_gateway_episode(spec_dir(spec), Recording, policy_key="b2rl", label=spec)
    return spec, -row["mean_reward"], records, defaults


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def inverse_action(p, d, low, high):
    """layout v2's two-sided mapping inverted: the action whose parameter is p (default d)."""
    if p >= d:
        return 0.0 if high == d else (p - d) / (high - d)
    return 0.0 if d == low else (p - d) / (d - low)


def eval_inner(model, out):
    command = ["python3", "-B", "scripts/b2rl.py", "eval", "--policy", str(model.relative_to(ROOT)), "--set", "inner",
               "--workers", str(WORKERS), "--fast", "rom", "--out", str(out.relative_to(ROOT))]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={k: v for k, v in os.environ.items() if k != "SB_FASTSIM"})
    if result.returncode:
        raise SystemExit(f"eval failed: {(result.stdout + result.stderr)[-600:]}")
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    rule = json.loads(RULE_INNER.read_text(encoding="utf-8"))["rows"]
    rel = [(rows[s]["cost"] - rule[s]["cost"]) / rule[s]["cost"] for s in sorted(rows)]
    diff = [rows[s]["cost"] - rule[s]["cost"] for s in sorted(rows)]
    return {"n": len(rel), "score": st.mean(rel), "se_score": st.stdev(rel) / math.sqrt(len(rel)),
            "mean_diff": st.mean(diff), "se": st.stdev(diff) / math.sqrt(len(diff)),
            "better": sum(d < 0 for d in diff), "worse": sum(d > 0 for d in diff)}


def main():
    from smart_building.policies.b2rl import LAYOUTS, load_model
    from smart_building.policies.g36 import SLOW_PARAMETER_RANGES
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{MODEL.parent.parent.name}-{MODEL.stem}"
    specs = json.loads((ROOT / "configs/b2rl/eval-sets.json").read_text(encoding="utf-8"))["train_val25"]
    model = load_model(MODEL)
    names = LAYOUTS[model.get("layout", "v1")]["parameters"]
    records, defaults, costs = [], {}, {}
    with mp.get_context("spawn").Pool(WORKERS, maxtasksperchild=1) as pool:
        for spec, cost, rows, d in pool.imap_unordered(episode, specs):
            costs[spec] = cost
            records += [{"spec": spec, **r} for r in rows]
            if d and defaults and d != defaults:
                raise SystemExit(f"the rule's defaults differ between buildings: {d} vs {defaults}")
            defaults.update(d)
            print(f"{spec}: {len(rows)} decisions, cost {cost:.5f}", flush=True)
    with (OUT / f"decisions-{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for r in records:
            stream.write(json.dumps(r) + "\n")
    hour = np.array([r["minute"] / 60.0 for r in records])
    stats = {}
    for k in names:
        values = np.array([r[k] for r in records])
        stats[k] = {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()),
                    "max": float(values.max()), "default": defaults[k],
                    "share_at_default": float(np.mean(np.isclose(values, defaults[k]))),
                    "corr_hour_cos": correlation(values, np.cos(2 * math.pi * hour / 24)),
                    "corr_hour_sin": correlation(values, np.sin(2 * math.pi * hour / 24)),
                    "corr_outside": correlation(values, np.array([r["outside"] for r in records])),
                    "corr_occupants": correlation(values, np.array([r["occupants"] for r in records]))}
    bias = [inverse_action(stats[k]["mean"], defaults[k], *SLOW_PARAMETER_RANGES[k]) for k in names]
    fixed = dict(model)
    last = model["layers"][-1]
    fixed["layers"] = model["layers"][:-1] + [{"weight": [[0.0] * len(last["weight"][0])] * len(names), "bias": bias}]
    fixed["source"] = {**model.get("source", {}), "offset_check": f"constant action = mean parameters of {name}"}
    fixed_path = OUT / f"model-fixed-{name}.json"
    fixed_path.write_text(json.dumps(fixed) + "\n", encoding="utf-8")
    summary = {"model": str(MODEL.relative_to(ROOT)), "decisions": len(records), "parameters": stats,
               "constant_action": dict(zip(names, bias)),
               "fixed_inner": eval_inner(fixed_path, OUT / f"inner-fixed-{name}.json"),
               "actor_inner": eval_inner(MODEL, OUT / f"inner-actor-{name}.json")}
    (OUT / f"summary-{name}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("fixed_inner", "actor_inner")}, indent=1))
    for k, v in stats.items():
        print(f"{k}: mean {v['mean']:.3f} (rule {v['default']:.3f}), std {v['std']:.3f}, at rule "
              f"{v['share_at_default']:.0%}, corr hour {v['corr_hour_cos']:+.2f}/{v['corr_hour_sin']:+.2f}, outside "
              f"{v['corr_outside']:+.2f}, occupants {v['corr_occupants']:+.2f}")


if __name__ == "__main__":
    main()
