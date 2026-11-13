"""Next per-room stages, automated (2026-11-01, docs/plan/geo-room-stages-2.md; user: items 1-5 as the next automated
stage, the generator included, the same rules). Starts from gm_tree_m1 (public 0.01124); every stage that passes
becomes the base of the next and goes to the Kaggle gate (pipeline/setup/geo_gate.py) in the background.

  n1  push the tree: data around the current tree (half the rooms follow it, half take the latin option) on search 40
      added to the latin data; corrections: a regularised tree, a finer tree
  n2  more buildings: search 40 + 80 scenarios of 34 generated TRAIN buildings never used before (configs/splits/
      v1-train-r1.json, official generator, seeds 10000-19999), latin + base runs; corrections: data around the current
      tree on all of them, a regularised tree
  n3  more options: + per-room comfort margins (both targets at the band edges; heating target +0.4 K; cooling target
      -0.4 K) = 8 options, on search 40 + 40 of the new scenarios; corrections: data around the current tree, regularised
  n4  time blocks (0-6, 6-10, 10-16, 16-24 h): options chosen per room and block; corrections as above
Pass: held-out 60 and INNER 29 each >= 0.5 points better than the base, S4 first 30 not more than 0.5 points worse.
State data/runs/b2rl/geo2/pipeline.json; log data/runs/b2rl/geo/pipeline.log (EVENT / RESULT / ERROR).

python3 -B pipeline/launch/launch.py pipeline/setup/geo_pipeline2.py [--workers 12]
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_stage_m as gsm  # noqa: E402
import room_stage as rs  # noqa: E402

GEO = ROOT / "data/runs/b2rl/geo"
OUT2 = GEO / "stage2"
STAGE_M = GEO / "stage_m"
STATE = ROOT / "data/runs/b2rl/geo2/pipeline.json"
NIGHT1 = ROOT / "data/runs/b2rl/search/models/model-night1_dcv07.json"
START = "gm_tree_m1"
MARGINS = [{"room_heat_margin_k": -0.1, "room_cool_margin_k": -0.1}, {"room_heat_margin_k": 0.4},
           {"room_cool_margin_k": 0.4}]
BLOCKS = [0, 6, 10, 16]
MAIN = {"features": gsm.TREE_FEATURES, "depth": 3, "min_leaf": 60, "min_groups": 6}
SMALL = {"features": gsm.SMALL_FEATURES, "depth": 2, "min_leaf": 150, "min_groups": 10}
FINE = {"features": gsm.TREE_FEATURES, "depth": 4, "min_leaf": 40, "min_groups": 6}


def save(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, STATE)


def fresh_scenarios(n_per_station=16):
    """Scenarios of generated TRAIN buildings that appear in neither search 40 nor held-out 60, station-stratified,
    one building after another (deterministic)."""
    rows = json.loads((ROOT / "configs/splits/v1-train-r1.json").read_text(encoding="utf-8"))["scenarios"]["train"]
    sets = json.loads((ROOT / "configs/b2rl/eval-sets.json").read_text(encoding="utf-8"))
    used_b = {s.split("/")[1] for s in sets[rs.SEARCH] + sets[rs.HOLDOUT]}
    fresh = sorted({f"train/{r['building']}/{r['weather']}" for r in rows if r["building"] not in used_b})
    out = []
    for station in ("atlanta", "denver", "fairbanks", "phoenix", "singapore"):
        pool = [s for s in fresh if s.split("/")[2].startswith(station)]
        by_b = {}
        for s in pool:
            by_b.setdefault(s.split("/")[1], []).append(s)
        picked, i = [], 0
        while len(picked) < n_per_station and any(by_b.values()):
            for b in sorted(by_b):
                if by_b[b] and len(picked) < n_per_station:
                    picked.append(by_b[b].pop(i % len(by_b[b]) if by_b[b] else 0))
            i += 1
        out += picked
    return out


def latin(prefix, options, blocks=None, k=None):
    k = k or len(options)
    return [(f"{prefix}{j}", {"mode": "latin", "seed": gsm.SEED, "shift": j, "options": options,
                              **({"blocks": blocks} if blocks else {})}) for j in range(k)]


def mix(prefix, tree, options, blocks=None):
    return [(f"{prefix}{j}", {"mode": "mix", "seed": gsm.SEED, "shift": j, "tree": tree, "options": options,
                              **({"blocks": blocks} if blocks else {})}) for j in range(len(options))]


def models(base, runs):
    return [rs.room_model(OUT2, base, n, p) for n, p in runs]


def run_stage(state, tag, base, specs, options, data_runs, attempts, workers, blocks=None):
    """attempts: list of (label, spec, extra runs recorded first or None)."""
    st = state["stages"].setdefault(tag, {"attempts": []})
    if st.get("done"):
        return st.get("passed")
    feats = rs.features(specs, workers)
    fixed0 = ("gm_fixed0", {"mode": "fixed", "option": 0, "options": options})   # the same run for any option list
    rs.record(OUT2, models(base, data_runs + [fixed0]), specs, workers, also=(STAGE_M,))
    runs = list(data_runs)
    done_labels = {a["label"] for a in st["attempts"]}
    for label, spec, extra in attempts:
        if extra:
            rs.record(OUT2, models(base, extra), specs, workers, also=(STAGE_M,))
            runs = runs + extra
        if label in done_labels:
            continue
        log_tag = f"{tag}{label}"
        result = rs.attempt(log_tag, base, state["base"], [OUT2, STAGE_M], runs, specs, feats, options, blocks, spec,
                            workers)
        result["label"] = label
        st["attempts"].append(result)
        save(state)
        if result["passed"]:
            st["done"], st["passed"] = True, result["name"]
            save(state)
            return result["name"]
    st["done"], st["passed"] = True, None
    save(state)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"base": START, "stages": {}}
    if state.get("done"):
        print("pipeline 2 already done")
        return
    base_model = json.loads(NIGHT1.read_text(encoding="utf-8"))          # schedule + rule config; no room policy
    sets = json.loads((ROOT / "configs/b2rl/eval-sets.json").read_text(encoding="utf-8"))
    search = sets[rs.SEARCH]
    fresh80 = state.setdefault("fresh80", fresh_scenarios())
    fresh40 = fresh80[::2]
    save(state)

    def current():
        m = json.loads((rs.MODELS / f"model-{state['base']}.json").read_text(encoding="utf-8"))
        return m["room_policy"]["tree"], m["room_policy"]["options"]

    try:
        rs.log("EVENT", f"pipeline 2 starts from {state['base']}; new generated scenarios {len(fresh80)} "
                        f"({len({s.split('/')[1] for s in fresh80})} buildings)")
        opts5 = current()[1]
        steps = [
            ("n1", lambda: run_stage(state, "n1", base_model, search, opts5, latin("gm_latin", opts5),
                                     [("a", MAIN, mix("gm_mixA", current()[0], opts5)), ("b", SMALL, None),
                                      ("c", FINE, None)], args.workers)),
            ("n2", lambda: run_stage(state, "n2", base_model, search + fresh80, opts5, latin("gm_latin", opts5),
                                     [("a", MAIN, None), ("b", MAIN, mix("gm_mixB", current()[0], opts5)),
                                      ("c", SMALL, None)], args.workers)),
            ("n3", lambda: run_stage(state, "n3", base_model, search + fresh40, opts5 + MARGINS,
                                     latin("gm_l8_", opts5 + MARGINS),
                                     [("a", MAIN, None), ("b", MAIN, mix("gm_mixC", current()[0], opts5 + MARGINS)),
                                      ("c", SMALL, None)], args.workers)),
        ]
        for tag, step in steps:
            passed = step()
            if passed:
                state["base"] = passed
                save(state)
            rs.log("EVENT", f"stage {tag} ends: passed {passed}; base now {state['base']}")
        options = current()[1]
        options = options if len(options) >= len(opts5 + MARGINS) else opts5 + MARGINS
        passed = run_stage(state, "n4", base_model, search + fresh40, options, latin("gm_lt_", options, BLOCKS),
                           [("a", MAIN, None), ("b", MAIN, mix("gm_mixD", current()[0], options, BLOCKS)),
                            ("c", SMALL, None)], args.workers, blocks=BLOCKS)
        if passed:
            state["base"] = passed
        state["done"] = True
        save(state)
        rs.log("EVENT", f"stage n4 ends: passed {passed}; pipeline 2 done, final base {state['base']}")
    except Exception:  # noqa: BLE001 - the watcher wakes the agent on this line
        rs.log("ERROR", traceback.format_exc().replace("\n", " | ")[-1500:])
        raise


if __name__ == "__main__":
    main()
