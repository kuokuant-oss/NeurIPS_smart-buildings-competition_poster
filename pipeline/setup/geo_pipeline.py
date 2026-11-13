"""Per-room geometry stages, automated (2026-10-31, docs/plan/geo-room-stages.md; user: rules first, then machine
learning, one or two corrections per stage when the cost falls short, and the automation back).

Stage R (rules only): candidates = night1_dcv07 (the current best) + per-room G36Config switches (62.1 per-room minimum,
heating maximum by absolute exterior wall length, their scope and combinations). Each round runs every candidate on
search 40 (train_breakdown40, whole episodes, fast simulator); the two best that beat the current best go on to held-out
60 (train_ceiling_rest60) and the leaderboard estimate (s4_first). A candidate passes when search 40 <= -1.0%, held-out
60 <= -0.7% and s4_first <= +0.3% (mean cost against night1_dcv07's existing rows). Round 1 is fixed; when nothing
passes, round 2 (first correction) moves the best one notch and combines the best of each mechanism; round 3 (second
correction) tries the prepared alternatives (scope core / perimeter, other wall lengths). After that the stage ends
with its best (or none).
Stage M (machine learning): pipeline/setup/geo_stage_m.py <base model> <base name> runs with its own corrections and
writes data/runs/b2rl/geo/stage_m/result.json; the pipeline waits for that script to exist.
A passing candidate gets INNER 29 on the fast simulator (for the Kaggle consistency check) and then
pipeline/setup/geo_gate.py (Kaggle official gates, billing check, submission under the standing rules) in the background.

Log lines (data/runs/b2rl/geo/pipeline.log): RESULT per evaluation, EVENT per stage step (the hourly watcher wakes on
EVENT and ERROR). State: data/runs/b2rl/geo/pipeline.json (resumable; finished evaluations are reused).

python3 -B pipeline/launch/launch.py pipeline/setup/geo_pipeline.py [--workers 12]
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "data/runs/b2rl/geo"
STATE = OUT / "pipeline.json"
LOG = OUT / "pipeline.log"
BASE_MODEL = ROOT / "data/runs/b2rl/search/models/model-night1_dcv07.json"
REF = {"train_breakdown40": "data/runs/b2rl/search/train_breakdown40/at_dcvnight1_001.json",
       "train_ceiling_rest60": "data/runs/b2rl/search/train_ceiling_rest60/at_dcvnight1_001.json",
       "s4_first": "data/runs/b2rl/eval/curve/night1_dcv07-s4_first-rom.json",
       "inner": "data/runs/b2rl/eval/curve/search-night1_dcv07-inner-rom.json"}
SEARCH, HOLDOUT, S4 = "train_breakdown40", "train_ceiling_rest60", "s4_first"
EXPECT = {SEARCH: -1.0, HOLDOUT: -0.7, S4: 0.3}
RA = {"1": {"zone_min_ra_lps_m2": 0.3, "zone_min_rp_lps_m2": 0.125},
      "15": {"zone_min_ra_lps_m2": 0.45, "zone_min_rp_lps_m2": 0.1875},
      "2": {"zone_min_ra_lps_m2": 0.6, "zone_min_rp_lps_m2": 0.25}}
HW = {f"{ref}p{str(p).replace('.', '')}": {"zone_heat_wall_ref_m": float(ref), "zone_heat_wall_power": p}
      for ref in (20, 30, 50) for p in (0.5, 1.0)}
# The per-room minimum is only eligible in its code-compliant form (zone_min_oa_divide: the 62.1 outdoor air divided by
# the AHU's outdoor-air fraction, G36 Vmin*). Supply air = the 62.1 rate (RA without the division) gives each room about
# a third of the code's outdoor air; it was run once (gr_ra1, search 40 -6.52%, 2026-10-31 20:16) and stays a labelled
# upper bound, never gated or submitted.
V62 = {**RA["1"], "zone_min_oa_divide": True}
ROUND1 = {"gr_v62": V62, "gr_v62core": {**V62, "zone_min_scope": "core"},
          "gr_v62perim": {**V62, "zone_min_scope": "perimeter"}, "gr_hw30p05": HW["30p05"],
          "gr_hw50p10": HW["50p10"], "gr_v62_hw30p05": {**V62, **HW["30p05"]}}
ROUND3_EXTRA = {"gr_hw20p10": HW["20p10"], "gr_hw20p05": HW["20p05"]}


def eligible(extra):
    """A per-room minimum must be the code-compliant one (outdoor air -> primary air)."""
    return not extra.get("zone_min_ra_lps_m2") or bool(extra.get("zone_min_oa_divide"))
WORKERS = 12


def log(kind, text):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind} {text}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"stage": "R", "rounds": {}}


def save_state(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, STATE)


def base_spec():
    source = json.loads(BASE_MODEL.read_text(encoding="utf-8"))["source"]["search"]
    return json.loads(source[source.index("{"):])


def build(name, extra):
    from search import build_schedule
    spec = copy.deepcopy(base_spec())
    spec["rule_config"] = {**spec["rule_config"], **extra}
    path, _ = build_schedule(name, spec)
    return path


def cost(row):
    return row["cost"] if "cost" in row else -row["mean_reward"]


def evaluate(name, model, subset, workers):
    out = OUT / subset / f"{name}.json"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "SB_FASTSIM"}
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(["nice", "-n", "5", "python3", "-B", "scripts/b2rl.py", "eval", "--policy",
                                 str(model.relative_to(ROOT)), "--set", subset, "--workers", str(workers), "--fast",
                                 "rom", "--out", str(out.relative_to(ROOT))], cwd=ROOT, env=env, capture_output=True,
                                text=True)
        if result.returncode:
            raise RuntimeError(f"{subset} {name}: eval failed: {(result.stdout + result.stderr).strip()[-600:]}")
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    ref = json.loads((ROOT / REF[subset]).read_text(encoding="utf-8"))["rows"]
    common = [s for s in ref if s in rows]
    rel = [cost(rows[s]) / cost(ref[s]) - 1 for s in common]
    r = {"mean": 100 * (st.mean(cost(rows[s]) for s in common) / st.mean(cost(ref[s]) for s in common) - 1),
         "per_scenario": 100 * st.mean(rel), "se": 100 * st.stdev(rel) / math.sqrt(len(rel)),
         "better": sum(x < 0 for x in rel), "worse": sum(x > 0 for x in rel), "n": len(common)}
    log("RESULT", f"{subset} {name}: mean cost {r['mean']:+.2f}% vs night1_dcv07 (per scenario {r['per_scenario']:+.2f}%, "
                  f"SE {r['se']:.2f}, {r['better']} better / {r['worse']} worse of {r['n']})")
    return r


def passes(res):
    return (SEARCH in res and HOLDOUT in res and S4 in res and res[SEARCH]["mean"] <= EXPECT[SEARCH]
            and res[HOLDOUT]["mean"] <= EXPECT[HOLDOUT] and res[S4]["mean"] <= EXPECT[S4])


def run_round(state, tag, candidates, workers):
    rnd = state["rounds"].setdefault(tag, {"candidates": candidates, "results": {}})
    save_state(state)
    log("EVENT", f"stage R {tag} starts: {', '.join(candidates)}")
    for name, extra in candidates.items():
        res = rnd["results"].setdefault(name, {})
        if SEARCH not in res:
            res[SEARCH] = evaluate(name, build(name, extra), SEARCH, workers)
            save_state(state)
    ranked = sorted(rnd["results"], key=lambda n: rnd["results"][n][SEARCH]["mean"])
    for name in [n for n in ranked[:2] if rnd["results"][n][SEARCH]["mean"] < 0]:
        res = rnd["results"][name]
        model = ROOT / f"data/runs/b2rl/search/models/model-{name}.json"
        for subset in (HOLDOUT, S4):
            if subset not in res:
                res[subset] = evaluate(name, model, subset, workers)
                save_state(state)
    passed = [n for n in ranked if passes(rnd["results"][n]) and eligible(rnd["candidates"][n])]
    summary = "; ".join(f"{n} {rnd['results'][n][SEARCH]['mean']:+.2f}%"
                        + (f" / held-out {rnd['results'][n][HOLDOUT]['mean']:+.2f}% / S4 {rnd['results'][n][S4]['mean']:+.2f}%"
                           if HOLDOUT in rnd["results"][n] else "") for n in ranked)
    log("EVENT", f"stage R {tag} done (search 40 / held-out 60 / S4 first 30 vs night1_dcv07): {summary}; "
                 f"{'PASSED ' + passed[0] if passed else 'nothing reaches the expectation'}")
    return passed[0] if passed else None


def all_results(state):
    out = {}
    for rnd in state["rounds"].values():
        for name, res in rnd["results"].items():
            out[name] = (rnd["candidates"][name], res)
    return out


def correction1(state):
    """Round 2: the best heating-maximum variant one notch away (length or power), and the best per-room minimum
    (code-compliant, any scope) combined with the best heating-maximum variant."""
    done = {n: v for n, v in all_results(state).items() if eligible(v[0])}
    ranked = sorted(done, key=lambda n: done[n][1][SEARCH]["mean"])
    ra = [n for n in ranked if "zone_min_ra_lps_m2" in done[n][0] and "zone_heat_wall_ref_m" not in done[n][0]]
    hw = [n for n in ranked if "zone_heat_wall_ref_m" in done[n][0] and "zone_min_ra_lps_m2" not in done[n][0]]
    out = {}
    if hw:
        c = done[hw[0]][0]
        for key, v in HW.items():
            if (v["zone_heat_wall_ref_m"] == c["zone_heat_wall_ref_m"]) != (v["zone_heat_wall_power"] == c["zone_heat_wall_power"]):
                out[f"gr_hw{key}"] = v
    if ra and hw:
        out[f"{ra[0]}_{hw[0][3:]}"] = {**done[ra[0]][0], **done[hw[0]][0]}
    return {k: v for k, v in out.items() if k not in all_results(state)}


def correction2(state):
    """Round 3: the prepared alternatives: other wall lengths, and each scope of the per-room minimum with the best
    heating-maximum variant."""
    done = {n: v for n, v in all_results(state).items() if eligible(v[0])}
    ranked = sorted(done, key=lambda n: done[n][1][SEARCH]["mean"])
    hw = [n for n in ranked if "zone_heat_wall_ref_m" in done[n][0] and "zone_min_ra_lps_m2" not in done[n][0]]
    best_hw = done[hw[0]][0] if hw else HW["30p05"]
    out = dict(ROUND3_EXTRA)
    for scope in ("core", "perimeter"):
        out[f"gr_v62{scope[:5]}_hwbest"] = {**V62, "zone_min_scope": scope, **best_hw}
    return {k: v for k, v in out.items() if k not in all_results(state)}


def launch_gate(state, name, workers):
    model = ROOT / f"data/runs/b2rl/search/models/model-{name}.json"
    extra, res = all_results(state)[name]
    if not eligible(extra):
        log("ERROR", f"{name} uses the non-compliant per-room minimum (upper bound only); no gate")
        return
    if "inner" not in res:
        res["inner"] = evaluate(name, model, "inner", workers)
        save_state(state)
    gate_log = OUT / f"gate-{name}.log"
    subprocess.Popen(["python3", "-B", str(ROOT / "pipeline/setup/geo_gate.py"), name, str(model.relative_to(ROOT))],
                     cwd=ROOT, stdout=open(gate_log, "a", encoding="utf-8"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    state.setdefault("gates", {})[name] = {"started": time.strftime("%Y-%m-%d %H:%M"), "log": str(gate_log.relative_to(ROOT))}
    save_state(state)
    log("EVENT", f"gate for {name} started in the background (Kaggle official INNER 29 + S4 first 30, then billing "
                 f"check and submission under the standing rules); log {gate_log.relative_to(ROOT)}")


def stage_r(state, workers):
    rounds = [("r1", lambda s: ROUND1), ("r2", correction1), ("r3", correction2)]
    for tag, make in rounds:
        if state["rounds"].get(tag, {}).get("done"):
            if state["rounds"][tag].get("passed"):
                return state["rounds"][tag]["passed"]
            continue
        candidates = state["rounds"].get(tag, {}).get("candidates") or make(state)
        if not candidates:
            log("EVENT", f"stage R {tag}: no new candidates")
            state["rounds"][tag] = {"candidates": {}, "results": {}, "done": True}
            save_state(state)
            continue
        passed = run_round(state, tag, candidates, workers)
        state["rounds"][tag]["done"], state["rounds"][tag]["passed"] = True, passed
        save_state(state)
        if passed:
            return passed
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    state = load_state()
    try:
        if state["stage"] == "R":
            passed = stage_r(state, args.workers)
            done = all_results(state)
            held = {n: r for n, (e, r) in done.items() if HOLDOUT in r and eligible(e)}
            best = min(held, key=lambda n: held[n][HOLDOUT]["mean"]) if held else None
            state["stage_r"] = {"passed": passed, "best_holdout": best,
                                "best_holdout_result": held.get(best) if best else None}
            if passed:
                launch_gate(state, passed, args.workers)
                base = passed
            elif best and held[best][HOLDOUT]["mean"] < 0 and held[best][S4]["mean"] <= EXPECT[S4]:
                base = best
            else:
                base = "night1_dcv07"
            state["stage"], state["stage_m_base"] = "M", base
            save_state(state)
            log("EVENT", f"stage R ends: passed {passed}; best on held-out {best}; stage M starts from {base}")
        if state["stage"] == "M":
            script = ROOT / "pipeline/setup/geo_stage_m.py"
            while not script.is_file():
                log("EVENT", "stage M waits for pipeline/setup/geo_stage_m.py") if not state.get("m_waiting") else None
                state["m_waiting"] = True
                save_state(state)
                time.sleep(600)
            base = state["stage_m_base"]
            model = BASE_MODEL if base == "night1_dcv07" else ROOT / f"data/runs/b2rl/search/models/model-{base}.json"
            log("EVENT", f"stage M runs from {base}")
            code = subprocess.run(["python3", "-B", str(ROOT / "pipeline/launch/launch.py"), str(script),
                                   str(model.relative_to(ROOT)), base, "--workers", str(args.workers)], cwd=ROOT).returncode
            result_file = OUT / "stage_m/result.json"
            result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else None
            if code or result is None:
                raise RuntimeError(f"stage M ended with code {code} and no result")
            state["stage_m"] = result
            if result.get("passed"):
                launch_gate(state, result["passed"], args.workers)
            state["stage"] = "done"
            save_state(state)
            log("EVENT", f"stage M ends: {json.dumps(result)[:400]}; pipeline done")
    except Exception:  # noqa: BLE001 - the watcher wakes the agent on this line
        log("ERROR", traceback.format_exc().replace("\n", " | ")[-1500:])
        raise


if __name__ == "__main__":
    main()
