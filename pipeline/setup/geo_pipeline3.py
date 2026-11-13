"""Pipeline 3: group-factor stages learned from building totals, with a diagnosis before every correction
(2026-11-01, docs/plan/geo-room-stages-3.md; user: after the analysis go straight back to automation, and the two
corrections must really be tuned from the analysis, not a formality).

Stages (each: a 12-run Plackett-Burman design on 80 scenarios -> factor effects from building totals -> the main
attempt (factors whose mean effect is below zero by more than one standard error) -> closed loop vs the current base):
  T1  per-room minimum airflow (code form) per size x core/exterior cell group, wall-length heating per size group
  T2  comfort margins per group, on the best T1 map
  T3  time blocks (night / morning / midday / evening), on the best map so far
Pass (unchanged): held-out 60 and INNER 29 each >= 0.5 points better than the base, S4 first 30 not more than 0.5
points worse; a pass goes to the Kaggle gate (geo_gate.py) and becomes the base.

After a failed attempt the pipeline records the candidate on the design scenarios, writes a diagnosis
(data/runs/b2rl/geo3/diagnosis-<stage>-<k>.json: predicted vs realised on the same scenarios, by part and station;
effect heterogeneity by station and building traits; closed loop by station; where it differs from gm_tree_m1 by cell
group), logs an EVENT and waits for data/runs/b2rl/geo3/correction-<stage>-<k>.json, written by the agent from that
diagnosis: {"why": ..., "script": "pipeline/setup/geo3_corr/<file>.py"} (build(ctx) -> {"name", "map", ...}) or
{"why": ..., "skip": true}. Two corrections per stage at most.

python3 -B pipeline/launch/launch.py pipeline/setup/geo_pipeline3.py [--workers 12]"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_stage_m as gsm  # noqa: E402
import group_stage as gs  # noqa: E402
import room_stage as rs  # noqa: E402

GEO = ROOT / "data/runs/b2rl/geo"
STATE_DIR = ROOT / "data/runs/b2rl/geo3"
STATE = STATE_DIR / "pipeline.json"
DIRS = [gs.OUT3, GEO / "stage2", GEO / "stage_m"]
NIGHT1 = ROOT / "data/runs/b2rl/search/models/model-night1_dcv07.json"
V62 = {"zone_min_ra_lps_m2": 0.3, "zone_min_rp_lps_m2": 0.125, "zone_min_oa_divide": True}
W20 = {"zone_heat_wall_ref_m": 20.0, "zone_heat_wall_power": 1.0}
SM, LX = ("s", "m"), ("l", "x")
NEUTRAL = {"zone_min_ra_lps_m2": 0.0, "zone_min_rp_lps_m2": 0.0, "zone_min_oa_divide": False,
           "zone_heat_wall_ref_m": 0.0, "room_heat_margin_k": 0.0, "room_cool_margin_k": 0.0}


def cells(areas, walls):
    return [[a, w] for a in areas for w in walls]


def stage_factors(tag):
    if tag == "T1":
        f = []
        for a in gs.AREAS:
            f.append({"name": f"min_{a}_core", "cells": cells([a], ["core"]), "delta": V62})
            f.append({"name": f"min_{a}_ext", "cells": cells([a], ["perim", "long"]), "delta": V62})
        f.append({"name": "wallheat_sm", "cells": cells(SM, ["long"]), "delta": W20})
        f.append({"name": "wallheat_lx", "cells": cells(LX, ["long"]), "delta": W20})
        return f, None
    if tag == "T2":
        f = []
        for name, group in (("sm", SM), ("lx", LX)):
            f += [{"name": f"heat_edge_core_{name}", "cells": cells(group, ["core"]), "delta": {"room_heat_margin_k": -0.1}},
                  {"name": f"heat_warm_ext_{name}", "cells": cells(group, ["perim", "long"]),
                   "delta": {"room_heat_margin_k": 0.4}},
                  {"name": f"cool_edge_{name}", "cells": cells(group, ["core", "perim"]), "delta": {"room_cool_margin_k": -0.1}},
                  {"name": f"cool_inner_long_{name}", "cells": cells(group, ["long"]), "delta": {"room_cool_margin_k": 0.4}}]
        return f, None
    if tag == "T3":
        f = []
        for name, group in (("sm", SM), ("lx", LX)):
            f += [{"name": f"morning_off_{name}", "cells": cells(group, gs.WALLS), "delta": NEUTRAL, "blocks": [1]},
                  {"name": f"evening_off_{name}", "cells": cells(group, gs.WALLS), "delta": NEUTRAL, "blocks": [3]},
                  {"name": f"morning_warm_{name}", "cells": cells(group, ["perim", "long"]),
                   "delta": {"room_heat_margin_k": 0.4}, "blocks": [1]},
                  {"name": f"midday_edge_{name}", "cells": cells(group, gs.WALLS),
                   "delta": {"room_heat_margin_k": -0.1, "room_cool_margin_k": -0.1}, "blocks": [2]}]
        return f, [0, 6, 10, 16]
    raise ValueError(tag)


def save(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, STATE)


def model_for(name, cell_map, blocks, base_model):
    policy = gs.policy_of(cell_map, blocks)
    path = rs.room_model(gs.OUT3, base_model, name, policy)
    return path


def final_model(name, cell_map, blocks, base_model):
    path = model_for(name, cell_map, blocks, base_model)
    final = rs.MODELS / f"model-{name}.json"
    final.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return final


def score(res):
    return (res[rs.HOLDOUT]["mean"] + res[rs.INNER]["mean"]) / 2.0


class Stage:
    def __init__(self, state, tag, specs, feats, base_model, workers):
        self.state, self.tag, self.specs, self.feats, self.base_model, self.workers = \
            state, tag, specs, feats, base_model, workers
        self.factors, self.blocks = stage_factors(tag)
        self.nb = len(self.blocks) if self.blocks else 1
        self.st = state["stages"].setdefault(tag, {"attempts": [], "diagnoses": []})
        self.base_map = state.get("build_on") or {}
        self.st.setdefault("base_map", self.base_map)
        self.base_map = self.st["base_map"]
        self.all_off_is_night1 = gs.is_empty(self.base_map)

    def record(self, models):
        rs.record(gs.OUT3, models, self.specs, self.workers, also=tuple(DIRS[1:]) + (gs.DIAG,))

    def design(self):
        runs = gs.design_runs(f"gt_{self.tag.lower()}_pb", self.base_map, self.factors, self.nb)
        models = [model_for(n, m, self.blocks, self.base_model) for n, lv, m in runs
                  if not (self.all_off_is_night1 and all(v < 0 for v in lv))]
        self.record(models)
        self.runs = runs
        self.base_run = "gm_fixed0" if self.all_off_is_night1 else runs[-1][0]
        self.per = gs.estimate(self.specs, self.feats, self.factors, runs, DIRS, self.base_run)
        self.summary = gs.summarise(self.per, self.factors)
        lob = gs.lobo(self.per, self.factors)
        rs.log("EVENT", f"{self.tag} design effects on {len(self.per)} scenarios (cost change when on, % of the stage "
                        f"base): " + ", ".join(f"{k} {100 * v['mean']:+.2f}±{100 * v['se']:.2f}"
                                               for k, v in self.summary.items())
               + f"; leave-buildings-out predicted {100 * lob['lobo_predicted']:+.2f}% (per-scenario oracle "
                 f"{100 * lob['per_scenario_oracle']:+.2f}%)")
        self.st["design"] = {"summary": self.summary, "lobo": lob,
                             "parts": {p: gs.summarise(self.per, self.factors, p) for p in gs.PARTS}}
        save(self.state)

    def attempt(self, label, cell_map, why, predicted=None):
        name = f"gt_{self.tag.lower()}{label}"
        model = final_model(name, cell_map, self.blocks, self.base_model)
        rs.log("EVENT", f"{self.tag}{label} attempt ({why}): predicted {100 * (predicted or 0):+.2f}% vs the stage "
                        f"base on the design scenarios; options {len(gs.policy_of(cell_map, self.blocks)['options'])}")
        res, gain, passed = rs.closed_loop(name, model, self.state["base"], self.workers)
        a = {"label": label, "name": name, "why": why, "map": cell_map, "predicted": predicted,
             "closed_loop": res, "against_base": gain, "passed": passed}
        self.st["attempts"].append(a)
        best = self.state.get("best_attempt")
        if best is None or score(res) < best["score"]:
            self.state["best_attempt"] = {"name": name, "score": score(res), "map": cell_map, "blocks": self.blocks}
        save(self.state)
        if passed:
            gsm.launch_gate(name, model, self.workers)
        return a

    def diagnose(self, k, a):
        """Predicted vs realised on the design scenarios, effect heterogeneity, closed loop by station, and where
        the candidate differs from gm_tree_m1 by cell group."""
        name = a["name"]
        self.record([gs.OUT3 / "models" / f"model-{name}.json"])
        levels = a.get("levels")
        rows = []
        for spec in self.specs:
            cand = rs.find_record(DIRS + [gs.DIAG], name, spec)
            base = rs.find_record(DIRS + [gs.DIAG], self.base_run, spec)
            m1 = rs.find_record([gs.DIAG], "gm_tree_m1", spec)
            if cand is None or base is None or spec not in self.per:
                continue
            norm = base["summary"]["cost"]
            row = {"spec": spec, **gs.scenario_traits(spec, self.feats, DIRS),
                   "actual": cand["summary"]["cost"] / norm - 1.0,
                   "actual_parts": {p: (float(np.sum(cand["zone_summary"][p])) - float(np.sum(base["zone_summary"][p])))
                                    / norm for p in gs.PARTS}}
            lv = (a.get("levels_by_spec") or {}).get(spec, levels)      # a climate map: each scenario's own bin
            if lv is not None:
                row["predicted"] = sum(e for e, on in zip(self.per[spec]["total"], lv) if on > 0 and e is not None)
                row["predicted_parts"] = {p: sum(e for e, on in zip(self.per[spec][p], lv) if on > 0 and e is not None)
                                          for p in gs.PARTS}
            if m1 is not None:
                row["vs_m1"] = cand["summary"]["cost"] / m1["summary"]["cost"] - 1.0
                cc, cm = gs.cell_costs(cand, spec, self.feats, norm), gs.cell_costs(m1, spec, self.feats, norm)
                row["vs_m1_cells"] = {c: cc.get(c, 0.0) - cm.get(c, 0.0) for c in set(cc) | set(cm)}
                row["vs_m1_parts"] = {p: (float(np.sum(cand["zone_summary"][p])) - float(np.sum(m1["zone_summary"][p])))
                                      / norm for p in gs.PARTS}
            rows.append(row)
        d = {"stage": self.tag, "k": k, "attempt": name, "scenarios": len(rows)}
        act = np.array([r["actual"] for r in rows])
        d["realised mean %"] = round(100 * float(act.mean()), 2)
        if all("predicted" in r for r in rows) and rows:
            pred = np.array([r["predicted"] for r in rows])
            d["predicted mean %"] = round(100 * float(pred.mean()), 2)
            d["slope actual on predicted"] = round(float(np.polyfit(pred, act, 1)[0]), 3) if pred.std() > 0 else None
            d["correlation"] = round(float(np.corrcoef(pred, act)[0, 1]), 3) if pred.std() > 0 else None
            d["by part predicted / realised %"] = {p: [round(100 * float(np.mean([r["predicted_parts"][p] for r in rows])), 2),
                                                       round(100 * float(np.mean([r["actual_parts"][p] for r in rows])), 2)]
                                                   for p in gs.PARTS}
        stations = sorted({r["station"] for r in rows})
        d["by station realised (predicted) %"] = {
            s: [round(100 * float(np.mean([r["actual"] for r in rows if r["station"] == s])), 2),
                round(100 * float(np.mean([r.get("predicted", 0.0) for r in rows if r["station"] == s])), 2)]
            for s in stations}
        m1rows = [r for r in rows if "vs_m1" in r]
        if m1rows:
            d["vs gm_tree_m1 on design scenarios %"] = round(100 * float(np.mean([r["vs_m1"] for r in m1rows])), 2)
            d["vs gm_tree_m1 by part %"] = {p: round(100 * float(np.mean([r["vs_m1_parts"][p] for r in m1rows])), 2)
                                            for p in gs.PARTS}
            cellsum = {}
            for r in m1rows:
                for c, v in r["vs_m1_cells"].items():
                    cellsum.setdefault(c, []).append(v)
            d["vs gm_tree_m1 by cell group (allocated, % of base)"] = {
                c: round(100 * float(np.sum(v)) / len(m1rows), 3) for c, v in sorted(cellsum.items())}
            d["vs gm_tree_m1 by station %"] = {
                s: round(100 * float(np.mean([r["vs_m1"] for r in m1rows if r["station"] == s])), 2) for s in stations}
        het = {}
        traits = {r["spec"]: r for r in rows}
        for i, fac in enumerate(self.factors):
            vals = [(self.per[s]["total"][i], traits[s]) for s in self.per if s in traits and
                    self.per[s]["total"][i] is not None]
            if len(vals) < 5:
                continue
            e = np.array([v for v, _ in vals])
            het[fac["name"]] = {
                "by station %": {s: round(100 * float(np.mean([v for v, t in vals if t["station"] == s])), 2)
                                 for s in stations if any(t["station"] == s for _, t in vals)},
                "corr with": {k: round(float(np.corrcoef(e, [t[k] for _, t in vals])[0, 1]), 2)
                              for k in ("oat_mean", "small_share", "long_share", "ahu_size", "rooms")
                              if np.std([t[k] for _, t in vals]) > 0}}
        d["factor heterogeneity"] = het
        d["closed loop by station vs base (points)"] = {s: gs.by_station(name, self.state["base"], s)
                                                        for s in (rs.HOLDOUT, rs.INNER, rs.S4)}
        path = STATE_DIR / f"diagnosis-{self.tag}-{k}.json"
        path.write_text(json.dumps({"diagnosis": d, "rows": rows}, indent=1) + "\n", encoding="utf-8")
        self.st["diagnoses"].append(str(path.relative_to(ROOT)))
        save(self.state)
        brief = {k2: d[k2] for k2 in ("realised mean %", "predicted mean %", "slope actual on predicted",
                                      "vs gm_tree_m1 on design scenarios %") if k2 in d}
        rs.log("EVENT", f"{self.tag} diagnosis {k} ready ({path.relative_to(ROOT)}): {json.dumps(brief)}; waiting for "
                        f"data/runs/b2rl/geo3/correction-{self.tag}-{k}.json")
        return d

    def wait_correction(self, k):
        path = STATE_DIR / f"correction-{self.tag}-{k}.json"
        while not path.is_file():
            time.sleep(60)
        spec = json.loads(path.read_text(encoding="utf-8"))
        rs.log("EVENT", f"{self.tag} correction {k}: {spec.get('why', '')[:400]}")
        return spec

    def run_correction(self, k, spec):
        if spec.get("skip"):
            return None
        mod_path = ROOT / spec["script"]
        mspec = importlib.util.spec_from_file_location(f"corr_{self.tag}_{k}", mod_path)
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
        ctx = SimpleNamespace(stage=self, k=k, gs=gs, rs=rs, specs=self.specs, feats=self.feats,
                              factors=self.factors, per=self.per, summary=self.summary, base_map=self.base_map,
                              blocks=self.blocks, runs=self.runs, dirs=DIRS, model_for=model_for,
                              base_model=self.base_model, record=self.record, log=rs.log, state=self.state)
        out = mod.build(ctx)
        if out.get("blocks", "same") != "same":
            self.blocks = out["blocks"]
        a = self.attempt(out.get("label", "bc"[k - 1]), out["map"], spec.get("why", "")[:200], out.get("predicted"))
        a["levels"] = out.get("levels")
        a["levels_by_spec"] = out.get("levels_by_spec")
        save(self.state)
        return a

    def run(self):
        if self.st.get("done"):
            return self.st.get("passed")
        self.design()
        if not self.st["attempts"]:
            levels = gs.choose(self.summary, self.factors)
            cell_map = gs.compose(self.base_map, self.factors, levels, self.nb)
            a = self.attempt("a", cell_map, "main rule: factors below zero by more than one standard error",
                             gs.predicted(self.per, levels))
            a["levels"] = levels
            save(self.state)
        a = self.st["attempts"][-1]
        while not a["passed"] and len(self.st["attempts"]) <= 2:
            k = len(self.st["attempts"])
            if len(self.st["diagnoses"]) < k:
                self.diagnose(k, a)
            spec = self.wait_correction(k)
            nxt = self.run_correction(k, spec)
            if nxt is None:
                break
            a = nxt
        passed = next((x["name"] for x in self.st["attempts"] if x["passed"]), None)
        self.st["done"], self.st["passed"] = True, passed
        if passed:
            self.state["base"] = passed
            self.state["build_on"] = next(x["map"] for x in self.st["attempts"] if x["passed"])
        elif self.state.get("best_attempt") and not self.state["best_attempt"].get("blocks"):
            self.state["build_on"] = self.state["best_attempt"]["map"]       # the next stage builds on the best map
        save(self.state)
        rs.log("EVENT", f"stage {self.tag} ends: passed {passed}; base now {self.state['base']}")
        return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"base": "gm_tree_m1", "stages": {}}
    if state.get("done"):
        print("pipeline 3 already done")
        return
    sets = json.loads((ROOT / "configs/b2rl/eval-sets.json").read_text(encoding="utf-8"))
    fresh80 = json.loads((ROOT / "data/runs/b2rl/geo2/pipeline.json").read_text(encoding="utf-8"))["fresh80"]
    specs = state.setdefault("specs", sets[rs.SEARCH] + fresh80[::2])
    save(state)
    base_model = json.loads(NIGHT1.read_text(encoding="utf-8"))
    feats = rs.features(specs, args.workers)
    try:
        rs.log("EVENT", f"pipeline 3 starts (group factors from building totals) from base {state['base']} on "
                        f"{len(specs)} design scenarios")
        for tag in ("T1", "T2", "T3"):
            Stage(state, tag, specs, feats, base_model, args.workers).run()
        state["done"] = True
        save(state)
        rs.log("EVENT", f"pipeline 3 done, final base {state['base']}")
    except Exception:  # noqa: BLE001 - the watcher wakes the agent on this line
        rs.log("ERROR", traceback.format_exc().replace("\n", " | ")[-1500:])
        raise


if __name__ == "__main__":
    main()
