"""Pipeline 4: richer per-room options (U1) and periods of the day (U2), tree step on building totals
(2026-11-02, docs/plan/geo-room-stages-4.md). Unattended, about 15 hours, deadline 2026-11-03 07:30 local.

Per stage: a balanced design (Plackett-Burman, 12 runs) on the real search scenarios plus one
scenario of each of the 26 generated buildings, per-scenario factor effects from the building totals, then the tree
step (group_stage4.py): per option type a shallow honest tree (a) full data, (b) bagged over buildings, (c) regularised
by an inner leave-buildings-out choice of the minimum leaf, against the main-effects rule; compared by leave-buildings-out
on the real buildings only (also trained without the generated ones); the best is the attempt's candidate, closed loop on
held-out 60 / INNER 29 / S4 first 30 (real scenarios only) against the stage base, pass = rooms.closed_loop's rule. Two
preset corrections per stage, no waiting:
  U1b noise: keep the factors negative in >= 2 of 3 building thirds, confirm them on the other 54 generated scenarios,
  re-choose; U1c climate: first split on the episode running-mean outside air, then cells, on the survivors.
  U2b edges: periods with wider edges (2 h warm-up / 3 h late; clock and schedule hours coincide, every scenario is 6-19)
  and a new design; U2c warm-up only, the larger levels.
A pass goes to the Kaggle gate (geo_gate.py, auto-submit under the standing rules) and becomes the next base. Steps are
skipped when they cannot end before the deadline. Resumable (records on disk, state in geo4/pipeline.json).

python3 -B pipeline/launch/launch.py pipeline/setup/geo_pipeline4.py [--workers 12]"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_stage_m as gsm  # noqa: E402
import group_stage as gs  # noqa: E402
import group_stage4 as g4  # noqa: E402
import room_stage as rs  # noqa: E402

GEO = ROOT / "data/runs/b2rl/geo"
STATE_DIR = ROOT / "data/runs/b2rl/geo4"
STATE, SUMMARY = STATE_DIR / "pipeline.json", STATE_DIR / "summary.json"
DIRS = [g4.OUT4, gs.OUT3, GEO / "stage2", GEO / "stage_m", gs.DIAG]
NIGHT1 = ROOT / "data/runs/b2rl/search/models/model-night1_dcv07.json"
SETUP = ROOT / "pipeline/setup"
DEADLINE = datetime.datetime(2026, 11, 3, 7, 30)
GATE_WAIT_UNTIL = datetime.datetime(2026, 11, 3, 14, 0)
EP_PER_MIN, LOOP_MIN = 5.0, 22.0                  # pipeline 3: 960 episodes in 178 min; a closed loop 17-20 min
WIDE = {"warm_h": 2.0, "late_h": 3.0}
NARROW = {"warm_h": 1.0, "late_h": 2.0}


def log(kind, text):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind} pipeline4: {text}"
    print(line, flush=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (STATE_DIR / "pipeline.log", GEO / "pipeline.log"):
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def has_record(name, spec):
    return any((d / "records" / f"{name}__{spec.replace('/', '__')}.json").is_file() for d in DIRS)


class Pipeline:
    def __init__(self, workers):
        self.workers = workers
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if STATE.is_file():
            self.state = json.loads(STATE.read_text(encoding="utf-8"))
        else:
            s3 = json.loads((ROOT / "data/runs/b2rl/geo3/pipeline.json").read_text(encoding="utf-8"))
            self.state = {"base_start": s3["base"], "base": s3["base"], "build_on": s3["build_on"], "stages": {},
                          "gated": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.real, self.gen1, self.gen2 = g4.scenario_sets()
        self.specs1 = self.real + self.gen1
        self.feats = rs.features(self.real + self.gen1 + self.gen2, workers)
        self.base_model = json.loads(NIGHT1.read_text(encoding="utf-8"))
        for name, fl in (("U1", g4.U1), ("U2", g4.U2), ("U2C", g4.U2C)):
            assert not g4.conflicts(fl), (name, g4.conflicts(fl))
        self.save()

    # ---------------------------------------------------------------- state and summary
    def save(self):
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, STATE)
        self.write_summary()

    def stage(self, tag):
        return self.state["stages"].setdefault(tag, {"attempts": []})

    def gate_status(self, name):
        slug = name.replace("_", "-")
        out = {"name": name}
        gates = SETUP / "gates.jsonl"
        if gates.is_file():
            rows = [json.loads(x) for x in gates.read_text(encoding="utf-8").splitlines() if x.strip()]
            row = ([r for r in rows if r["job"] == f"jg-b2rl-{slug}"] or [None])[-1]
            if row:
                out["kaggle_gate"] = {"inner_rel": row["inner"]["rel"], "s4_first_rel": row["s4_first"]["rel"],
                                      "passed": row["passed"]}
        sub = SETUP / f"submit-{slug}.json"
        if sub.is_file():
            try:
                out["submission"] = json.loads(sub.read_text(encoding="utf-8")).get("submitted")
            except Exception:  # noqa: BLE001 - a half-written state file
                pass
        lines = [x for x in (GEO / "pipeline.log").read_text(encoding="utf-8").splitlines() if f"gate {name}:" in x]
        out["log"] = [x[:300] for x in lines[-6:]]
        out["finished"] = any(("submit_candidate.py ended" in x or " ERROR " in x or "nothing submitted" in x
                               or "passed False" in x) for x in lines)
        return out

    def write_summary(self):
        gates = {n: self.gate_status(n) for n in self.state.get("gated", [])}
        out = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "deadline": str(DEADLINE), "base_start":
               self.state["base_start"], "base_now": self.state["base"], "gates": gates, "stages": {}}
        for tag, st in self.state["stages"].items():
            out["stages"][tag] = {k: v for k, v in st.items() if k != "attempts"}
            out["stages"][tag]["attempts"] = []
            for a in st["attempts"]:
                row = {k: v for k, v in a.items() if k not in ("map",)}
                row["gate"] = gates.get(a["name"])
                row["submission_ref"] = ((gates.get(a["name"]) or {}).get("submission") or {}).get("ref")
                out["stages"][tag]["attempts"].append(row)
        tmp = SUMMARY.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, SUMMARY)

    # ---------------------------------------------------------------- models and recording
    def model_for(self, name, cell_map, nper, periods):
        return rs.room_model(g4.OUT4, self.base_model, name, g4.policy_of(cell_map, nper, periods))

    def final_model(self, name, cell_map, nper, periods):
        path = self.model_for(name, cell_map, nper, periods)
        final = rs.MODELS / f"model-{name}.json"
        final.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return final

    def have_time(self, episodes, what, loops=1):
        minutes = episodes / EP_PER_MIN + 8 + loops * LOOP_MIN
        end = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
        ok = end <= DEADLINE
        if not ok:
            log("EVENT", f"skip {what}: needs about {minutes:.0f} min, would end {end:%m-%d %H:%M}, after the deadline "
                         f"{DEADLINE:%m-%d %H:%M}")
        return ok

    def missing(self, runs, specs, base_names):
        n = 0
        for name, levels, _ in runs:
            if all(v < 0 for v in levels):
                n += sum(not (has_record(name, s) or any(has_record(b, s) for b in base_names)) for s in specs)
            else:
                n += sum(not has_record(name, s) for s in specs)
        return n

    def design(self, prefix, base_map, factors, nper, periods, specs, base_names):
        """Record the design's runs on specs (the all-off row only where no base record exists) and estimate."""
        runs = g4.design_runs(prefix, base_map, factors, nper)
        paths = {n: self.model_for(n, m, nper, periods) for n, lv, m in runs}
        rs.record(g4.OUT4, [paths[n] for n, lv, m in runs if not all(v < 0 for v in lv)], specs, self.workers,
                  also=tuple(DIRS[1:]))
        need = [s for s in specs if not any(has_record(b, s) for b in base_names)]
        if need:
            rs.record(g4.OUT4, [paths[runs[-1][0]]], need, self.workers, also=tuple(DIRS[1:]))
        per = g4.estimate(specs, self.feats, factors, runs, DIRS, base_names)
        return runs, per

    # ---------------------------------------------------------------- the tree step and an attempt
    def choose(self, tag, per, factors, active, levels, base_map, nper, note):
        """The four rules compared by leave-buildings-out on the real buildings; the best one's decision as a map."""
        U = g4.Units(per, factors, active)
        t0 = time.time()
        lob = g4.lobo_real(U, self.real, levels=levels)
        best = min(g4.KINDS, key=lambda k: (round(lob[k]["all"], 9), g4.KINDS.index(k)))
        fit = g4.fit_kind(best, U, U.scen, levels, seed=0)
        dec = g4.decisions(fit, factors, active)
        cuts = g4.decision_cuts(dec)
        cell_map = g4.compose_decided(base_map, factors, g4.dec_function(dec, len(factors)), cuts, nper)
        on_any = any(v.any() for v in dec.values())
        pred = g4.predicted_total(U, fit, U.scen)
        info = {"scenarios": len(U.scen), "real": len([s for s in U.scen if s in set(self.real)]),
                "generated": len([s for s in U.scen if s not in set(self.real)]),
                "lobo_real_pred_pct": {k: {lab: round(100 * v, 3) for lab, v in d.items()} for k, d in lob.items()},
                "chosen_rule": best, "fit_info": fit.info, "options_on": g4.describe(dec, factors) if on_any else {},
                "predicted_design_pct": round(100 * pred, 3), "note": note}
        log("EVENT", f"{tag} tree step ({len(U.scen)} scenarios, {time.time() - t0:.0f} s): leave-buildings-out on real "
                     "buildings, predicted change %, trained with / without the generated buildings: "
                     + ", ".join(f"{k} {100 * v['all']:+.2f}/{100 * v.get('real', float('nan')):+.2f}"
                                 for k, v in lob.items() if k in g4.KINDS)
                     + f"; chosen {best}; predicted on the design scenarios {100 * pred:+.2f}%; "
                     f"{len(cuts)} outside-air cuts; on: {sorted(info['options_on'])}")
        return cell_map, on_any, info

    def attempt(self, st, tag, label, cell_map, on_any, info, nper, periods, why):
        name = f"gt_{tag.lower()}{label}"
        a = {"label": label, "name": name, "why": why, **info, "map": cell_map, "nper": nper, "periods": periods}
        if not on_any:
            a.update(passed=False, closed_loop=None, against_base=None, error="no option switched on: same as the base")
            log("EVENT", f"{tag}{label} ({why}): no option switched on, nothing to run")
            st["attempts"].append(a)
            self.save()
            return a
        model = self.final_model(name, cell_map, nper, periods)
        log("EVENT", f"{tag}{label} attempt ({why}): {len(json.loads(model.read_text(encoding='utf-8'))['room_policy']['options'])}"
                     f" options, base {self.state['base']}")
        res, gain, passed = rs.closed_loop(name, model, self.state["base"], self.workers)
        a.update(closed_loop=res, against_base=gain, passed=passed, base=self.state["base"])
        st["attempts"].append(a)
        if passed:
            gsm.launch_gate(name, model, self.workers)
            self.state["gated"].append(name)
        self.save()
        return a

    def safe_attempt(self, st, tag, label, why, fn):
        """fn() -> (cell_map, on_any, info, nper, periods); an exception is logged and counts as a failed attempt."""
        if any(a["label"] == label for a in st["attempts"]):
            return next(a for a in st["attempts"] if a["label"] == label)
        try:
            cell_map, on_any, info, nper, periods = fn()
            return self.attempt(st, tag, label, cell_map, on_any, info, nper, periods, why)
        except Exception:  # noqa: BLE001 - unattended: log and go on to the next preset correction
            log("ERROR", f"{tag}{label}: " + traceback.format_exc().replace("\n", " | ")[-1200:])
            a = {"label": label, "name": f"gt_{tag.lower()}{label}", "why": why, "passed": False,
                 "error": traceback.format_exc()[-800:]}
            st["attempts"].append(a)
            self.save()
            return a

    def finish(self, tag, st):
        passed = next((a for a in st["attempts"] if a.get("passed")), None)
        st["done"], st["passed"] = True, passed["name"] if passed else None
        if passed:
            self.state["base"], self.state["build_on"] = passed["name"], passed["map"]
            self.state["base_nper"], self.state["base_periods"] = passed.get("nper", 1), passed.get("periods")
        self.save()
        log("EVENT", f"stage {tag} ends: passed {st['passed']}; base now {self.state['base']}")

    def effects_event(self, tag, per, factors):
        s = g4.summarise(per, factors)
        log("EVENT", f"{tag} design effects on {len(per)} scenarios (cost change when on, % of the stage base): "
                     + ", ".join(f"{k} {100 * v['mean']:+.2f}±{100 * v['se']:.2f}" for k, v in s.items()))
        return s

    # ---------------------------------------------------------------- U1
    def stage_u1(self):
        st = self.stage("U1")
        if st.get("done"):
            return
        factors, base_map, nper = g4.U1, self.state["build_on"], 1
        base_names = ["gt_t3_pb11"] if self.state["base"] == self.state["base_start"] else []
        runs = g4.design_runs("gt_u1a_pb", base_map, factors, nper)
        log("EVENT", f"U1 starts: {len(factors)} factors, {len(self.specs1)} design scenarios ({len(self.real)} real + "
                     f"{len(self.gen1)} generated, one per generated building), {self.missing(runs, self.specs1, base_names)}"
                     " episodes to record")
        runs, per1 = self.design("gt_u1a_pb", base_map, factors, nper, None, self.specs1, base_names)
        st["design"] = self.effects_event("U1", per1, factors)
        active = list(range(len(factors)))
        self.safe_attempt(st, "U1", "a", "tree step on all factors",
                          lambda: (*self.choose("U1a", per1, factors, active, None, base_map, nper, "all factors"),
                                   nper, None))
        per_all = per1
        if not any(a.get("passed") for a in st["attempts"]):
            keep, table = g4.screen(per1, factors, active)
            st["screen"] = {"kept": [factors[i]["name"] for i in keep], "negative in thirds": table}
            log("EVENT", f"U1b screen (negative in >= 2 of 3 building thirds): kept {st['screen']['kept']}")
            self.save()
            if keep and self.safe_attempt_needed(st, "b"):
                sub = [factors[i] for i in keep]
                cruns = g4.design_runs("gt_u1b_pb", base_map, sub, nper)
                miss = self.missing(cruns, self.gen2, base_names)
                if self.have_time(miss, "U1b confirmation design"):
                    cruns, per2 = self.design("gt_u1b_pb", base_map, sub, nper, None, self.gen2, base_names)
                    per_all = {**per1, **g4.embed(per2, len(factors), keep)}
                    self.effects_event("U1b confirmation (survivors, generated replicates)", per2, sub)
                    self.safe_attempt(st, "U1", "b", "noise: survivors, confirmed on more scenarios",
                                      lambda: (*self.choose("U1b", per_all, factors, keep, None, base_map, nper,
                                                            "survivors, confirmation design"), nper, None))
            elif keep:
                per_all = per1
            if keep and not any(a.get("passed") for a in st["attempts"]) and self.safe_attempt_needed(st, "c") \
                    and self.have_time(0, "U1c"):
                per_c = self.merge_confirm(per1, base_map, factors, keep, base_names)
                self.safe_attempt(st, "U1", "c", "climate: first split on outside air, survivors",
                                  lambda: (*self.choose("U1c", per_c, factors, keep, g4.OAT_FIRST, base_map, nper,
                                                        "outside air first, survivors"), nper, None))
        self.finish("U1", st)

    def safe_attempt_needed(self, st, label):
        return not any(a["label"] == label for a in st["attempts"])

    def merge_confirm(self, per1, base_map, factors, keep, base_names):
        """per1 plus the confirmation design's effects when it has been recorded (else per1 alone)."""
        sub = [factors[i] for i in keep]
        cruns = g4.design_runs("gt_u1b_pb", base_map, sub, 1)
        if self.missing(cruns, self.gen2, base_names) == 0:
            per2 = g4.estimate(self.gen2, self.feats, sub, cruns, DIRS, base_names)
            return {**per1, **g4.embed(per2, len(factors), keep)}
        return per1

    # ---------------------------------------------------------------- U2
    def stage_u2(self):
        st = self.stage("U2")
        if st.get("done"):
            return
        factors, nper = g4.U2, 4
        base_map = self.state["build_on"]
        base_names = ["gt_t3_pb11"] if self.state["base"] == self.state["base_start"] else []
        runs = g4.design_runs("gt_u2a_pb", base_map, factors, nper)
        miss = self.missing(runs, self.specs1, base_names)
        log("EVENT", f"U2 starts on base {self.state['base']}: {len(factors)} factors x periods, {len(self.specs1)} "
                     f"scenarios, {miss} episodes to record")
        if not self.have_time(miss, "U2a design"):
            self.finish("U2", st)
            return
        runs, per = self.design("gt_u2a_pb", base_map, factors, nper, NARROW, self.specs1, base_names)
        st["design"] = self.effects_event("U2a", per, factors)
        active = list(range(len(factors)))
        a = self.safe_attempt(st, "U2", "a", "periods 1 h warm-up / 2 h late, tree step",
                              lambda: (*self.choose("U2a", per, factors, active, None, base_map, nper, "periods narrow"),
                                       nper, NARROW))
        if not a.get("passed"):
            runs_b = g4.design_runs("gt_u2b_pb", base_map, factors, nper)
            miss = self.missing(runs_b, self.specs1, base_names)
            if self.have_time(miss, "U2b design"):
                runs_b, per_b = self.design("gt_u2b_pb", base_map, factors, nper, WIDE, self.specs1, base_names)
                st["design_b"] = self.effects_event("U2b (2 h warm-up / 3 h late)", per_b, factors)
                a = self.safe_attempt(st, "U2", "b", "edges: wider periods, new design",
                                      lambda: (*self.choose("U2b", per_b, factors, active, None, base_map, nper,
                                                            "periods wide"), nper, WIDE))
        if not any(x.get("passed") for x in st["attempts"]):
            fc = g4.U2C
            runs_c = g4.design_runs("gt_u2c_pb", base_map, fc, nper)
            miss = self.missing(runs_c, self.specs1, base_names)
            if self.have_time(miss, "U2c design"):
                runs_c, per_c = self.design("gt_u2c_pb", base_map, fc, nper, NARROW, self.specs1, base_names)
                st["design_c"] = self.effects_event("U2c (warm-up only, larger levels)", per_c, fc)
                self.safe_attempt(st, "U2", "c", "warm-up only, larger levels",
                                  lambda: (*self.choose("U2c", per_c, fc, list(range(len(fc))), None, base_map, nper,
                                                        "warm-up only"), nper, NARROW))
        self.finish("U2", st)

    # ---------------------------------------------------------------- gates
    def wait_for_gates(self):
        while datetime.datetime.now() < GATE_WAIT_UNTIL:
            pending = [n for n in self.state.get("gated", []) if not self.gate_status(n)["finished"]]
            self.write_summary()
            if not pending:
                return
            log("RESULT", f"waiting for the gates of {pending}")
            time.sleep(300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    os.environ.setdefault("SB_FASTSIM", "rom")
    pipe = Pipeline(args.workers)
    try:
        log("EVENT", f"pipeline 4 starts from base {pipe.state['base']} (deadline {DEADLINE:%m-%d %H:%M})")
        for stage in (pipe.stage_u1, pipe.stage_u2):
            try:
                stage()
            except Exception:  # noqa: BLE001 - a failed stage must not stop the next one
                log("ERROR", f"{stage.__name__}: " + traceback.format_exc().replace("\n", " | ")[-1500:])
        pipe.state["done"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pipe.save()
        log("EVENT", f"pipeline 4 work done, base {pipe.state['base']}; waiting for the gates")
        pipe.wait_for_gates()
        pipe.save()
        log("EVENT", f"pipeline 4 done, summary {SUMMARY.relative_to(ROOT)}")
    except Exception:  # noqa: BLE001
        log("ERROR", traceback.format_exc().replace("\n", " | ")[-1500:])
        raise


if __name__ == "__main__":
    main()
