"""Stage 戊 after the data (2026-11-06; spec docs/plan/tree-stage-wu-20261106.md §3-4): learn -> deploy -> rerun on
the design scenarios -> closed loop vs gt_r5pa2 -> Kaggle gate -> submission rule -> score wait. Every shortfall wakes
the coordinator (exit 3, <state>/wake.json): the user wants both corrections diagnosed by the coordinator (11/06).

  learn   : per-scenario effects (geo5_wu.estimate, nominal totals, 24-run fold-over), units with the deployable
            features (area band, wall band, oat) and the report features, leave-buildings-out on the real buildings of
            every way (geo5_wu_learn.KINDS_ALL, cost-weighted, trained with every generated scenario; "real" = without
            them), the most negative one is the candidate (round 4's rule); reports: learning curve over 0 / 26 / 52 /
            all generated buildings, physical vs adjacency-graph representation (teacher), gate fidelities.
  deploy  : the candidate fitted on every scenario -> room cell map (group_stage4.compose_decided + policy_of, the
            base map's climate cuts merged) and building-wide climate gates (b2rl rule_oat_gates); checked against
            the decisions on every cut bin; predicted change on the design scenarios (cost-weighted, in sample).
  verify  : the candidate recorded on the design scenarios; realised cost-weighted change vs predicted; wake when the
            realised saving is under half of the predicted one (or the prediction is not a saving).
  closed  : held-out 60 / INNER 29 / S4 first 30 vs gt_r5pa2, pass = <= -0.5 / <= -0.5 / <= +0.5 points (unchanged).
  gate    : pipeline/setup/geo_gate.py (GATE_NO_SUBMIT=1, billing check on).
  submit  : geo5_submit.decide_paper(final=True) -> submit_candidate.py; score : score_wait.py.

python3 -B pipeline/launch/launch.py pipeline/setup/geo5_wu_stage.py --label a [--until learn|deploy|verify|closed|gate]
        [--kinds ...] [--factors ...] [--levels default|oat_first] [--submit auto|dry] [--wait-data]"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_lib as L  # noqa: E402
import geo5_options as O  # noqa: E402
import geo5_record as R  # noqa: E402
import geo5_submit as SUB  # noqa: E402
import geo5_wu as W  # noqa: E402
import geo5_wu_learn as WL  # noqa: E402
import group_stage as gs  # noqa: E402
import group_stage4 as g4  # noqa: E402

STATE = W.OUT / "stage"
MODELS = ROOT / "data/runs/b2rl/search/models"
HOLDOUT, INNER, S4 = "train_ceiling_rest60", "inner", "s4_first"
STAGES = ("learn", "deploy", "verify", "closed", "gate", "submit", "score")


class Wake(Exception):
    pass


def feats_by_spec(specs):
    uf = json.loads((W.G / "tree2/unit_features.json").read_text(encoding="utf-8"))
    return {s: uf[s.split("/")[1]] for s in specs}


def extra_fn(feats, bmap, per):
    """Report / teacher features of a (scenario, factor) unit: the base map's settings of the affected cells at the
    scenario's climate bin, AHU and neighbour statistics of the affected rooms (a building-wide factor: all rooms)."""
    cache = {}

    def build(spec):
        f = feats[spec]
        rooms = list(f["rooms"])
        cells = [gs.cell_of(a, w) for a, w in zip(f["area"], f["wall"])]
        ahu = np.asarray(f["ahu"])
        nb = {}
        for a, b, length in f["shared"]:
            nb.setdefault(a, []).append((b, length))
            nb.setdefault(b, []).append((a, length))
        j = sum(per[spec]["oat"] > c for c in bmap["oat_cuts"])
        base = bmap["maps"][j]
        return rooms, cells, ahu, nb, base, f

    def fn(spec, i):
        if spec not in cache:
            cache[spec] = build(spec)
        rooms, cells, ahu, nb, base, f = cache[spec]
        fac = W.FACTORS[i]
        sel = [k for k, c in enumerate(cells) if fac.get("whole") or list(c) in fac["cells"]]
        sel = sel or list(range(len(rooms)))
        sset = set(sel)
        opts = [base.get(g4._key(cells[k], 0), {}) for k in sel]
        ahu_n = {g: int(np.sum(ahu == g)) for g in np.unique(ahu)}
        aff = {g: sum(1 for k in sel if ahu[k] == g) for g in ahu_n}
        same = [(aff[ahu[k]] - 1) / (ahu_n[ahu[k]] - 1) if ahu_n[ahu[k]] > 1 else 0.0 for k in sel]
        num = {r: k for k, r in enumerate(rooms)}
        nbc = [len(nb.get(rooms[k], [])) for k in sel]
        nbw = [sum(x for _, x in nb.get(rooms[k], [])) for k in sel]
        nbo = [np.mean([num.get(b) in sset for b, _ in nb.get(rooms[k], [])]) if nb.get(rooms[k]) else 0.0
               for k in sel]
        ext = f["ahu_exterior_share"]
        return {"has_min": float(np.mean([("zone_min_ra_lps_m2" in o) for o in opts])),
                "base_ratio": float(np.mean([o.get("room_heat_max_ratio", 0.0) for o in opts])),
                "base_cool": float(np.mean([o.get("room_cool_margin_k", 0.0) for o in opts])),
                "base_heat": float(np.mean([o.get("room_heat_margin_k", 0.0) for o in opts])),
                "ahu_rooms": float(np.mean([ahu_n[ahu[k]] for k in sel])),
                "ahu_ext": float(np.mean([ext[str(int(ahu[k]))] if str(int(ahu[k])) in ext else ext[int(ahu[k])]
                                          for k in sel])),
                "nb_same": float(np.mean(same)),
                "depth": float(np.mean([min(d, 30.0) if d is not None else 30.0 for d in
                                        [f["depth"][k] for k in sel]])),
                "wall_ns": float(np.mean([f["wall_n"][k] + f["wall_s"][k] for k in sel])),
                "wall_ew": float(np.mean([f["wall_e"][k] + f["wall_w"][k] for k in sel])),
                "nb_count": float(np.mean(nbc)), "nb_wall": float(np.mean(nbw)), "nb_on": float(np.mean(nbo))}
    return fn


class Stage:
    def __init__(self, a):
        self.a = a
        self.ctx = L.Ctx(STATE, tag="wu-stage")
        self.st = self.ctx.state
        self.P = W.plan()
        self.att = self.st.setdefault("attempts", {}).setdefault(a.label, {"stages": {}})
        self.name = f"gt_wu{a.label}"

    def log(self, kind, text):
        self.ctx.log(kind, text)

    def save(self):
        self.ctx.save()

    def done(self, s):
        return self.att["stages"].get(s, {}).get("done")

    # ------------------------------------------------------------ learn
    def units(self, specs=None):
        specs = specs or self.P["specs"]
        feats = feats_by_spec(specs)
        per = W.estimate(self.P, {s: {"area": f["area"], "wall": f["wall"]} for s, f in feats.items()}, specs)
        active = list(range(len(W.FACTORS))) if self.a.factors == "all" else \
            [k for k, f in enumerate(W.FACTORS) if f["name"] in self.a.factors.split(",")]
        U = WL.Units(per, W.FACTORS, active, weighted=True, extra=extra_fn(feats, self.P["bmap"], per))
        return per, U, active

    def stage_learn(self, st):
        per, U, active = self.units()
        real = [s for s in self.P["real"] if s in per]
        gen = [s for s in per if s not in set(real)]
        self.log("EVENT", f"learn {self.name}: {len(per)} scenarios with all 24 runs ({len(real)} real, {len(gen)} "
                          f"generated), {len(active)} factors, {len(U.Y)} units")
        cfg = WL.UPGRADE
        levels = g4.OAT_FIRST if self.a.levels == "oat_first" else None
        summ = g4.summarise(per, W.FACTORS)
        cw = {}
        for i in active:
            vals = [(per[s]["eff"][i], per[s]["cost"]) for s in per if per[s]["eff"][i] is not None]
            e, w = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
            cw[W.FACTORS[i]["name"]] = {"cw": float(np.sum(w * e) / np.sum(w)), "eq": float(np.mean(e)), "n": len(e),
                                        "se_eq": summ[W.FACTORS[i]["name"]]["se"]}
        self.log("RESULT", "design effects (cost-weighted / equal-weight, % of the rule's cost): " +
                 ", ".join(f"{k} {100 * v['cw']:+.2f}/{100 * v['eq']:+.2f}" for k, v in cw.items()))
        kinds = tuple(self.a.kinds.split(",")) if self.a.kinds else WL.KINDS_ALL
        t0 = time.time()
        lob = WL.lobo_real(U, real, cfg, kinds=kinds, levels=levels)
        best = min(kinds, key=lambda k: (round(lob[k]["all"], 9), kinds.index(k)))
        self.log("RESULT", f"leave-buildings-out on {len(real)} real scenarios ({time.time() - t0:.0f} s), predicted "
                           "cost-weighted change % with / without the generated buildings: " +
                 ", ".join(f"{k} {100 * lob[k]['all']:+.2f}/{100 * lob[k].get('real', float('nan')):+.2f}" for k in kinds)
                 + f"; per-scenario oracle {100 * lob['oracle']['all']:+.2f}; chosen {best}")
        rep = {}
        try:                                         # reports only (paper): learning curve and representation
            blds = sorted({s.split("/")[1] for s in gen})
            curve = {}
            for n in (0, 26, 52, len(blds)):
                vals = []
                for seed in ((0,) if n in (0, len(blds)) else (0, 1, 2)):
                    pick = set(np.random.default_rng(seed).permutation(blds)[:n].tolist())
                    sub = [s for s in gen if s.split("/")[1] in pick]
                    r = WL.lobo_real(U, real, cfg, kinds=(best, "tree"), levels=levels, gen=sub, variants=("all",))
                    vals.append({k: r[k]["all"] for k in (best, "tree")})
                curve[n] = {k: float(np.mean([v[k] for v in vals])) for k in (best, "tree")}
            rep["learning_curve"] = curve
            r = WL.lobo_real(U, real, cfg, kinds=("teacher_phys", "teacher_graph"), levels=levels, variants=("all",))
            rep["representation"] = {k: r[k]["all"] for k in ("teacher_phys", "teacher_graph")}
            self.log("RESULT", "report: learning curve (generated buildings -> predicted %): " +
                     "; ".join(f"{n}: " + ", ".join(f"{k} {100 * v:+.2f}" for k, v in d.items())
                               for n, d in curve.items()) + "; representation (teacher, physical vs adjacency graph): "
                     + ", ".join(f"{k} {100 * v:+.2f}" for k, v in rep["representation"].items()))
        except Exception:  # noqa: BLE001 - a report must not stop the stage
            self.log("WARN", "reports failed: " + traceback.format_exc().replace("\n", " | ")[-800:])
        st.update(scenarios=len(per), real=len(real), generated=len(gen), effects=cw, lobo=lob, chosen=best,
                  kinds=list(kinds), levels=self.a.levels, factors=[W.FACTORS[i]["name"] for i in active],
                  reports=rep)

    # ------------------------------------------------------------ deploy
    def stage_deploy(self, st):
        per, U, active = self.units()
        learn = self.att["stages"]["learn"]
        levels = g4.OAT_FIRST if self.a.levels == "oat_first" else None
        fit = WL.fit_kind(learn["chosen"], U, U.scen, WL.UPGRADE, levels, seed=0)
        dec = WL.decisions(fit, W.FACTORS, active)
        room_dec = {j: dec[W.NW + j] for j in range(len(W.ROOM)) if (W.NW + j) in dec}
        cuts = g4.decision_cuts(room_dec)
        cmap = g4.compose_decided(self.P["bmap"], W.ROOM, g4.dec_function(room_dec, len(W.ROOM)), cuts, 1)
        whole_on, gates = [], []
        from dataclasses import fields
        from smart_building.policies.g36 import G36Config
        default = {f.name: f.default for f in fields(G36Config)}
        for i in range(W.NW):
            if i not in dec or not dec[i].any():
                continue
            v = dec[i]
            lvl = W.WHOLE[i]
            whole_on.append(lvl)
            if v.all():
                continue
            ch = np.where(v[1:] != v[:-1])[0]
            if len(ch) != 1:
                raise Wake(f"building-wide factor {lvl}: decision is not one climate interval ({len(ch)} changes)")
            c = float((WL.GRID[ch[0]] + WL.GRID[ch[0] + 1]) / 2.0)
            on = self.P["by"][lvl]["build"]["rule_config"]
            gates.append({"id": lvl, "on": dict(on), "off": {k: default[k] for k in on},
                          "lo": c if v[-1] else None, "hi": None if v[-1] else c})
        model = O.build(self.P["base"], [self.P["by"][x] for x in self.P["rule_ids"] + whole_on], self.name)
        model["room_policy"] = g4.policy_of(cmap)
        if gates:
            model["rule_oat_gates"] = gates
        model["source"]["geo5_wu"] = {"label": self.a.label, "kind": learn["chosen"], "whole_on": whole_on,
                                      "gates": gates, "room_on": g4.describe(room_dec, W.ROOM)}
        on_any = bool(whole_on) or any(v.any() for v in room_dec.values())
        if not on_any:
            raise Wake(f"{self.name}: the chosen way ({learn['chosen']}) switches nothing on")
        # check: on every bin of the composed map, each room factor's keys are set exactly where decided (factors never
        # share a key on a common cell: group_stage4.conflicts)
        dfn = g4.dec_function(room_dec, len(W.ROOM))
        bad = 0
        for j, rep in enumerate(g4.bin_reps(cmap["oat_cuts"] if gs.is_climate(cmap) else [])):
            m = cmap["maps"][j] if gs.is_climate(cmap) else cmap
            j0 = sum(rep > c for c in self.P["bmap"]["oat_cuts"])
            for k, fac in enumerate(W.ROOM):
                want = bool(dfn(k, rep))
                for c in fac["cells"]:
                    basev = self.P["bmap"]["maps"][j0].get(g4._key(c, 0), {})
                    got = m.get(g4._key(c, 0), {})
                    bad += sum(got.get(key) != (val if want else basev.get(key)) for key, val in fac["delta"].items())
        if bad:
            raise Wake(f"{self.name}: the composed room map disagrees with the decisions in {bad} places")
        path = W.OUT / "models" / f"model-{self.name}.json"
        O.write_model(model, path)
        pred = WL.score(U, fit, U.scen, WL.UPGRADE)
        pred_real = WL.score(U, fit, [s for s in self.P["real"] if s in U.sid], WL.UPGRADE)
        self.log("RESULT", f"deploy {self.name} ({learn['chosen']}): room factors on {g4.describe(room_dec, W.ROOM)}; "
                           f"building-wide on {whole_on} gates {[(g['id'], g['lo'], g['hi']) for g in gates]}; "
                           f"predicted on the design scenarios {100 * pred:+.2f}% (real {100 * pred_real:+.2f}%), "
                           f"cost-weighted")
        st.update(model=self.ctx.rel(path), sha=O.behaviour_sha(model), predicted=pred, predicted_real=pred_real,
                  room_on=g4.describe(room_dec, W.ROOM), whole_on=whole_on, gates=gates, fit_info=fit.info)

    # ------------------------------------------------------------ verify on the design scenarios
    def stage_verify(self, st):
        dep = self.att["stages"]["deploy"]
        rel = dep["model"]
        pairs = [(rel, s) for s in self.P["specs"]]
        try:
            R.record(self.ctx, f"verify {self.name}", pairs, W.OUT, self.a.workers)
        except L.Wake as w:
            raise Wake(f"verify recording: {w.reason}: {w.text}")
        rows = {}
        for s in self.P["specs"]:
            c = json.loads(R.record_path(W.OUT, self.name, s).read_text(encoding="utf-8"))["summary"]["cost"]
            b = W.find_cost(W.RULE, s)["summary"]["cost"]
            rows[s] = (c, b)
        real = set(self.P["real"])
        cwf = lambda sel: float(sum(rows[s][0] - rows[s][1] for s in sel) / sum(rows[s][1] for s in sel))  # noqa: E731
        realised, realised_real = cwf(list(rows)), cwf([s for s in rows if s in real])
        eq = float(np.mean([rows[s][0] / rows[s][1] - 1 for s in rows]))
        pred = dep["predicted"]
        ok = pred < 0 and realised <= 0.5 * pred
        st.update(realised=realised, realised_real=realised_real, realised_eq=eq, predicted=pred, passed=ok)
        self.log("RESULT", f"verify {self.name} on the {len(rows)} design scenarios: realised {100 * realised:+.2f}% "
                           f"(real {100 * realised_real:+.2f}%, equal-weight {100 * eq:+.2f}%) vs predicted "
                           f"{100 * pred:+.2f}% (real {100 * dep['predicted_real']:+.2f}%), cost-weighted; "
                           f"{'OK' if ok else 'UNDER HALF OF THE PREDICTION'}")
        if not ok:
            raise Wake(f"verify {self.name}: realised {100 * realised:+.2f}% vs predicted {100 * pred:+.2f}%")

    # ------------------------------------------------------------ closed loop
    def stage_closed(self, st):
        import geo_pipeline as gp
        src = ROOT / self.att["stages"]["deploy"]["model"]
        final = MODELS / f"model-{self.name}.json"
        shutil.copyfile(src, final)
        ref, ref_model = W.RULE, MODELS / f"model-{W.RULE}.json"
        res, gain = {}, {}
        for subset in (HOLDOUT, INNER, S4):
            for n, m in ((self.name, final), (ref, ref_model)):
                out = gp.OUT / subset / f"{n}.json"
                if not out.is_file():
                    env = {k: x for k, x in os.environ.items() if k != "SB_FASTSIM"}
                    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    el = L.run_monitored(self.ctx, f"closed loop {n} {subset}",
                                         ["nice", "-n", "5", "python3", "-B", "scripts/b2rl.py", "eval", "--policy",
                                          self.ctx.rel(m), "--set", subset, "--workers",
                                          str(L.worker_count(self.a.workers)), "--fast", "rom", "--out",
                                          self.ctx.rel(out)], env=env, estimate_min=self.ctx.estimate("closed_loop", 18.0))
                    self.ctx.measured("closed_loop", el)
            res[subset] = gp.evaluate(self.name, final, subset, self.a.workers)
            gain[subset] = res[subset]["mean"] - gp.evaluate(ref, ref_model, subset, self.a.workers)["mean"]
        passed = gain[HOLDOUT] <= -0.5 and gain[INNER] <= -0.5 and gain[S4] <= 0.5
        st.update(result=res, against_base=gain, passed=passed, model=self.ctx.rel(final))
        self.log("RESULT", f"closed loop {self.name} vs {ref}: held-out 60 {gain[HOLDOUT]:+.2f}, INNER 29 "
                           f"{gain[INNER]:+.2f}, S4 first 30 {gain[S4]:+.2f} points; "
                           f"{'PASSED' if passed else 'below the pass rule'}")
        if not passed:
            raise Wake(f"closed loop {self.name} below the pass rule: {gain}")

    # ------------------------------------------------------------ Kaggle gate and submission
    def stage_gate(self, st):
        model = self.att["stages"]["closed"]["model"]
        out = f"{self.ctx.rel(STATE)}/gate/{self.name}.json"
        env = dict(os.environ, GATE_NO_SUBMIT="1", GATE_RESULT=out)
        env.pop("GATE_SKIP_BILLING", None)
        el = L.run_monitored(self.ctx, f"gate {self.name}", ["python3", "-B", "pipeline/setup/geo_gate.py", self.name,
                                                             model], env=env, estimate_min=self.ctx.estimate("gate", 190.0))
        self.ctx.measured("gate", el)
        res = L.read_json(ROOT / out) or {"passed": False, "reason": "geo_gate.py wrote no verdict"}
        st.update(verdict=res)
        self.log("RESULT", f"Kaggle gate {self.name}: {'PASSED' if res.get('passed') else 'NOT passed'} "
                           f"({res.get('reason')})")
        if not res.get("passed"):
            raise Wake(f"Kaggle gate {self.name}: {res.get('reason')}")

    def stage_submit(self, st):
        verdict = self.att["stages"]["gate"]["verdict"]
        gate = verdict["gate"]
        model = self.att["stages"]["closed"]["model"]
        p5 = json.loads((W.G / "pipeline.json").read_text(encoding="utf-8"))
        paper_refs = [x["ref"] for x in p5.get("submissions", []) if x.get("track") == "paper" and x.get("ref")]
        n, day, _ = SUB.count_today()
        rows = SUB.leaderboard()
        scored = [r for r in rows if r["ref"] in paper_refs and r["public"] is not None and r["s4_rel"] is not None]
        dec = SUB.decide_paper(gate, n, False, scored, rows, exclude=paper_refs, final=True)
        gates_file = verdict.get("gates_file") or self.ctx.rel(STATE / "gate" / f"gates-{self.name}.json")
        if not verdict.get("gates_file"):
            L.atomic_json(ROOT / gates_file, gate)
        desc = (f"paper rule gt_r5pa2 + stage-wu tree ({self.att['stages']['learn']['chosen']}): room settings by "
                "cell x climate, building-wide climate gates; physics options only")
        dec["commands"] = SUB.commands(self.name, model, gates_file, desc)
        dec.update(mode=self.a.submit, day=day, name=self.name)
        st["decision"] = dec
        self.log("RESULT", f"submission rule {self.name}: estimate {dec['estimate']:.5f}; today's submissions {n}; "
                           + ("SUBMIT" if dec["submit"] else "no submission: " + "; ".join(dec["reasons"])))
        if not dec["submit"]:
            raise Wake(f"submission refused: {dec['reasons']}")
        if self.a.submit != "auto":
            self.log("EVENT", "submission dry run: nothing submitted")
            return
        wait_extra = 0.0
        if dec["quota_wait"]:
            now = datetime.datetime.now(datetime.timezone.utc)
            wait_extra = ((now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
                          - now).total_seconds() / 60.0
        L.run_monitored(self.ctx, f"submit {self.name}", dec["commands"]["submit"],
                        estimate_min=self.ctx.estimate("submit", 30.0) + wait_extra)
        ref = SUB.submitted_ref(self.name)
        if not ref:
            raise Wake(f"submit_candidate.py ended without a ref for {self.name}")
        st["ref"] = ref
        self.log("RESULT", f"submitted {self.name}: ref {ref}")

    def stage_score(self, st):
        ref = self.att["stages"].get("submit", {}).get("ref")
        if not ref:
            return
        L.run_monitored(self.ctx, f"score {ref}", ["python3", "-B", "pipeline/setup/score_wait.py", ref],
                        estimate_min=self.ctx.estimate("score", 540.0))
        row = next((r for r in SUB.leaderboard() if r["ref"] == ref), None)
        st["public"] = row and row["public"]
        self.log("RESULT", f"score {self.name} ref {ref}: public {st['public']}")

    # ------------------------------------------------------------ driver
    def run(self):
        for s in STAGES:
            if self.done(s):
                continue
            if self.a.submit == "none" and s in ("submit", "score"):
                break
            st = self.att["stages"].setdefault(s, {})
            st["started"] = L.now()
            self.save()
            try:
                getattr(self, f"stage_{s}")(st)
            except (Wake, L.Wake) as w:
                st["wake"] = str(w)
                self.save()
                payload = {"time": L.now(), "label": self.a.label, "stage": s, "reason": str(w),
                           "state": self.ctx.rel(self.ctx.state_path), "log": self.ctx.rel(self.ctx.log_path)}
                L.atomic_json(STATE / "wake.json", payload)
                self.log("WAKE", f"{self.name} {s}: {w}")
                return 3
            except Exception:  # noqa: BLE001
                st["error"] = traceback.format_exc()[-2000:]
                self.save()
                L.atomic_json(STATE / "wake.json", {"time": L.now(), "label": self.a.label, "stage": s,
                                                     "reason": "crash", "trace": st["error"]})
                self.log("WAKE", f"{self.name} {s}: crash " + st["error"].replace("\n", " | ")[-600:])
                return 3
            st["done"] = L.now()
            self.save()
            if self.a.until == s:
                self.log("EVENT", f"{self.name}: stopped after {s} (--until)")
                return 0
        self.log("EVENT", f"{self.name}: all stages done")
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="a")
    p.add_argument("--until", choices=STAGES)
    p.add_argument("--kinds", default="")
    p.add_argument("--factors", default="all")
    p.add_argument("--levels", default="default", choices=("default", "oat_first"))
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--submit", default="auto", choices=("auto", "dry", "none"))
    p.add_argument("--wait-data", action="store_true", help="wait for the data collection to finish first")
    p.add_argument("--redo", default="", help="comma list of stages of this label to run again")
    a = p.parse_args()
    if a.wait_data:
        while not (L.read_json(W.OUT / "pipeline.json") or {}).get("done"):
            if (W.OUT / "wake.json").is_file() and (W.OUT / "wake.json").stat().st_mtime > time.time() - 3600 * 48 \
                    and "wu data" in (W.OUT / "wake.json").read_text(encoding="utf-8"):
                print("the data collection woke (wake.json); not starting the stage")
                return 3
            time.sleep(300)
    stage = Stage(a)
    for s in [x for x in a.redo.split(",") if x]:
        stage.att["stages"].pop(s, None)
    if (STATE / "wake.json").is_file():
        os.replace(STATE / "wake.json", STATE / f"wake-handled-{time.strftime('%Y%m%d-%H%M%S')}.json")
    return stage.run()


if __name__ == "__main__":
    sys.exit(main())
