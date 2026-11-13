"""Stage 戊之三 chain (2026-11-09; spec docs/plan/tree-round3-20261109.md). State data/runs/b2rl/geo5/r3/stage.
  learn    building-wide: per-scenario effects of the three whole-plot options -> climate-only kind chosen by
           leave-buildings-out among rulefit / figs / tree / me; rooms: joint room tree (geo5_r3_learn.fit_rooms),
           leave-buildings-out calibration reported
  deploy   gt_r3<label>1: gt_wub's rule and four gates + the new gates + the stage base room map with the room tree's
           per-room options (composer as geo5_tk_room.compose_policy); every design room x climate edge checked
  verify   the candidate on the design scenarios (realised vs predicted against gt_r5pa2; < half -> wake)
  select   candidate and gt_r3b on the selection set (r3/sets.json; gt_wub, gt_r5pa2 recorded there already): paired
           change against gt_wub, scenario-bootstrap SE (point 2)
  dagger   the candidate's own selection episodes (and gt_r3b's) join the room data; refit -> gt_r3<label>2, deployed
           and checked, then run on the selection set (point 6)
  choose   the candidate with the better paired mean against gt_wub on the selection set; neither better -> wake
  closed / gate / submit / score   geo5_wu_stage's (pass rule unchanged, against gt_r5pa2)
python3 -B pipeline/launch/launch.py pipeline/setup/geo5_r3_stage.py --label a --wait-data --submit auto"""
import argparse
import copy
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_lib as L  # noqa: E402
import geo5_options as O  # noqa: E402
import geo5_r3_collect as C  # noqa: E402
import geo5_r3_learn as RL  # noqa: E402
import geo5_record as R  # noqa: E402
import geo5_tk_learn as TK  # noqa: E402
import geo5_tk_room as RM  # noqa: E402
import geo5_wu as W  # noqa: E402
import geo5_wu_learn as WL  # noqa: E402
import geo5_wu_stage as S  # noqa: E402
import group_stage as gs  # noqa: E402

S.STATE = C.OUT / "stage"
STAGES = ("learn", "deploy", "verify", "select", "dagger", "choose", "closed", "gate", "submit", "score")
WUB = "gt_wub"
KINDS = ("rulefit", "figs", "tree", "me")
R3B = C.OUT / "models/model-gt_r3b.json"


def base_model():
    wub, rule = json.loads(C.WUB.read_text(encoding="utf-8")), json.loads(C.RULE.read_text(encoding="utf-8"))
    m = copy.deepcopy(wub)
    m["room_policy"] = copy.deepcopy(rule["room_policy"])
    m["source"]["geo5_r3"] = {"base": "gt_wub rule + gt_r5pa2 room map"}
    return m


def room_factors():
    allc = [list(c) for c in gs.CELLS]
    return [{"name": d["name"], "type": d["name"], "cells": d["cells"] if d["cells"] is not None else allc,
             "delta": d["set"]} for d in C.DELTAS]


class _RF:
    """Adapter: geo5_tk_room.compose_policy expects models keyed by type with RoomFit-like trees using FEATS."""

    def __init__(self, fit):
        self.models = {C.DELTAS[j]["name"]: self._conv(t) for j, t in enumerate(fit.trees)}

    @staticmethod
    def _conv(n):
        if "f" not in n:
            return {"on": bool(n.get("on", False))}
        return {"f": n["f"], "t": n["t"], "l": _RF._conv(n["l"]), "r": _RF._conv(n["r"])}

    def on_rows(self, ftype, X):
        node = self.models.get(ftype)
        out = np.zeros(len(X), bool)
        for k, x in enumerate(X):
            n = node
            while n is not None and "f" in n:
                n = n["l"] if x[n["f"]] <= n["t"] else n["r"]
            out[k] = bool(n and n.get("on", False))
        return out


def room_fit_of(spec):
    """A room kind of geo5_r3_kinds fitted on all the data; tree features mapped back to the run-time ones."""
    import geo5_r3_kinds as K
    saved = (RM.room_rows, RL.RM.room_rows, RL.FEATS, RL.SPLIT_FEATS, RL.NO)
    try:
        K.setup_features()
        if spec.startswith("ahu-"):        # 2026-11-10 correction: per-AHU decisions (geo5_r3_ahu)
            import geo5_r3_ahu as A
            A.DAGGER = spec.endswith("+dg")   # + the relabelling data (geo5_r3_dagger2)
            A.DATA = A.AhuData()
            K.DATA = A._AddData(A.DATA)
            K.prepare_scales()
            K.DATA = A.DATA
            K.BAGS = 10
            fit = A.fit(spec[:-3] if A.DAGGER else spec, list(A.DATA.scen), seed=0)
        else:
            kind, rep = spec.split("|")
            K.DATA = RL.Data()
            K.prepare_scales()
            K.use_rep(rep)
            fit = K.fit_room_kind(kind, K.DATA.obs, K.DATA, list(K.DATA.scen), seed=0)
    finally:
        RM.room_rows, RL.RM.room_rows, RL.FEATS, RL.SPLIT_FEATS, RL.NO = saved

    def remap(n):
        if "f" in n:
            name = K.FEATS_X[n["f"]]
            if name not in RM.FEATS_ALL:
                raise S.Wake(f"room kind {spec}: feature {name} is not computed at run time (not deployable)")
            n["f"] = RM.FEATS_ALL.index(name)
            remap(n["l"])
            remap(n["r"])
    for tr in fit.trees:
        remap(tr)
    return fit


def compose(bmap, fit):
    """Room policy of the room tree on the stage base map (geo5_tk_room.compose_policy with the r3 room options)."""
    saved = W.ROOM
    try:
        W.ROOM = room_factors()
        return RM.compose_policy(bmap, _RF(fit))
    finally:
        W.ROOM = saved


def intended(bmap, fit, rows_b, k, oat):
    saved = W.ROOM
    try:
        W.ROOM = room_factors()
        return RM.intended_option(bmap, _RF(fit), rows_b, k, oat)
    finally:
        W.ROOM = saved


class R3Stage(S.Stage):
    def __init__(self, a):
        self.a = a
        self.ctx = L.Ctx(S.STATE, tag="r3-stage")
        self.st = self.ctx.state
        self.P = W.plan()
        self.att = self.st.setdefault("attempts", {}).setdefault(a.label, {"stages": {}})
        self.name = self.att["stages"].get("choose", {}).get("name") or f"gt_r3{a.label}1"

    # ------------------------------------------------------------ learning
    def whole_fit(self, data, kind=None, extra_scen=None):
        per = RL.whole_effects(data)
        fac = [W.whole_factor(w) for w, _ in C.WHOLE]
        U = WL.Units(per, fac, list(range(len(fac))), weighted=True)
        real = [s for s in self.P["real"] if s in per]
        res = {}
        if kind is None:
            for k in KINDS:
                res[k] = TK.compare(U, real, WL.UPGRADE, kinds=(k,))[k]
            kind = min(KINDS, key=lambda k: (round(res[k]["lobo"] + res[k]["loco"], 9), KINDS.index(k)))
        fit = TK.fit_kind(kind, U, U.scen, WL.UPGRADE, None, seed=0)
        dec = WL.decisions(fit, fac, list(range(len(fac))))
        return per, U, kind, dec, res

    def gates_of(self, dec):
        from smart_building.policies.g36 import G36Config
        default = {f.name: f.default for f in fields(G36Config)}
        gates, on_keys = [], {}
        for i, (name, cfg) in enumerate(C.WHOLE):
            v = dec[i]
            if not v.any():
                continue
            on_keys.update(cfg)
            if v.all():
                continue
            ch = np.where(v[1:] != v[:-1])[0]
            if len(ch) != 1:
                raise S.Wake(f"building-wide {name}: decision is not one climate interval ({len(ch)} changes)")
            c = float((WL.GRID[ch[0]] + WL.GRID[ch[0] + 1]) / 2.0)
            gates.append({"id": name, "on": dict(cfg), "off": {k: default[k] for k in cfg},
                          "lo": c if v[-1] else None, "hi": None if v[-1] else c})
        return gates, on_keys

    def stage_learn(self, st):
        t0 = time.time()
        data = RL.Data()
        per, U, kind, dec, res = self.whole_fit(data, kind=self.a.whole_kind or None)
        self.log("RESULT", f"learn {self.name}: {len(data.scen)} scenarios, {len(data.obs)} episodes; building-wide "
                           "kinds (leave-buildings-out / leave-one-climate-out, %): " +
                 ", ".join(f"{k} {100 * v['lobo']:+.2f}/{100 * v['loco']:+.2f}" for k, v in res.items()) +
                 f"; chosen {kind}; decisions " + ", ".join(
                     f"{C.WHOLE[i][0]} {self._iv(v)}" for i, v in dec.items()))
        if self.a.room_kind != "tree|basic":
            fit = room_fit_of(self.a.room_kind)
            rooms_on = self._room_summary(fit, data)
            rk = self.a.room_kind
            kj = ((L.read_json(C.OUT / ("kinds-ahu-dagger.json" if rk.endswith("+dg") else "kinds-ahu.json")) or {})
                  .get("kinds", {}).get(rk[:-3] if rk.endswith("+dg") else rk)
                  if rk.startswith("ahu-") else
                  (L.read_json(C.OUT / "kinds-rooms.json") or {}).get("rooms", {}).get(self.a.room_kind)) or \
                {"lobo": float("nan"), "loco": float("nan")}
            self.log("RESULT", f"room kind {self.a.room_kind}: on-shares {rooms_on}; in-sample model value "
                               f"{100 * RL.room_value(fit, data, list(data.scen)):+.2f}%; honest referee "
                               f"leave-buildings-out {100 * kj['lobo']:+.2f}%, leave-one-climate-out "
                               f"{100 * kj['loco']:+.2f}%")
            st.update(kind=kind, chosen=f"r3 {kind} + rooms {self.a.room_kind}", whole_lobo=res,
                      dec={str(i): v.tolist() for i, v in dec.items()},
                      room_trees=json.loads(json.dumps(fit.trees, default=float)), beta_b=fit.beta_b.tolist(),
                      heldout_room_value=float((kj["lobo"] + kj["loco"]) / 2))
            return
        fit = RL.fit_rooms(data.obs, seed=0)
        # leave-buildings-out calibration of the room model on the real scenarios' buildings
        real_b = sorted({s.split("/")[1] for s in self.P["real"] if s in data.scen})
        rng = np.random.default_rng(0)
        fold = {b: k % 5 for k, b in enumerate(rng.permutation(real_b))}
        cal, val = [], []
        for f in range(5):
            test_b = {b for b, ff in fold.items() if ff == f}
            tr = [o for o in data.obs if o["b"] not in test_b]
            te = [o for o in data.obs if o["b"] in test_b]
            if not te:
                continue
            ff = RL.fit_rooms(tr, seed=f)
            cal.append(RL.calibration(ff, te))
            val.append(RL.room_value(ff, data, sorted({o["s"] for o in te})))
        rooms_on = self._room_summary(fit, data)
        self.log("RESULT", f"room tree: on-shares {rooms_on}; in-sample room value {100 * RL.room_value(fit, data, list(data.scen)):+.2f}%; "
                           f"leave-buildings-out calibration corr {np.mean([c['corr'] for c in cal]):+.2f} slope "
                           f"{np.mean([c['slope'] for c in cal]):.2f}, held-out room value {100 * np.mean(val):+.2f}% "
                           f"({time.time() - t0:.0f} s)")
        st.update(kind=kind, chosen=f"r3 {kind} + room tree", whole_lobo=res,
                  dec={str(i): v.tolist() for i, v in dec.items()},
                  room_trees=json.loads(json.dumps(fit.trees, default=float)), beta_b=fit.beta_b.tolist(),
                  calibration=cal, heldout_room_value=float(np.mean(val)))

    @staticmethod
    def _iv(v):
        on = [g for g, x in zip(WL.GRID, v) if x]
        if not on:
            return "off"
        if len(on) == len(v):
            return "everywhere"
        return f"{on[0]:.1f}..{on[-1]:.1f} C"

    @staticmethod
    def _room_summary(fit, data):
        out = {}
        for j, d in enumerate(C.DELTAS):
            n = on = 0
            for s, sc in data.scen.items():
                idx = np.nonzero(sc["elig"][j])[0]
                X = RL.Data._rows(sc["rr"], idx, sc["oat"])
                n += len(X)
                on += int(fit.on_rows(j, X).sum())
            out[d["name"]] = f"{on / max(1, n):.0%}"
        return out

    # ------------------------------------------------------------ deployment of a fit
    def build(self, name, dec, fit, data):
        gates, on_keys = self.gates_of(dec)
        m = base_model()
        m["rule_config"].update(on_keys)
        m["rule_oat_gates"] = list(m.get("rule_oat_gates", [])) + gates
        bmap = self.P["bmap"]
        m["room_policy"] = compose(bmap, fit)
        m["source"]["geo5_r3"] = {"name": name, "gates": gates, "room_trees": json.loads(json.dumps(fit.trees,
                                                                                                  default=float))}
        sys.path.insert(0, str(W.ROOT / "packages"))
        from smart_building.g36 import rooms
        feats = json.loads((C.G / "tree2/unit_features.json").read_text(encoding="utf-8"))
        bad = checked = 0
        for s in data.scen:
            fb = feats[s.split("/")[1]]
            rows = RM.room_rows(fb)
            nums = [int(r) for r in fb["rooms"]]
            oats = {data.scen[s]["oat"]} | {c + d for c in bmap["oat_cuts"] for d in (-0.01, 0.01)}
            for oat in sorted(oats):
                ch = rooms.option_index(m["room_policy"], rows, nums, oat)
                for k in range(len(nums)):
                    checked += 1
                    bad += m["room_policy"]["options"][ch[k]] != intended(bmap, fit, rows, k, oat)
        if bad:
            raise S.Wake(f"{name}: composed room policy differs from the room tree in {bad} of {checked} checks")
        path = C.OUT / "models" / f"model-{name}.json"
        O.write_model(m, path)
        return path, gates, checked

    def predicted(self, dec, fit, data, scenarios, room_const=None):
        """Cost-weighted predicted change against gt_r5pa2: the scenario's base (intercept - sum of whole effects),
        the switched-on whole-plot effects, the room tree's model value (room_const: instead one honest held-out room
        value for every scenario, for the room kinds whose own model value is not calibrated)."""
        num = den = 0.0
        by = {}
        for o in data.obs:
            if o["share"] is not None:
                by.setdefault(o["s"], []).append(o)
        for s in scenarios:
            os_ = by.get(s)
            if not os_:
                continue
            X = np.array([np.concatenate([[1.0], o["x"], o["share"]]) for o in os_])
            y = np.array([o["y"] for o in os_])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            oat = data.scen[s]["oat"]
            g = int(np.argmin(np.abs(WL.GRID - oat)))
            val = beta[0] - beta[1:1 + RL.NB].sum() + sum(2.0 * beta[1 + b] for b in range(RL.NB) if dec[b][g])
            val += RL.room_value(fit, data, [s]) if room_const is None else room_const
            num += data.scen[s]["cost"] * val
            den += data.scen[s]["cost"]
        return num / den

    def stage_deploy(self, st):
        learn = self.att["stages"]["learn"]
        data = RL.Data()
        dec = {int(k): np.array(v, bool) for k, v in learn["dec"].items()}
        fit = RL.RoomFit(learn["room_trees"], np.array(learn["beta_b"]))
        name = f"gt_r3{self.a.label}1"
        path, gates, checked = self.build(name, dec, fit, data)
        real = [s for s in self.P["real"] if s in data.scen]
        rc = None if getattr(self.a, "room_kind", "tree|basic") == "tree|basic" else learn.get("heldout_room_value")
        pred, pred_real = (self.predicted(dec, fit, data, list(data.scen), rc),
                           self.predicted(dec, fit, data, real, rc))
        self.log("RESULT", f"deploy {name}: new gates {[(g['id'], g['lo'], g['hi']) for g in gates]}; {checked} room x "
                           f"climate checks exact; predicted on the design scenarios {100 * pred:+.2f}% (real "
                           f"{100 * pred_real:+.2f}%) against gt_r5pa2" +
                 ("" if rc is None else f" (rooms: honest held-out value {100 * rc:+.2f}%)"))
        st.update(model=self.ctx.rel(path), name=name, predicted=pred, predicted_real=pred_real, gates=gates,
                  sha=O.behaviour_sha(json.loads(path.read_text())))

    # ------------------------------------------------------------ verify on the design scenarios
    def stage_verify(self, st):
        dep = self.att["stages"]["deploy"]
        data_specs = json.loads((C.OUT / "pipeline.json").read_text(encoding="utf-8"))["specs"]
        pairs = [(dep["model"], s) for s in data_specs]
        R.record(self.ctx, f"verify {dep['name']}", pairs, C.OUT, self.a.workers)
        num = den = 0.0
        for s in data_specs:
            rp = R.record_path(C.OUT, dep["name"], s)
            norm, _ = RL.rule_cost(s)
            if norm is None or not rp.is_file():
                continue
            num += json.loads(rp.read_text())["summary"]["cost"] - norm
            den += norm
        realised = num / den
        # two-sided (review 11-10): at least half the honest prediction, at most 2.5 times it (else miscalibrated)
        ok = dep["predicted"] < 0 and 2.5 * dep["predicted"] <= realised <= 0.5 * dep["predicted"]
        st.update(realised=realised, predicted=dep["predicted"], passed=ok)
        self.log("RESULT", f"verify {dep['name']}: realised {100 * realised:+.2f}% vs predicted "
                           f"{100 * dep['predicted']:+.2f}% against gt_r5pa2 on the design scenarios; "
                           f"{'OK' if ok else 'UNDER HALF OF THE PREDICTION'}")
        if not ok:
            raise S.Wake(f"verify {dep['name']}: realised {100 * realised:+.2f}% vs predicted {100 * dep['predicted']:+.2f}%")

    # ------------------------------------------------------------ selection set (point 2)
    def selection(self, names_models):
        sets = json.loads((C.OUT / "sets.json").read_text(encoding="utf-8"))
        sel = sets["selection"]
        if not R3B.is_file():
            O.write_model(base_model(), R3B)
        pairs = [(m, s) for _, m in names_models for s in sel] + [(self.ctx.rel(R3B), s) for s in sel]
        R.record(self.ctx, "selection " + ", ".join(n for n, _ in names_models), pairs, C.OUT, self.a.workers)
        cost = lambda n, s: json.loads(R.record_path(C.OUT, n, s).read_text())["summary"]["cost"]  # noqa: E731
        out = {}
        rule = np.array([cost("gt_r5pa2", s) for s in sel])
        wub = np.array([cost(WUB, s) for s in sel])
        for n, _ in names_models + [("gt_r3b", None)]:
            c = np.array([cost(n, s) for s in sel])
            d = c - wub
            g = float(d.sum() / rule.sum())
            rng = np.random.default_rng(0)
            bs = [float(d[ii].sum() / rule[ii].sum()) for ii in (rng.integers(0, len(sel), len(sel)) for _ in range(4000))]
            out[n] = {"vs_wub": g, "se": float(np.std(bs)), "vs_rule": float((c - rule).sum() / rule.sum())}
        out[WUB] = {"vs_wub": 0.0, "se": 0.0, "vs_rule": float((wub - rule).sum() / rule.sum())}
        return out

    def stage_select(self, st):
        dep = self.att["stages"]["deploy"]
        res = self.selection([(dep["name"], dep["model"])])
        st.update(result=res)
        self.log("RESULT", "selection set (60 unused official scenarios), change % against gt_wub (SE) / against "
                           "gt_r5pa2: " + ", ".join(f"{n} {100 * v['vs_wub']:+.2f} ({100 * v['se']:.2f}) / "
                                                   f"{100 * v['vs_rule']:+.2f}" for n, v in res.items()))

    # ------------------------------------------------------------ DAgger (point 6)
    def stage_dagger(self, st):
        dep = self.att["stages"]["deploy"]
        if self.a.no_dagger:
            st.update(identical=True, skipped=True)
            self.log("EVENT", "dagger skipped (--no-dagger)")
            return
        learn = self.att["stages"]["learn"]
        data = RL.Data()
        sets = json.loads((C.OUT / "sets.json").read_text(encoding="utf-8"))
        feats = json.loads((C.G / "tree2/unit_features.json").read_text(encoding="utf-8"))
        dec = {int(k): np.array(v, bool) for k, v in learn["dec"].items()}
        fit1 = RL.RoomFit(learn["room_trees"], np.array(learn["beta_b"]))
        added = 0
        for s in sets["selection"]:
            fb = feats.get(s.split("/")[1])
            norm, oat = None, None
            rp_rule = R.record_path(C.OUT, "gt_r5pa2", s)
            if fb is None or not rp_rule.is_file():
                continue
            rec = json.loads(rp_rule.read_text())
            norm, oat = rec["summary"]["cost"], float(np.mean(rec["hourly"]["outside_t"]))
            rr = RM.room_rows(fb)
            cells = [tuple(gs.cell_of(a_, w_)) for a_, w_ in zip(rr["area"], rr["wall"])]
            g = int(np.argmin(np.abs(WL.GRID - oat)))
            x_c = np.array([1.0 if dec[b][g] else -1.0 for b in range(RL.NB)])
            rows_c = {}
            for j, d in enumerate(C.DELTAS):
                el = np.nonzero([d["cells"] is None or list(c) in d["cells"] for c in cells])[0]
                X = RL.Data._rows(rr, el, oat)
                on = fit1.on_rows(j, X)
                rows_c[j] = X[on]
            c_cost = json.loads(R.record_path(C.OUT, dep["name"], s).read_text())["summary"]["cost"]
            b_cost = json.loads(R.record_path(C.OUT, "gt_r3b", s).read_text())["summary"]["cost"]
            data.add(s, s.split("/")[1], norm, oat, c_cost / norm - 1.0, x_c, rows_c)
            data.add(s, s.split("/")[1], norm, oat, b_cost / norm - 1.0, -np.ones(RL.NB),
                     {j: np.zeros((0, len(RL.FEATS))) for j in range(RL.NO)})
            data.scen.setdefault(s, {"oat": oat, "cost": norm, "rr": rr, "n": 2, "elig": [
                np.array([d["cells"] is None or list(c) in d["cells"] for c in cells]) for d in C.DELTAS]})
            added += 2
        fit2 = RL.fit_rooms(data.obs, seed=0)
        name2 = f"gt_r3{self.a.label}2"
        design_scen = [s for s in data.scen if s not in set(sets["selection"])]
        path, gates, checked = self.build(name2, dec, fit2, data)
        pred2 = self.predicted(dec, fit2, data, design_scen)
        same = O.behaviour_sha(json.loads(path.read_text())) == dep["sha"]
        self.log("RESULT", f"dagger: {added} selection episodes added; refit room tree on-shares "
                           f"{self._room_summary(fit2, data)}; {name2} predicted {100 * pred2:+.2f}% on the design "
                           f"scenarios" + ("; IDENTICAL to the first candidate" if same else ""))
        st.update(name=name2, model=self.ctx.rel(path), predicted=pred2, identical=same,
                  room_trees=json.loads(json.dumps(fit2.trees, default=float)))
        if not same:
            res = self.selection([(name2, self.ctx.rel(path))])
            st["selection"] = res
            self.log("RESULT", f"selection set: {name2} {100 * res[name2]['vs_wub']:+.2f}% ({100 * res[name2]['se']:.2f}) "
                               "against gt_wub")

    def stage_choose(self, st):
        s1 = self.att["stages"]["select"]["result"]
        dg = self.att["stages"]["dagger"]
        dep = self.att["stages"]["deploy"]
        cands = [(dep["name"], dep["model"], s1[dep["name"]]["vs_wub"], s1[dep["name"]]["se"])]
        if not dg.get("identical") and dg.get("selection"):
            r2 = dg["selection"][dg["name"]]
            cands.append((dg["name"], dg["model"], r2["vs_wub"], r2["se"]))
        best = min(cands, key=lambda c: c[2])
        bar = 0.0
        if self.a.beat:
            sel = json.loads((C.OUT / "sets.json").read_text(encoding="utf-8"))["selection"]
            cost = lambda n, s: json.loads(R.record_path(C.OUT, n, s).read_text())["summary"]["cost"]  # noqa: E731
            bar = float(sum(cost(self.a.beat, s) - cost(WUB, s) for s in sel) / sum(cost("gt_r5pa2", s) for s in sel))
            self.log("RESULT", f"choose: {self.a.beat} {100 * bar:+.2f}% against gt_wub on the selection set (the bar)")
        st.update(candidates=[c[:4] for c in cands], name=best[0], model=best[1], bar=bar)
        self.log("RESULT", f"choose: {best[0]} ({100 * best[2]:+.2f}% against gt_wub on the selection set, SE "
                           f"{100 * best[3]:.2f})")
        self.name = best[0]
        self.att["stages"]["deploy"]["model"] = best[1]
        if best[2] >= bar:
            raise S.Wake(f"no candidate beats {self.a.beat or WUB} on the selection set (best {best[0]} "
                         f"{100 * best[2]:+.2f}% vs {100 * bar:+.2f}% against gt_wub)")

    def stage_closed(self, st):
        ch = self.att["stages"]["choose"]
        self.name = ch["name"]
        self.att["stages"]["deploy"]["model"] = ch["model"]
        super().stage_closed(st)
        if self.a.beat:
            import geo_pipeline as gp
            ref_model = S.MODELS / f"model-{W.RULE}.json"
            bm = S.MODELS / f"model-{self.a.beat}.json"
            gb = {sub: gp.evaluate(self.a.beat, bm, sub, self.a.workers)["mean"]
                  - gp.evaluate(W.RULE, ref_model, sub, self.a.workers)["mean"] for sub in st["against_base"]}
            mine, theirs = float(np.mean(list(st["against_base"].values()))), float(np.mean(list(gb.values())))
            st.update(beat=self.a.beat, beat_gain=gb, mean=mine, beat_mean=theirs)
            self.log("RESULT", f"closed loop mean of the three sets {mine:+.2f} vs {self.a.beat} {theirs:+.2f} points")
            if mine >= theirs:
                raise S.Wake(f"closed loop {self.name} passes but is not better than {self.a.beat} "
                             f"({mine:+.2f} vs {theirs:+.2f}): not submitted")

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
            except (S.Wake, L.Wake) as w:
                st["wake"] = str(w)
                self.save()
                L.atomic_json(S.STATE / "wake.json", {"time": L.now(), "label": self.a.label, "stage": s,
                                                       "reason": str(w)})
                self.log("WAKE", f"{self.name} {s}: {w}")
                return 3
            except Exception:  # noqa: BLE001
                import traceback
                st["error"] = traceback.format_exc()[-2000:]
                self.save()
                L.atomic_json(S.STATE / "wake.json", {"time": L.now(), "label": self.a.label, "stage": s,
                                                       "reason": "crash", "trace": st["error"]})
                self.log("WAKE", f"{self.name} {s}: crash " + st["error"].replace("\n", " | ")[-600:])
                return 3
            st["done"] = L.now()
            self.save()
            if self.a.until == s:
                return 0
        self.log("EVENT", f"{self.name}: all stages done")
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="a")
    p.add_argument("--until", choices=STAGES)
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--submit", default="auto", choices=("auto", "dry", "none"))
    p.add_argument("--wait-data", action="store_true")
    p.add_argument("--redo", default="")
    p.add_argument("--state", default="stage")
    p.add_argument("--whole-kind", default="")
    p.add_argument("--room-kind", default="tree|basic")
    p.add_argument("--beat", default="")
    p.add_argument("--no-dagger", action="store_true")
    a = p.parse_args()
    S.STATE = C.OUT / a.state
    a.factors, a.kinds, a.levels, a.wait_data_ = "all", "", "default", a.wait_data
    if a.wait_data:
        while not (L.read_json(C.OUT / "pipeline.json") or {}).get("done"):
            if (C.OUT / "wake.json").is_file() and (C.OUT / "wake.json").stat().st_mtime > time.time() - 3600 * 48:
                print("the data collection woke; not starting")
                return 3
            time.sleep(300)
    stage = R3Stage(a)
    for s in [x for x in a.redo.split(",") if x]:
        stage.att["stages"].pop(s, None)
    S.STATE.mkdir(parents=True, exist_ok=True)
    if (S.STATE / "wake.json").is_file():
        os.replace(S.STATE / "wake.json", S.STATE / f"wake-handled-{time.strftime('%Y%m%d-%H%M%S')}.json")
    return stage.run()


if __name__ == "__main__":
    sys.exit(main())
