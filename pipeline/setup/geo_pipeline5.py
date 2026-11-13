"""Pipeline 5 (2026-11-04, docs/plan/pipeline5-plan-20261104.md, docs/plan/geo-pipeline5.md): option levels ->
stage 乙 single-option screening on a stratified 30 (shared) -> per track: stage 丁 rule optimisation (split-plot
design, fit, best combination) -> re-run on the design scenarios (predicted vs realised) -> closed loop -> Kaggle
gate (geo_gate.py, no submission) -> the track's submission rule -> score wait for every submission. Stage 戊 (the
tree) has hooks only (geo5_tree.py; --tree-smoke runs one overlay episode).

Two tracks from the same screening data (user 2026-11-04):
- competition: every option level (physics and competition-only); nominal cost; its Kaggle gate skips the billing
  refusal (GATE_SKIP_BILLING); submitted within the daily quota (4 automatic a day);
- paper: physics levels only, judged on the billing-corrected cost; gated and recorded the same way; submitted after
  the competition submission within the same quota (geo5_submit.decide_paper).
Each track's submit stage is the result of a whole optimisation run: it is submitted without the 0.0001 score
threshold, as the new pure-rule baseline, waiting for the next Kaggle day when today's automatic slots are used (user
2026-11-05); the threshold is still computed and logged.
Order: options, screen, competition.{design, verify, closed_loop, gate, submit}, paper.{...}, score.

Two chances per stage: a gate that fails, a realised result more than half off the prediction, a crash, host memory
over 18 GB or a job over 2x its estimate writes <state>/wake.json and exits with code 3 (code 4 when it was the
stage's second attempt). The coordinator, woken by the exit of its background task, diagnoses, fixes, and continues
with --resume --attempt 2. Runs in the foreground of the launching wsl.exe session (never daemonised: WSL stops the
distro when no session is connected).

python3 -B pipeline/launch/launch.py pipeline/setup/geo_pipeline5.py [--workers 12] [--tracks competition,paper]
        [--submit dry|auto] [--resume --attempt 2] [--redo stage,...]   (SB_FASTSIM=rom)
Test hooks (smoke only): --specs, --only, --max-steps, --keep-all, --fake-closed-loop, --fake-gate, --fake-count,
--stop-after-episodes, --until, --tree-smoke."""
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
import geo5_design as D  # noqa: E402
import geo5_lib as L  # noqa: E402
from geo5_lib import Wake  # noqa: E402
import geo5_options as O  # noqa: E402
import geo5_record as R  # noqa: E402
import geo5_screen as S  # noqa: E402
import geo5_submit as SUB  # noqa: E402

TRACKS = ("competition", "paper")
TRACK_STAGES = ("design", "verify", "closed_loop", "gate", "submit")
# measured in pipeline 4 for gt_u1b (data/runs/b2rl/geo/pipeline.log 2026-11-02/03): closed loop 21:12-21:28,
# Kaggle gate 21:28-00:35, submit_candidate 00:35-00:59; scoring about 8 h (memory kaggle-scoring-wait)
SEEDS = {"closed_loop": 16.0, "gate": 187.0, "submit": 24.0, "score": 480.0}
MODELS = ROOT / "data/runs/b2rl/search/models"
HOLDOUT, INNER, S4 = "train_ceiling_rest60", "inner", "s4_first"


def stage_list(tracks):
    out = ["options", "screen"]
    for t in TRACKS:
        if t in tracks:
            out += [f"{t}.{s}" for s in TRACK_STAGES]
    return out + ["score"]


class Pipeline:
    def __init__(self, a):
        self.a = a
        self.ctx = L.Ctx(ROOT / a.state)
        st = self.ctx.state
        if not st:
            st.update({"started": L.now(), "options_file": a.options, "stages": {}, "wakes": [], "args": vars(a)})
        for k, v in SEEDS.items():
            if not self.ctx.estimate(k):
                self.ctx.timing[k] = {"minutes": v, "measured": "pipeline 4, gt_u1b"}
        self.spec = O.load(ROOT / a.options)
        if a.only:
            keep = set(a.only.split(","))
            self.spec["options"] = [lv for lv in self.spec["options"] if lv["id"] in keep]
        self.levels = O.by_id(self.spec)
        self.base = O.base_model(self.spec)
        self.tracks = [t for t in TRACKS if t in a.tracks.split(",")]
        self.stages = stage_list(self.tracks)
        if a.specs:
            self.specs = a.specs.split(",")
            self.weights = {s: 1.0 for s in self.specs}
        else:
            path = ROOT / a.screen_set
            sc = L.read_json(path) or S.build_screen30(path)
            self.specs, self.weights = sc["specs"], sc["weights"]
        st["specs"], st["weights"] = self.specs, self.weights
        self.meta = st.setdefault("meta", {})
        for s in self.specs:
            if s not in self.meta:
                se, month = S.season(s)
                _, r = S.base_full_record(s)
                oat = round(float(np.mean(r["hourly"]["outside_t"])), 2) if r else None
                self.meta[s] = {"city": S.city(s), "season": se, "month": month, "oat_c": oat}
        self.ms = a.max_steps or None
        self.prefix = "smk_" if self.ms else ""
        self.ctx.summary_fn = self.summary
        # option review 2026-11-05 (docs/plan/option-review-20261105.md §三): the design takes its factors and the
        # track's base levels (always on, never factors: o546_1 for competition, o542_1 for paper, the fold set)
        # from the coordinator's reviewed list, not from the screen verdicts
        self.review = L.read_json(ROOT / a.review) if getattr(a, "review", None) else None
        if self.review:
            for t in TRACKS:
                for lid in (self.review.get(t, {}).get("base", []) + self.review.get(t, {}).get("factors", [])
                            + self.review.get(t, {}).get("loop_reference", [])):
                    if lid not in self.levels:
                        raise SystemExit(f"review {a.review}: {t} level {lid} is not in {a.options}")

    # ---------------------------------------------------------------- helpers
    @property
    def st(self):
        return self.ctx.state

    def stage(self, name):
        return self.st["stages"].setdefault(name, {"attempt": 1, "status": "pending"})

    def log(self, kind, text):
        self.ctx.log(kind, text)

    def model_path(self, name):
        return self.ctx.dir / "models" / f"model-{name}.json"

    def rel(self, p):
        return Path(p).resolve().relative_to(ROOT).as_posix()

    def record_of(self, name, spec):
        """Record + billing of a run name on a scenario: this state dir first; for whole-month runs also the reused
        records (base: options gt_u1b, then geo4 gt_u2a_pb11; levels: an earlier model with the same behaviour)."""
        r = L.read_json(R.record_path(self.ctx.dir, name, spec, self.ms))
        if r is not None:
            return r, L.read_json(R.billing_path(self.ctx.dir, name, spec, self.ms))
        alias = self.st.get("alias", {}).get(name)
        if alias and alias != name:
            return self.record_of(alias, spec)
        if self.ms:
            return None, None
        if name == "gt_u1b":
            for _, d, n in S.BASE_SOURCES:
                r = L.read_json(d / "records" / R.key(n, spec))
                if r is not None:
                    return r, L.read_json(d / "billing" / R.key(n, spec))
            return None, None
        for reuse in (self.st.get("options", {}).get(name) or {}).get("reuse", []):
            d = O.REUSE_RECORD_DIRS["options"]
            r = L.read_json(d / "records" / R.key(reuse, spec))
            if r is not None:
                return r, L.read_json(d / "billing" / R.key(reuse, spec))
        return None, None

    def entries(self, name):
        out = {}
        for s in self.specs:
            r, b = self.record_of(name, s)
            out[s] = S.entry(r, b) if r is not None else None
        return out

    def cost(self, name, spec, track):
        """The track's cost of a run on a scenario: nominal (competition) or billing-corrected (paper; nominal when
        the run has no billing audit)."""
        r, b = self.record_of(name, spec)
        if track == "paper":
            h = S.honest(S.entry(r, b))
            if h is not None:
                return h
        return r["summary"]["cost"]

    def ensure(self, job, names):
        """Record every (name, scenario) with no record (own or reused)."""
        pairs = []
        for n in names:
            path = self.model_path(n) if n != "gt_u1b" else ROOT / self.spec["base"]["model"]
            for s in self.specs:
                if self.record_of(n, s)[0] is None:
                    pairs.append((self.rel(path), s))
        have = len(names) * len(self.specs) - len(pairs)
        self.log("EVENT", f"{job}: {len(names) * len(self.specs)} episodes needed, {have} already on disk or reused, "
                          f"{len(pairs)} to record")
        if pairs:
            R.record(self.ctx, job, pairs, self.ctx.dir, self.a.workers, self.ms, self.a.stop_after_episodes)

    def register(self, name, model):
        """Write a run model; a run that behaves like an earlier one in this state dir reuses its records."""
        sha = O.behaviour_sha(model)
        shas = self.st.setdefault("shas", {})
        same = next((n for n, h in shas.items() if h == sha and n != name), None)
        if name != "gt_u1b" and sha == O.behaviour_sha(self.base):
            same = "gt_u1b"
        shas[name] = sha
        if same:
            self.st.setdefault("alias", {})[name] = same
        O.write_model(model, self.model_path(name))
        return same

    def letter(self, track):
        return "ab"[min(self.stage(f"{track}.design")["attempt"], 2) - 1]

    def base_levels(self, track):
        """The track's always-on levels from the reviewed list ([] without one: the base is gt_u1b)."""
        return [self.levels[i] for i in ((self.review or {}).get(track) or {}).get("base", [])]

    def track_base(self, track):
        """Name of the track's base run (gt_u1b + its always-on levels), registered and recorded; gt_u1b when the
        track has none."""
        lvs = self.base_levels(track)
        if not lvs:
            return "gt_u1b"
        name = f"{self.prefix}g5{track[0]}base"
        model = O.build(self.base, lvs, name)
        model["source"]["geo5"]["track"] = track
        self.register(name, model)
        self.ensure(f"{track} base", [name])
        return name

    def final_name(self, track):
        return f"{self.prefix}gt_r5{track[0]}{self.letter(track)}"

    # ---------------------------------------------------------------- stage: options
    def stage_options(self, st, track=None):
        bad = O.conflicts(self.spec["options"])
        rows = {}
        from smart_building.policies.b2rl import load_model
        index = O.reuse_index() if not self.ms else {}
        for lid, lv in self.levels.items():
            model = O.build(self.base, [lv], lid)
            self.register(lid, model)
            load_model(self.model_path(lid))
            sha = O.behaviour_sha(model)
            reuse = [n for n in index.get(sha, []) if n != "gt_u1b"]
            have = [n for n in reuse if all((O.REUSE_RECORD_DIRS["options"] / "records" / R.key(n, s)).is_file()
                                            for s in self.specs)]
            rows[lid] = {"model": self.rel(self.model_path(lid)), "behaviour_sha": sha, "reuse": have[:1],
                         "layer": lv["layer"], "category": lv["category"], "option": lv["option"]}
        self.st["options"] = rows
        st["conflicts"] = [list(c) for c in bad]
        self.log("EVENT", f"options: {len(rows)} levels built on {self.spec['base']['name']} "
                          f"({sum(lv['category'] == 'competition' for lv in self.levels.values())} competition-only); "
                          f"whole-month records reused for {sum(bool(r['reuse']) for r in rows.values())}; "
                          f"exclusive pairs {len(bad)}")

    # ---------------------------------------------------------------- stage 乙: screen (shared by the tracks)
    def stage_screen(self, st, track=None):
        """Batches of levels (options.json order); after each batch the cumulative report and the per-track direction
        summaries (direction_<track>_batch<k>.json / .md); after batches 1 and 2 an information wake
        (paper_direction_ready, user 2026-11-04): not a failure, a plain --resume continues."""
        ids = list(self.levels)
        marks = [self.levels[i].get("batch") for i in ids]
        if all(m is not None for m in marks):     # the option set's own screening batches (option-set-20261104.md)
            batches = [[i for i, m in zip(ids, marks) if m == b] for b in sorted(set(marks))]
        else:
            nb = max(1, min(self.a.batches, len(ids)))
            size = -(-len(ids) // nb)
            batches = [ids[i:i + size] for i in range(0, len(ids), size)]
        nb = len(batches)
        self.log("EVENT", f"screen starts: {len(ids)} levels x {len(self.specs)} scenarios in {nb} batches"
                          f"{' (first %d steps)' % self.ms if self.ms else ''}")
        st.setdefault("directions", [])
        st.setdefault("info_wakes", [])
        for k, chunk in enumerate(batches, start=1):
            self.ensure(f"screen batch {k}/{nb}", ["gt_u1b"] + chunk)
            done = [x for b in batches[:k] for x in b]
            self.screen_report(st, done, final=k == nb)
            if k not in st["directions"]:
                paths = []
                for t in TRACKS:
                    d, md = S.direction(L.read_json(ROOT / st["report"]), self.levels, t, k, nb)
                    p = self.ctx.dir / f"direction_{t}_batch{k}.json"
                    L.atomic_json(p, d)
                    p.with_suffix(".md").write_text(md, encoding="utf-8", newline="\n")
                    paths += [p, p.with_suffix(".md")]
                    self.log("RESULT", f"direction {t} after batch {k}/{nb}: {len(d['helps'])} levels help, "
                                       f"{len(d['conditional'])} conditional; estimated saving "
                                       f"{d['estimated_saving_pct']:+.2f}% ({d['metric']}, additive), ceiling "
                                       f"{d['ceiling_pct']:+.2f}%")
                st["directions"].append(k)
                st.setdefault("direction_files", {})[str(k)] = [self.rel(p) for p in paths]
                self.ctx.save()
            if k <= 2 and k not in st["info_wakes"]:
                st["info_wakes"].append(k)
                self.ctx.save()
                files = st["direction_files"][str(k)]
                raise Wake("paper_direction_ready", f"screen batch {k}/{nb} done: direction summaries per track "
                                                    f"({', '.join(files)}); not a failure, --resume continues",
                           {"batch": k, "batches": nb, "files": files}, [ROOT / f for f in files], info=True)
        self.log("EVENT", "screen survivors: " + "; ".join(f"{t} {st['survivors'][t]}" for t in TRACKS))
        if not any(st["survivors"][t] for t in self.tracks):
            raise Wake("gate", "screen: no level survived the harm filter on any track", {"table": st["table"]},
                       [ROOT / st["report"]])

    def screen_report(self, st, ids, final):
        base = self.entries("gt_u1b")
        levels = {}
        for lid in ids:
            lv = self.levels[lid]
            ref = lv.get("compare_with")                  # review S1: a stacked level against the level it sits on
            basis = self.entries(ref) if ref and ref in self.levels else base
            res = S.compare(basis, self.entries(lid), self.specs, self.weights, self.meta, lv.get("conditions"))
            if res is not None and ref:
                res["compared_with"] = ref
            row = {"layer": lv["layer"], "category": lv["category"], "option": lv["option"], "result": res}
            for t in TRACKS:
                row[t] = S.verdict(res, lv, t) if res is not None else {"keep": False, "status": "no data"}
            levels[lid] = row
        report = {"made": L.now(), "base": "gt_u1b", "scenarios": len(self.specs), "max_steps": self.ms,
                  "levels": levels}
        best = {}
        for t in TRACKS:          # the stage's best single version per track
            cand = [(lid, (r["result"].get("billing") or r["result"])["cw_pct"] if t == "paper"
                     else r["result"]["cw_pct"]) for lid, r in levels.items() if r[t]["keep"]]
            best[t] = min(cand, key=lambda x: x[1]) if cand else None
        report["best_single"] = best
        out = self.ctx.dir / "screen"
        L.atomic_json(out / "report.json", report)
        (out / "report.md").write_text(S.table_md(report), encoding="utf-8", newline="\n")
        st["report"] = self.rel(out / "report.json")
        st["best_single"] = best
        st["table"] = {lid: {"cw_pct": (r["result"] or {}).get("cw_pct"),
                             "billing_cw_pct": ((r["result"] or {}).get("billing") or {}).get("cw_pct"),
                             "oracle_pct": (r["result"] or {}).get("oracle_pct"),
                             "paper": r["paper"]["status"], "competition": r["competition"]["status"]}
                       for lid, r in levels.items()}
        logged = st.setdefault("logged", [])
        for lid, r in st["table"].items():
            if lid in logged:
                continue
            logged.append(lid)
            if r["cw_pct"] is None:
                self.log("RESULT", f"screen {lid}: no data")
                continue
            bill = "—" if r["billing_cw_pct"] is None else f"{r['billing_cw_pct']:+.3f}%"
            self.log("RESULT", f"screen {lid}: cost-weighted {r['cw_pct']:+.3f}% (billing-corrected {bill}), upper "
                               f"bound {r['oracle_pct']:+.3f}%; paper {r['paper']}, competition {r['competition']}")
        st["survivors"] = {t: [lid for lid, r in levels.items() if r[t]["keep"]] for t in TRACKS}
        if self.a.keep_all:
            st["survivors"] = {t: [lid for lid in levels if t == "competition"
                                   or self.levels[lid]["category"] == "physics"] for t in TRACKS}
            if final:
                self.log("EVENT", "screen: --keep-all (test hook): every level of the track goes to its design")

    # ---------------------------------------------------------------- stage 丁: design + fit (per track)
    def stage_design(self, st, track):
        scr = self.stage("screen")
        rep = L.read_json(ROOT / scr["report"])

        def score(lid):
            r = rep["levels"][lid]["result"]
            return (r.get("billing") or r)["cw_pct"] if track == "paper" else r["cw_pct"]

        reviewed = (self.review or {}).get(track)
        if reviewed is not None:                       # the coordinator's list (review 2026-11-05)
            chosen = list(reviewed.get("factors", []))
        else:
            best = {}
            for lid in scr["survivors"][track]:
                opt = self.levels[lid]["option"]
                if opt not in best or score(lid) < score(best[opt]):
                    best[opt] = lid
            chosen = sorted(best.values(), key=score)
        if self.a.combo_factors:
            chosen = [c for c in self.a.combo_factors.split(",") if c in self.levels]
        if track == "paper":
            chosen = [c for c in chosen if self.levels[c]["category"] == "physics"]
        base_lv = self.base_levels(track)
        if not chosen:
            raise Wake("gate", f"{track} design: no factor", {"survivors": scr["survivors"], "review": reviewed})
        dropped = []
        clash = [c for c in O.conflicts(base_lv + [self.levels[c] for c in chosen])
                 if {c[0], c[1]} & {lv["id"] for lv in base_lv}]
        if clash:                                      # the reviewed list itself is inconsistent: never guess
            raise Wake("gate", f"{track} design: a factor clashes with the track's base levels: {clash}",
                       {"clash": clash})
        for x, y, why in O.conflicts([self.levels[c] for c in chosen]):
            loser = y if score(x) <= score(y) else x
            if loser in chosen:
                chosen.remove(loser)
                dropped.append([loser, why])
        tb = self.track_base(track)
        whole = [c for c in chosen if self.levels[c]["layer"] in O.WHOLE]
        sub = [c for c in chosen if self.levels[c]["layer"] == "room"]
        des = D.make(whole, sub, max_runs=self.a.max_runs)
        runs = []
        for r, row in enumerate(des["rows"]):
            on = [n for n, v in zip(des["names"], row) if v > 0]
            name = f"{self.prefix}g5{track[0]}d{self.letter(track)}{r:02d}"
            model = O.build(self.base, base_lv + [self.levels[n] for n in on], name)
            same = self.register(name, model)
            runs.append({"name": name, "row": row, "on": on, "same_as": same})
        st.update(factors=des["names"], whole=whole, sub=sub, conflicts_dropped=dropped, track_base=tb,
                  base_levels=[lv["id"] for lv in base_lv],
                  design={k: v for k, v in des.items() if k != "rows"}, runs=runs)
        self.log("EVENT", f"{track} design: {len(whole)} whole-plot + {len(sub)} sub-plot factors -> {des['runs']} "
                          f"runs ({des['base_runs']} base{' + sub-plot fold-over' if des['foldover'] else ''}), main "
                          f"effects clear {des['me_clear']}, {len(des['pairs'])} of "
                          f"{len(chosen) * (len(chosen) - 1) // 2} pairwise interactions estimable; "
                          f"{sum(1 for x in runs if x['same_as'])} runs reuse earlier records")
        self.ctx.save()
        self.ensure(f"{track} design", [x["name"] for x in runs])
        y, w = {}, {}
        for s in self.specs:
            b = self.cost(tb, s, track)                # the all-off row is the track's base run
            y[s] = [self.cost(x["name"], s, track) / b - 1.0 for x in runs] + [0.0]
            w[s] = self.weights.get(s, 1.0) * b
        g0 = np.array([self.cost("gt_u1b", s, track) for s in self.specs])
        t0 = np.array([self.cost(tb, s, track) for s in self.specs])
        sw = np.array([self.weights.get(s, 1.0) for s in self.specs])
        base_rel = float(np.sum(sw * (t0 - g0)) / np.sum(sw * g0))   # the track base against gt_u1b
        fit = D.fit(des, y, w)
        x, pred = D.choose(fit, len(des["names"]))
        if self.a.combo:
            want = set(self.a.combo.split(","))
            x = [1 if n in want else -1 for n in des["names"]]
            pred = D.predict(fit, x)
        on = [n for n, v in zip(des["names"], x) if v > 0]
        per = D.per_scenario_prediction(fit, x)
        city = {}
        for s, p in per.items():
            city.setdefault(self.meta[s]["city"], []).append((w[s], p))
        st["fit"] = {"terms": [[des["names"][i] for i in t] for t in fit["terms"]], "mean": fit["mean"],
                     "se": fit["se"], "use": fit["use"], "n": fit["n"], "n_eff": fit["n_eff"]}
        st.update(chosen=on, chosen_x=x, predicted_pct=100.0 * pred, metric="billing-corrected" if track == "paper"
                  else "nominal", predicted_city_pct={c: 100.0 * sum(a * b for a, b in v) / sum(a for a, _ in v)
                                                      for c, v in city.items()},
                  base_vs_gt_u1b_pct=100.0 * base_rel,
                  predicted_vs_gt_u1b_pct=100.0 * ((1.0 + base_rel) * (1.0 + pred) - 1.0))
        self.log("RESULT", f"{track} design fit ({fit['n']} scenarios, {st['metric']} cost): chosen "
                           f"{on or 'nothing'}, predicted cost-weighted {100 * pred:+.3f}% vs the track base {tb} "
                           f"({100 * base_rel:+.3f}% vs gt_u1b; together about "
                           f"{st['predicted_vs_gt_u1b_pct']:+.3f}%); by city "
                           + ", ".join(f"{c} {v:+.2f}%" for c, v in sorted(st["predicted_city_pct"].items())))
        if not on and not base_lv:
            raise Wake("gate", f"{track} design: no combination predicted better than the base", {"fit": st["fit"]})

    # ---------------------------------------------------------------- verify on the design scenarios
    def stage_verify(self, st, track):
        des = self.stage(f"{track}.design")
        name = self.final_name(track)
        base_ids = des.get("base_levels", [])
        tb = des.get("track_base", "gt_u1b")           # realised against the base the prediction is relative to
        now = [lv["id"] for lv in self.base_levels(track)]
        if self.review and now != base_ids:
            # the reviewed base changed after the design (coordinator fix after the closed loop, user 2026-11-05):
            # the design's combination on the current base, realised against the current base alone, so the
            # prediction (the factors' effect over the base) still applies; new names, never the old records
            self.log("EVENT", f"{track} verify: the reviewed base changed after the design: {base_ids} -> {now}")
            base_ids, name, tb = now, f"{name}2", f"{self.prefix}g5{track[0]}base2"
            bm = O.build(self.base, self.base_levels(track), tb)
            bm["source"]["geo5"]["track"] = track
            self.register(tb, bm)
            self.ensure(f"{track} base2", [tb])
        model = O.build(self.base, [self.levels[n] for n in base_ids + des["chosen"]], name)
        model["source"]["geo5"]["track"] = track
        old = self.st.get("shas", {}).get(name)
        if old and old != O.behaviour_sha(model):
            raise Wake("crash", f"{track} verify: {name} already names a model with another behaviour; its records "
                                "would be reused for a different model")
        same = self.register(name, model)
        self.log("EVENT", f"{track} verify {name} (base {base_ids} + {des['chosen']}) on the {len(self.specs)} "
                          "design scenarios" + (f"; behaves like {same}, its records stand in" if same else ""))
        self.ensure(f"{track} verify", [name])
        b = np.array([self.cost(tb, s, track) for s in self.specs])
        o = np.array([self.cost(name, s, track) for s in self.specs])
        bn = np.array([self.cost("gt_u1b", s, "competition") for s in self.specs])
        on_ = np.array([self.cost(name, s, "competition") for s in self.specs])
        w = np.array([self.weights.get(s, 1.0) for s in self.specs])
        realised = 100.0 * float(np.sum(w * (o - b)) / np.sum(w * b))
        nominal = 100.0 * float(np.sum(w * (on_ - bn)) / np.sum(w * bn))
        g = np.array([self.cost("gt_u1b", s, track) for s in self.specs])
        vs_gt = 100.0 * float(np.sum(w * (o - g)) / np.sum(w * g))
        pred = des["predicted_pct"]
        st.update(name=name, model=self.rel(self.model_path(name)), realised_pct=realised, predicted_pct=pred,
                  nominal_pct=nominal, realised_vs_gt_u1b_pct=vs_gt, track_base=tb, metric=des["metric"],
                  same_as=same, wins=int(np.sum(o < b)), losses=int(np.sum(o > b)),
                  behaviour_sha=O.behaviour_sha(model))
        self.log("RESULT", f"{track} verify {name}: realised {realised:+.3f}% vs predicted {pred:+.3f}% against the "
                           f"track base {tb} ({des['metric']}, cost-weighted; {vs_gt:+.3f}% vs gt_u1b, nominal "
                           f"{nominal:+.3f}% vs gt_u1b; {st['wins']} better / {st['losses']} worse)")
        if tb != des.get("track_base", "gt_u1b"):
            # the base changed after the design: the factors' effect over the design's base does not carry to the
            # new base (a removed base level can overlap the factors: 11/05, the humid low supply air did nothing
            # with the chosen factors on), so the check is the whole combination against gt_u1b, which the design
            # also predicted
            realised, pred = vs_gt, des.get("predicted_vs_gt_u1b_pct", pred)
            st.update(check="against gt_u1b (base changed after the design)", check_realised_pct=realised,
                      check_predicted_pct=pred)
            self.log("EVENT", f"{track} verify {name}: base changed after the design, so the check is against gt_u1b: "
                              f"realised {realised:+.3f}% vs predicted {pred:+.3f}%")
        tol = self.a.mismatch * abs(pred)
        if abs(realised - pred) > tol:
            raise Wake("mismatch", f"{track} verify: realised {realised:+.3f}% differs from the predicted {pred:+.3f}% "
                                   f"by more than half ({abs(realised - pred):.3f} > {tol:.3f} points)",
                       {"realised_pct": realised, "predicted_pct": pred}, [self.model_path(name)])

    # ---------------------------------------------------------------- closed loop
    def stage_closed_loop(self, st, track):
        v = self.stage(f"{track}.verify")
        name, model = v["name"], ROOT / v["model"]
        if self.a.fake_closed_loop:
            st.update(fake=True, passed=self.a.fake_closed_loop == "pass")
            self.log("EVENT", f"{track} closed loop {name}: FAKE ({self.a.fake_closed_loop}, test hook)")
            if not st["passed"]:
                raise Wake("gate", f"{track} closed loop {name}: fake failure (test hook)")
            return
        final = MODELS / f"model-{name}.json"
        shutil.copyfile(model, final)
        # the reference: gt_u1b, or gt_u1b + the track's mandatory levels when the reviewed list names them (paper:
        # the ventilation code, which gt_u1b does not keep; user 2026-11-05)
        ref_ids = ((self.review or {}).get(track) or {}).get("loop_reference", [])
        ref, ref_model = "gt_u1b", MODELS / "model-gt_u1b.json"
        if ref_ids:
            ref = f"{self.prefix}g5{track[0]}loopref"
            rm = O.build(self.base, [self.levels[i] for i in ref_ids], ref)
            rm["source"]["geo5"]["track"] = track
            old = self.st.get("shas", {}).get(ref)
            if old and old != O.behaviour_sha(rm):
                raise Wake("crash", f"{track} closed loop: {ref} already names a model with another behaviour")
            self.st.setdefault("shas", {})[ref] = O.behaviour_sha(rm)
            ref_model = MODELS / f"model-{ref}.json"
            O.write_model(rm, ref_model)
        import geo_pipeline as gp
        res, gain = {}, {}
        for subset in (HOLDOUT, INNER, S4):
            for n, m in ((name, final), (ref, ref_model)):
                out = gp.OUT / subset / f"{n}.json"
                if not out.is_file():
                    env = {k: x for k, x in os.environ.items() if k != "SB_FASTSIM"}
                    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    el = L.run_monitored(self.ctx, f"closed loop {n} {subset}",
                                         ["nice", "-n", "5", "python3", "-B", "scripts/b2rl.py", "eval", "--policy",
                                          self.rel(m), "--set", subset, "--workers", str(L.worker_count(self.a.workers)),
                                          "--fast", "rom", "--out", self.rel(out)], env=env,
                                         estimate_min=self.ctx.estimate("closed_loop") / 3.0)
                    self.ctx.measured("closed_loop", 3.0 * el)
            res[subset] = gp.evaluate(name, final, subset, self.a.workers)
            gain[subset] = res[subset]["mean"] - gp.evaluate(ref, ref_model, subset, self.a.workers)["mean"]
        vs_gt = {}
        if ref != "gt_u1b":                            # reported only
            for subset in (HOLDOUT, INNER, S4):
                vs_gt[subset] = res[subset]["mean"] - gp.evaluate("gt_u1b", MODELS / "model-gt_u1b.json", subset,
                                                                  self.a.workers)["mean"]
            self.log("RESULT", f"{track} closed loop {name} against gt_u1b (reported only): held-out 60 "
                               f"{vs_gt[HOLDOUT]:+.2f}, INNER 29 {vs_gt[INNER]:+.2f}, S4 first 30 {vs_gt[S4]:+.2f} points")
        passed = gain[HOLDOUT] <= self.a.loop_need and gain[INNER] <= self.a.loop_need and gain[S4] <= self.a.loop_s4_max
        st.update(result=res, against_base=gain, passed=passed, model=self.rel(final), reference=ref,
                  reference_levels=ref_ids, against_gt_u1b=vs_gt or gain)
        self.log("RESULT", f"{track} closed loop {name} against {ref}{f' (gt_u1b + {ref_ids})' if ref_ids else ''} "
                           "(nominal cost): held-out 60 "
                           f"{gain[HOLDOUT]:+.2f}, INNER 29 {gain[INNER]:+.2f}, S4 first 30 {gain[S4]:+.2f} points; "
                           f"{'PASSED' if passed else 'below the pass rule'}")
        if not passed:
            raise Wake("gate", f"{track} closed loop {name} below the pass rule (held-out and INNER <= "
                               f"{self.a.loop_need}, S4 <= {self.a.loop_s4_max} points)", {"against_base": gain})

    # ---------------------------------------------------------------- Kaggle gate
    def stage_gate(self, st, track):
        name = self.stage(f"{track}.verify")["name"]
        model = self.stage(f"{track}.closed_loop").get("model") or self.stage(f"{track}.verify")["model"]
        if self.a.fake_gate:
            ref = SUB.gate_of("gt-u1b")
            gate = json.loads(json.dumps(ref))
            gate["job"] = f"jg-b2rl-{name.replace('_', '-')}"
            gate["s4_first"]["rel"] = self.a.fake_gate_s4 + (0.0 if track == "competition" else 0.004)
            passed = self.a.fake_gate == "pass"
            gate["passed"] = gate["inner"]["gate"] = gate["s4_first"]["gate"] = passed
            res = {"name": name, "passed": passed, "reason": f"FAKE {self.a.fake_gate} (test hook)", "gate": gate,
                   "fake": True}
            L.atomic_json(self.ctx.dir / "gate" / f"{name}.json", res)
            self.log("EVENT", f"{track} gate {name}: FAKE {self.a.fake_gate} (test hook)")
        else:
            out = f"{self.rel(self.ctx.dir)}/gate/{name}.json"
            env = dict(os.environ, GATE_NO_SUBMIT="1", GATE_RESULT=out)
            if track == "competition":
                env["GATE_SKIP_BILLING"] = "1"           # the billing sawtooth may be on (user 2026-11-04)
            else:
                env.pop("GATE_SKIP_BILLING", None)
            el = L.run_monitored(self.ctx, f"{track} gate {name}",
                                 ["python3", "-B", "pipeline/setup/geo_gate.py", name, model], env=env,
                                 estimate_min=self.ctx.estimate("gate"))
            self.ctx.measured("gate", el)
            res = L.read_json(ROOT / out) or {"passed": False, "reason": "geo_gate.py wrote no verdict"}
        st.update(verdict=res)
        if not res.get("passed"):
            raise Wake("gate", f"{track} Kaggle gate {name}: {res.get('reason')}", {"verdict": res},
                       [self.ctx.dir / "gate" / f"{name}.json", ROOT / "data/runs/b2rl/geo/pipeline.log"])

    # ---------------------------------------------------------------- submission
    def count(self):
        if self.a.fake_count is not None:
            return self.a.fake_count + sum(1 for x in self.st.get("submissions", []) if x.get("fake")), "fake"
        try:
            n, day, _ = SUB.count_today()
            return n, day
        except Exception as error:  # noqa: BLE001 - the decision then refuses (count unknown)
            self.log("ERROR", f"submission count: {repr(error)[-300:]}")
            return None, None

    def stage_submit(self, st, track):
        name = self.stage(f"{track}.verify")["name"]
        model = self.stage(f"{track}.closed_loop").get("model") or self.stage(f"{track}.verify")["model"]
        verdict = self.stage(f"{track}.gate")["verdict"]
        gate = verdict["gate"]
        subs = self.st.setdefault("submissions", [])
        paper_refs = [x["ref"] for x in subs if x["track"] == "paper" and x.get("ref")]
        count, day = self.count()
        if track == "competition":
            dec = SUB.decide(gate, count, exclude=paper_refs)
        else:
            comp = self.stage("competition.verify")
            duplicate = comp.get("behaviour_sha") == self.stage("paper.verify").get("behaviour_sha")
            rows = SUB.leaderboard()
            scored = [r for r in rows if r["ref"] in paper_refs and r["public"] is not None and r["s4_rel"] is not None]
            dec = SUB.decide_paper(gate, count, duplicate, scored, rows, exclude=paper_refs)
        dec["backtest"] = SUB.backtest()
        gates_file = verdict.get("gates_file") or self.rel(self.ctx.dir / "gate" / f"gates-{name}.json")
        if not verdict.get("gates_file"):
            L.atomic_json(ROOT / gates_file, gate)
        label = ("competition version (competition-only options allowed)" if track == "competition"
                 else "paper version (physics options only)")
        desc = f"gt_u1b + pipeline-5 rule options {self.stage(f'{track}.design')['chosen']}, {label}"
        dec["commands"] = SUB.commands(name, model, gates_file, desc)
        dec.update(mode=self.a.submit, day=day, name=name, track=track)
        st["decision"] = dec
        self.log("RESULT", f"{track} submission rule for {name}: estimated public {dec['estimate']:.5f} (INNER-based "
                           f"{dec['estimate_inner']:.5f}), competition reference {dec['reference']:.5f} (anchor "
                           f"{dec['anchor']['slug']} {dec['anchor']['public']}), threshold {dec['threshold']:.5f}"
                           " (reported only: a whole optimisation run is submitted without it); "
                           f"today's submissions {count} (UTC day {day}); "
                           + ("SUBMIT" if dec["submit"] else "no submission: " + "; ".join(dec["reasons"])))
        L.atomic_json(self.ctx.dir / "submit" / f"{name}-{self.a.submit}.json", dec)
        if not dec["submit"]:
            return
        if self.a.submit != "auto":
            subs.append({"track": track, "name": name, "ref": None, "estimate": dec["estimate"], "dry": True,
                         "fake": self.a.fake_count is not None, "time": L.now()})
            self.log("EVENT", f"{track} submission DRY RUN: would run {' '.join(dec['commands']['submit'][:5])} ... "
                              "then score_wait.py; nothing submitted")
            return
        wait_extra = 0.0
        if dec["quota_wait"]:
            now = datetime.datetime.now(datetime.timezone.utc)
            wait_extra = ((now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
                          - now).total_seconds() / 60.0
        el = L.run_monitored(self.ctx, f"{track} submit {name}", dec["commands"]["submit"],
                             estimate_min=self.ctx.estimate("submit") + wait_extra)
        ref = SUB.submitted_ref(name)
        if not ref:
            raise Wake("crash", f"submit_candidate.py ended without a submission ref for {name}",
                       paths=[ROOT / "pipeline/setup/submit.log"])
        self.ctx.measured("submit", el - wait_extra)
        subs.append({"track": track, "name": name, "ref": ref, "estimate": dec["estimate"], "time": L.now()})
        st["submitted_ref"] = ref
        self.log("RESULT", f"{track} submitted {name}: ref {ref}; score in about 8 hours (stage score)")

    def stage_score(self, st, track=None):
        """Hourly score wait (score_wait.py) for every real submission of this run, then the public scores."""
        todo = [x for x in self.st.get("submissions", []) if x.get("ref") and x.get("public") is None]
        if not todo:
            self.log("EVENT", "score: no real submission to wait for")
            return
        for x in todo:
            L.run_monitored(self.ctx, f"score {x['ref']}", ["python3", "-B", "pipeline/setup/score_wait.py", x["ref"]],
                            estimate_min=self.ctx.estimate("score"))
            row = next((r for r in SUB.leaderboard() if r["ref"] == x["ref"]), None)
            x["public"] = row and row["public"]
            self.log("RESULT", f"score {x['track']} {x['ref']}: public {x['public']} (estimated {x['estimate']:.5f})")
            self.ctx.save()

    # ---------------------------------------------------------------- stage 戊 hook
    def tree_smoke(self):
        """One overlay episode (first max_steps steps) on the first scenario: two-stage randomised room option over
        the track's stage-丁 version (the base when there is none yet), per-unit rows. No training."""
        import multiprocessing as mp
        import geo5_tree as T
        track = self.a.tree_track
        room = [lid for lid, lv in self.levels.items() if lv["layer"] == "room"
                and (track == "competition" or lv["category"] == "physics")]
        if not room:
            raise Wake("crash", "tree smoke: no room-layer level in the option file")
        spec = self.specs[0]
        with mp.get_context("spawn").Pool(1) as pool:
            feat = pool.apply(T.unit_features, (spec,))
        deltas = []
        for lid in room[:2]:
            d = {"add": {}, "set": {}}
            for w in self.levels[lid]["build"]["wrappers"]:
                d["add" if w["name"] == "room_option_add" else "set"].update(w.get("args", {}))
            deltas.append(d)
        asg = T.two_stage_assign(feat, len(deltas), seed=11)
        rule = self.stage(f"{track}.verify").get("model")
        rule_model = L.read_json(ROOT / rule) if rule else self.base
        name = f"{self.prefix}g5{track[0]}tree00"
        model = T.overlay_model(rule_model, deltas, asg)
        O.write_model(model, self.model_path(name))
        R.record(self.ctx, "tree smoke", [(self.rel(self.model_path(name)), spec)], self.ctx.dir, 1, self.ms)
        rec = L.read_json(R.record_path(self.ctx.dir, name, spec, self.ms))
        bill = L.read_json(R.billing_path(self.ctx.dir, name, spec, self.ms))
        rows = T.unit_rows(rec, feat, asg, len(deltas))
        out = self.ctx.dir / "tree" / track
        L.atomic_json(out / "features.json", {spec: feat})
        with (out / "units.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        folds = T.lobo_folds(self.specs)
        info = {"track": track, "rule": rule or "gt_u1b", "spec": spec, "rooms": len(feat["rooms"]),
                "ahus": len(feat["ahu_rooms"]), "rows": len(rows), "rooms_coded": len(asg["codes"]),
                "share": asg["share"], "wrapper_calls": bill["room_wrapper_calls"], "folds": folds,
                "cost": rec["summary"]["cost"]}
        self.st.setdefault("tree_smoke", {})[track] = info
        self.log("RESULT", f"tree smoke ({track}, rule {info['rule']}) {spec}: {info['rooms']} rooms on {info['ahus']} "
                           f"AHUs, {info['rooms_coded']} rooms with a room option, overlay applied "
                           f"{info['wrapper_calls']} times, {len(rows)} unit rows; AHU shares {asg['share']}")
        for t in T.TREES.values():
            try:
                t().fit(rows, folds)
            except NotImplementedError as e:
                self.log("EVENT", f"tree {t.name}: interface present ({e})")

    # ---------------------------------------------------------------- run
    def summary(self):
        st = self.st
        out = {"updated": L.now(), "state_dir": self.rel(self.ctx.dir), "tracks": self.tracks, "max_steps": self.ms,
               "scenarios": len(self.specs), "stages": {}, "submissions": st.get("submissions", []),
               "wakes": st.get("wakes", []), "timing": self.ctx.timing}
        for n in self.stages:
            s = st["stages"].get(n)
            if s:
                out["stages"][n] = {k: v for k, v in s.items() if k not in ("runs", "fit")}
        best = {}
        for t in self.tracks:
            v = st["stages"].get(f"{t}.verify", {})
            best[t] = {"single_level": (st["stages"].get("screen", {}).get("best_single") or {}).get(t),
                       "rule_version": v.get("name"), "options": st["stages"].get(f"{t}.design", {}).get("chosen"),
                       "realised_pct": v.get("realised_pct"), "metric": v.get("metric")}
        out["best_per_track"] = best
        if "tree_smoke" in st:
            out["tree_smoke"] = {t: {k: x for k, x in v.items() if k != "folds"} for t, v in st["tree_smoke"].items()}
        return out

    def run(self):
        a = self.a
        if a.tree_smoke:
            self.tree_smoke()
            self.ctx.save()
            return 0
        for name in self.stages:
            st = self.stage(name)
            if st["status"] == "done":
                continue
            track, _, kind = name.rpartition(".")
            st.update(status="running", started=L.now())
            self.ctx.save()
            t0 = time.time()
            try:
                getattr(self, f"stage_{kind}")(st, track or None)
            except Wake as w:
                return self.wake(name, st, w)
            except Exception:  # noqa: BLE001 - every crash wakes the coordinator
                return self.wake(name, st, Wake("crash", f"stage {name}: {traceback.format_exc()[-1500:]}"))
            except KeyboardInterrupt:
                L.kill_tree()
                st["status"] = "interrupted"
                self.ctx.save()
                raise
            st.update(status="done", ended=L.now(), minutes=round((time.time() - t0) / 60.0, 2))
            self.ctx.save()
            self.log("EVENT", f"stage {name} done in {st['minutes']:.1f} min (attempt {st['attempt']})")
            if a.until == name:
                self.log("EVENT", f"--until {name}: stopping here")
                return 0
        self.st["done"] = L.now()
        self.ctx.save()
        self.log("EVENT", f"pipeline 5 done; summary {self.rel(self.ctx.summary_path)}")
        return 0

    def wake(self, name, st, w):
        L.kill_tree()
        if w.info:                      # information wake: the stage pauses, its attempt is not used up
            st.update(status="paused", paused=L.now())
            payload = self.ctx.write_wake(w, name, st["attempt"], False)
            self.st["wake_pending"] = {"stage": name, "attempt": st["attempt"], "reason": w.reason, "info": True}
            self.st["wakes"].append({"stage": name, "attempt": st["attempt"], "reason": w.reason, "time": L.now(),
                                     "info": True})
            self.ctx.save()
            self.log("WAKE", f"stage {name}: {w.reason} (information, not a failure): {w.text[:400]} -> "
                             f"{self.rel(self.ctx.wake_path)}; continue with --resume")
            print(json.dumps(payload, indent=1), flush=True)
            return L.EXIT_INFO
        final = st["attempt"] >= 2
        st.update(status="woken", woken=L.now())
        st.setdefault("wakes", []).append({"time": L.now(), "attempt": st["attempt"], "reason": w.reason,
                                           "text": w.text[:600]})
        payload = self.ctx.write_wake(w, name, st["attempt"], final)
        self.st["wake_pending"] = {"stage": name, "attempt": st["attempt"], "reason": w.reason, "final": final}
        self.st["wakes"].append({"stage": name, "attempt": st["attempt"], "reason": w.reason, "time": L.now()})
        self.ctx.save()
        self.log("WAKE", f"stage {name} attempt {st['attempt']}: {w.reason}: {w.text[:500]} -> "
                         f"{self.rel(self.ctx.wake_path)}{' (second attempt: FINAL)' if final else ''}")
        print(json.dumps(payload, indent=1), flush=True)
        return L.EXIT_FINAL if final else L.EXIT_WAKE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="data/runs/b2rl/geo5")
    p.add_argument("--options", default="data/runs/b2rl/geo5/options.json")
    p.add_argument("--screen-set", default="data/runs/b2rl/geo5/screen30.json")
    p.add_argument("--tracks", default="competition,paper")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-runs", type=int, default=64)
    p.add_argument("--batches", type=int, default=3, help="screening batches (direction summaries after each)")
    p.add_argument("--mismatch", type=float, default=0.5, help="wake when |realised - predicted| > this x |predicted|")
    p.add_argument("--loop-need", type=float, default=-0.5)
    p.add_argument("--loop-s4-max", type=float, default=0.5)
    p.add_argument("--submit", choices=("dry", "auto"), default="dry")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--attempt", type=int)
    p.add_argument("--redo", default="", help="stages to run again (and every later one), e.g. competition.design")
    p.add_argument("--combo", default="", help="coordinator fix: force the chosen combination (level ids)")
    p.add_argument("--combo-factors", default="", help="coordinator fix: the design's factors (level ids)")
    p.add_argument("--review", default="data/runs/b2rl/geo5/review.json",
                   help="the coordinator's reviewed list per track: base levels (always on) and design factors")
    # test hooks
    p.add_argument("--specs", default="")
    p.add_argument("--only", default="")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--keep-all", action="store_true")
    p.add_argument("--fake-closed-loop", choices=("pass", "fail"))
    p.add_argument("--fake-gate", choices=("pass", "fail"))
    p.add_argument("--fake-gate-s4", type=float, default=-0.23)
    p.add_argument("--fake-count", type=int)
    p.add_argument("--stop-after-episodes", type=int)
    p.add_argument("--until")
    p.add_argument("--tree-smoke", action="store_true")
    p.add_argument("--tree-track", choices=TRACKS, default="competition")
    a = p.parse_args()
    os.environ.setdefault("SB_FASTSIM", "rom")
    state = ROOT / a.state / "pipeline.json"
    if state.is_file() and not a.resume and not a.tree_smoke:
        print(f"{state.relative_to(ROOT)} exists: continue with --resume (and --attempt N after a wake)", flush=True)
        return 2
    if state.is_file():               # a resume keeps the run's own inputs unless they are given again
        stored = (L.read_json(state) or {}).get("args", {})
        for k in ("options", "screen_set", "specs", "only", "max_steps", "tracks", "keep_all", "max_runs", "batches"):
            if k in stored and getattr(a, k) == p.get_default(k):
                setattr(a, k, stored[k])
    pipe = Pipeline(a)
    st = pipe.st
    pending = st.get("wake_pending")
    if pending and pending.get("info") and not a.tree_smoke:
        if pipe.ctx.wake_path.is_file():
            pipe.ctx.wake_path.replace(pipe.ctx.dir / "diag" / f"wake-read-{time.strftime('%Y%m%d-%H%M%S')}.json")
        st.pop("wake_pending")
        pipe.stage(pending["stage"])["status"] = "pending"
        pipe.log("EVENT", f"resume after the information wake {pending['reason']} of stage {pending['stage']} "
                          f"(attempt {pending['attempt']} kept)")
    elif pending and not a.tree_smoke:
        if not a.attempt or a.attempt <= pending["attempt"]:
            print(f"a wake is pending ({pending}); after the fix: --resume --attempt {pending['attempt'] + 1}",
                  flush=True)
            return 2
        if a.attempt > 2:
            pipe.log("EVENT", f"attempt {a.attempt} of stage {pending['stage']}: beyond the plan's two chances, "
                              "explicitly asked by the coordinator")
        s = pipe.stage(pending["stage"])
        s.update(attempt=a.attempt, status="pending")
        if pipe.ctx.wake_path.is_file():
            pipe.ctx.wake_path.replace(pipe.ctx.dir / "diag" / f"wake-handled-{time.strftime('%Y%m%d-%H%M%S')}.json")
        st.pop("wake_pending")
        pipe.log("EVENT", f"resume: stage {pending['stage']} attempt {a.attempt} (woken by {pending['reason']})")
    for name in [x for x in a.redo.split(",") if x]:
        hits = [n for n in pipe.stages if n == name or n.endswith(f".{name}")]
        for h in hits:
            for n in pipe.stages[pipe.stages.index(h):]:
                if n in st["stages"] and (n.split(".")[0] == h.split(".")[0] or "." not in h or "." not in n):
                    st["stages"][n]["status"] = "pending"
        pipe.log("EVENT", f"--redo {name}: {hits} and the later stages of the same track run again")
    pipe.ctx.save()
    pipe.log("EVENT", f"pipeline 5 {'resumes' if a.resume else 'starts'}: tracks {pipe.tracks}, {len(pipe.levels)} "
                      f"levels, {len(pipe.specs)} scenarios, workers <= {a.workers}, submit {a.submit}"
                      + (f", first {pipe.ms} steps (smoke)" if pipe.ms else ""))
    return pipe.run()


if __name__ == "__main__":
    sys.exit(main())
