"""Stage 戊之三 correction (2026-11-10; spec docs/plan/tree-round3-20261109.md §2f): room options decided per AHU.
The two-stage data showed that a room option's effect is not additive over rooms (heating airflow ratio 0.6 in large
rooms with exterior wall: whole AHU on -0.96 %, half on +0.22 %), while every room learner assumed additivity.
- Data: per episode, per room option and per AHU, the coverage of the option's eligible rooms (0, 1/2, 1 as drawn).
  Pseudo-options for the learners: "full" = the eligible rooms of fully covered AHUs (the effect we can deploy),
  "partial" = the eligible rooms of partly covered AHUs (a nuisance column, never deployed).
- Decisions split only on AHU-constant features (the AHU's exterior-wall share, its number of rooms, the climate), so
  every AHU is all on or all off; run-time features exist already (rooms.room_features ahu_ext_share, ahu_size).
- Kinds: ahu-me, ahu-tree, ahu-bic1, ahu-lin, ahu-bag (geo5_r3_kinds learners on the pseudo-options); baselines: the
  room-level kinds of gt_r3a1 (tree|basic), gt_r3c1 (lin|rich), gt_r3d1 (lin|graph), and no room option.
- Honest coverage-aware referee: decisions from the training buildings; on the test scenarios' own episodes, per
  option, the response to coverage (c, c^2, weighted by the AHU's eligible rooms) of the AHUs the decision switches on
  (and, separately, of the others), scenario fixed effects; value = that response at the decided coverage (1 for an
  AHU kind, the room kinds' own fraction otherwise), cost-weighted.
Output data/runs/b2rl/geo5/r3/kinds-ahu.json.
python3 -B pipeline/launch/launch.py pipeline/setup/geo5_r3_ahu.py [--procs 12] [--smoke]"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_r3_collect as C  # noqa: E402
import geo5_r3_kinds as K  # noqa: E402
import geo5_r3_learn as RL  # noqa: E402
import geo5_record as R  # noqa: E402
import geo5_tk_room as RM  # noqa: E402
import group_stage as gs  # noqa: E402

OUT = C.OUT / "kinds-ahu.json"
NO = len(C.DELTAS)
DAGGER = False
AHU_FEATS = ["ahu_ext_share", "ahu_size"]
AHU_KINDS = ("me", "tree", "bic1", "lin", "bag")
ROOM_BASE = [("tree", "basic"), ("lin", "rich"), ("lin", "graph")]
CURRENT = "room:tree|basic"
DATA = None


class AhuData:
    """Episodes of the round with: obs (pseudo-options: j = rooms of fully covered AHUs, NO + j = rooms of partly
    covered AHUs), obs_add (the room-additive rows of geo5_r3_learn, for the room-level kinds), and per episode the
    coverage of every option in every AHU."""

    def __init__(self):
        st = json.loads((C.OUT / "pipeline.json").read_text(encoding="utf-8"))
        feats = json.loads((C.G / "tree2/unit_features.json").read_text(encoding="utf-8"))
        self.scen, self.obs, self.obs_add = {}, [], []
        for i, s in enumerate(st["specs"]):
            norm, oat = RL.rule_cost(s)
            if norm is None:
                continue
            fb = feats[s.split("/")[1]]
            rr = RM.room_rows(fb)
            nums = [int(r) for r in fb["rooms"]]
            ahu = np.asarray(fb["ahu"])
            cells = [tuple(gs.cell_of(a, w)) for a, w in zip(rr["area"], rr["wall"])]
            elig = [np.array([d["cells"] is None or list(c) in d["cells"] for c in cells]) for d in C.DELTAS]
            got, got_add = [], []
            for r, bits in enumerate(st["rows"]):
                for k in range(st["reps"]):
                    name = f"r3r{r}k{k}s{i:03d}"
                    rp = R.record_path(C.OUT, name, s)
                    if not rp.is_file():
                        continue
                    cost = json.loads(rp.read_text(encoding="utf-8"))["summary"]["cost"]
                    asg = json.loads((C.OUT / "assign" / f"{name}.json").read_text(encoding="utf-8"))
                    code = np.array([int(asg["codes"].get(str(n), 0)) for n in nums])
                    x = np.array([1.0 if v else -1.0 for v in bits])
                    rows, rows_add, cov = {}, {}, {}
                    for j in range(NO):
                        on = (code >> j & 1).astype(bool) & elig[j]
                        rows_add[j] = RL.Data._rows(rr, np.nonzero(on)[0], oat)
                        full, part, cv = [], [], {}
                        for g in np.unique(ahu[elig[j]]):
                            m = elig[j] & (ahu == g)
                            c = float(on[m].mean())
                            cv[int(g)] = (int(m.sum()), c)
                            if c >= 1.0:
                                full += list(np.nonzero(m)[0])
                            elif c > 0.0:
                                part += list(np.nonzero(m)[0])
                        rows[j] = RL.Data._rows(rr, np.array(full, int), oat)
                        rows[NO + j] = RL.Data._rows(rr, np.array(part, int), oat)
                        cov[j] = cv
                    base = {"s": s, "b": s.split("/")[1], "w": norm, "oat": oat, "y": cost / norm - 1.0, "x": x}
                    got.append({**base, "rows": rows, "cov": cov, "share": None})
                    got_add.append({**base, "rows": rows_add, "code": code, "share": np.array(
                        [float(((code >> j & 1).astype(bool) & elig[j]).sum()) / max(1, elig[j].sum())
                         for j in range(NO)])})
            if len(got) >= 8:
                self.obs += got
                self.obs_add += got_add
                self.scen[s] = {"oat": oat, "cost": norm, "rr": rr, "elig": elig + elig, "ahu": ahu, "n": len(got)}
        if DAGGER:
            self._add_dagger(feats)

    def _add_dagger(self, feats):
        """Relabelling data (geo5_r3_dagger2): the student's per-AHU decisions randomly flipped on new buildings;
        scenario keys "dg|<spec>" (training only: never a test scenario), building-wide columns 0 (the student's own
        gates, constant within a scenario), coverage 0 / 1 per AHU -> only the "full" pseudo-options."""
        D = C.OUT / "dagger2"
        if not (D / "pipeline.json").is_file():
            return
        n0 = len(self.obs)
        for ap in sorted((D / "assign").glob("dg*.json")):
            asg = json.loads(ap.read_text(encoding="utf-8"))
            s = asg["spec"]
            rp = R.record_path(D, ap.stem, s)
            rule = [R.record_path(d, "gt_r5pa2", s) for d in (C.G / "tree", C.OUT, D)]
            rule = next((p for p in rule if p.is_file()), None)
            if not rp.is_file() or rule is None or s.split("/")[1] not in feats:
                continue
            rec = json.loads(rp.read_text(encoding="utf-8"))
            norm = json.loads(rule.read_text(encoding="utf-8"))["summary"]["cost"]
            oat = float(np.mean(rec["hourly"]["outside_t"]))
            fb = feats[s.split("/")[1]]
            rr = RM.room_rows(fb)
            ahu = np.asarray(fb["ahu"])
            cells = [tuple(gs.cell_of(a, w)) for a, w in zip(rr["area"], rr["wall"])]
            elig = [np.array([d["cells"] is None or list(c) in d["cells"] for c in cells]) for d in C.DELTAS]
            rows, cov = {}, {}
            for j, d in enumerate(C.DELTAS):
                cv = {int(g): float(c) for g, c in asg["coverage"][d["name"]].items()}
                full = [k for k in np.nonzero(elig[j])[0] if cv.get(int(ahu[k]), 0.0) >= 1.0]
                rows[j] = RL.Data._rows(rr, np.array(full, int), oat)
                rows[NO + j] = RL.Data._rows(rr, np.array([], int), oat)
                cov[j] = {g: (int((elig[j] & (ahu == g)).sum()), c) for g, c in cv.items()}
            key = f"dg|{s}"
            self.obs.append({"s": key, "b": s.split("/")[1], "w": norm, "oat": oat, "y": rec["summary"]["cost"] / norm
                             - 1.0, "x": np.zeros(RL.NB), "rows": rows, "cov": cov, "share": None})
            sc = self.scen.setdefault(key, {"oat": oat, "cost": norm, "rr": rr, "elig": elig + elig, "ahu": ahu, "n": 0})
            sc["n"] += 1
        print(f"relabelling data: {len(self.obs) - n0} episodes on "
              f"{len({o['s'] for o in self.obs[n0:]})} new scenarios", flush=True)


AHU_FEATS_G = AHU_FEATS + list(RM.AHU_GEOM)     # fixed candidate three v2: + the room geometry of the whole AHU


def _ahu_mode(feats=None):
    RL.NO = 2 * NO
    RL.SPLIT_FEATS = [K.FEATS_X.index(f) for f in (feats or AHU_FEATS)] + [K.FEATS_X.index("oat")]
    key = tuple(RL.SPLIT_FEATS)
    if key not in K._SCALE:
        allX = np.vstack([RL.Data._rows(sc["rr"], np.arange(len(sc["rr"]["area"])), sc["oat"])
                          for sc in DATA.scen.values()])
        K._SCALE[key] = (allX[:, RL.SPLIT_FEATS].mean(0), allX[:, RL.SPLIT_FEATS].std(0) + 1e-9)


def _room_mode(rep):
    RL.NO = NO
    K.use_rep(rep)


class _AddData:
    """geo5_r3_kinds' view of the data for the room-level kinds (elig per real option)."""

    def __init__(self, d):
        self.scen = {s: dict(v, elig=v["elig"][:NO]) for s, v in d.scen.items()}
        self.obs = d.obs_add


GUARD = True


def guarded(tree, lo, hi):
    """Support guard (2026-11-10, second correction): the option is on only inside the climate range of the training
    scenarios [lo, hi] (no extrapolation to an unseen climate); a split on the climate wraps the tree."""
    oi = K.FEATS_X.index("oat")
    off = {"gamma": 0.0, "on": False}
    t = {"f": oi, "t": float(lo) - 1e-6, "l": dict(off), "r": {"f": oi, "t": float(hi), "l": tree, "r": dict(off)}}
    RL._number(t)
    return t


def fit_lin3(train, seed=0, feats=None):
    """候選三 fixed (user 11/10 03:2x "把候選三做完"): candidate three's teacher (linear exposure model on the rich
    room geometry, geo5_r3_kinds.fit_lin) fitted on the per-AHU pseudo-options (rooms of fully covered AHUs; partly
    covered AHUs as a nuisance), then per AHU the sum of the teacher's room effects over the option's eligible rooms
    = the AHU's full-coverage effect, distilled (regret-weighted) to a depth-2 tree on AHU-constant features -> every
    AHU all on or all off."""
    trs = set(train)
    RL.NO = 2 * NO
    try:
        K.use_rep("rich")
        th, bb = K.fit_lin([o for o in DATA.obs if o["s"] in trs])
        rows, tg, wq = {j: [] for j in range(NO)}, {j: [] for j in range(NO)}, {j: [] for j in range(NO)}
        for s in train:
            sc = DATA.scen.get(s)
            if sc is None:
                continue
            for j in range(NO):
                el = sc["elig"][j]
                for g in np.unique(sc["ahu"][el]):
                    idx = np.nonzero(el & (sc["ahu"] == g))[0]
                    X = RL.Data._rows(sc["rr"], idx, sc["oat"])
                    rows[j].append(X[0])
                    tg[j].append(float((K._phi(X) @ th[j]).sum()))
                    wq[j].append(sc["cost"])
        _ahu_mode(feats)
        trees = [K._distil(np.array(rows[j]), np.array(tg[j]), np.array(wq[j])) if rows[j] else
                 {"depth": 0, "id": 0, "gamma": 0.0, "on": False} for j in range(NO)]
    finally:
        RL.NO = NO
    return RL.RoomFit(trees, bb)


NONADD = ("ratio06_LXe",)        # options shown non-additive over rooms by the two-stage data (audit §2f)


def _lin_se(obs, lam=5.0):
    """geo5_r3_kinds.fit_lin with the ridge covariance: theta (NO x P), building-wide betas, per-option P x P cov."""
    pr = RL.Prep(obs)
    P = K._phi(np.zeros((1, len(RL.FEATS)))).shape[1]
    A = np.zeros((pr.n, RL.NB + RL.NO * P))
    A[:, :RL.NB] = pr.xb
    for j in range(RL.NO):
        if len(pr.X[j]):
            blk = np.zeros((pr.n, P))
            np.add.at(blk, pr.q[j], K._phi(pr.X[j]))
            A[:, RL.NB + j * P: RL.NB + (j + 1) * P] = blk
    Ad, yd = RL.demean(A, pr.sid, pr.w), RL.demean(pr.y, pr.sid, pr.w)
    pen = np.r_[np.zeros(RL.NB), np.ones(RL.NO * P)]
    beta, cov, _ = RL.wls(Ad, yd, pr.w, (pen, np.zeros_like(pen)), lam * pr.n / A.shape[1])
    th = beta[RL.NB:].reshape(RL.NO, P)
    covs = [cov[RL.NB + j * P: RL.NB + (j + 1) * P, RL.NB + j * P: RL.NB + (j + 1) * P] for j in range(RL.NO)]
    return th, beta[:RL.NB], covs


def fit_lin3e(train, seed=0):
    trs = set(train)
    names = [d["name"] for d in C.DELTAS]
    add = _AddData(DATA)
    trees = [None] * NO
    # room-by-room options: candidate three's teacher on the additive data, gated by the evidence rule
    _room_mode("rich")
    th, bb, covs = _lin_se([o for o in add.obs if o["s"] in trs])
    Xq, wq = K._query(add, list(s for s in train if s in add.scen))
    for j in range(NO):
        if names[j] in NONADD:
            continue
        ph = K._phi(Xq[j]) if len(Xq[j]) else np.zeros((0, th.shape[1]))
        pred = ph @ th[j]
        se = np.sqrt(np.maximum(np.einsum("ip,pq,iq->i", ph, covs[j], ph), 0.0))
        gated = np.where(pred + se < 0, pred, np.abs(pred) + 1e-12)
        trees[j] = K._distil(Xq[j], gated, wq[j])
    # the non-additive options: per AHU, teacher on fully covered AHUs, AHU sum + SE, distilled on AHU features
    RL.NO = 2 * NO
    try:
        K.use_rep("rich")
        th2, _, covs2 = _lin_se([o for o in DATA.obs if o["s"] in trs])
        for j in range(NO):
            if names[j] not in NONADD:
                continue
            rows, tg, wv = [], [], []
            for s in train:
                sc = DATA.scen.get(s)
                if sc is None:
                    continue
                el = sc["elig"][j]
                for g in np.unique(sc["ahu"][el]):
                    idx = np.nonzero(el & (sc["ahu"] == g))[0]
                    X = RL.Data._rows(sc["rr"], idx, sc["oat"])
                    ph = K._phi(X).sum(0)
                    pred = float(ph @ th2[j])
                    se = float(np.sqrt(max(ph @ covs2[j] @ ph, 0.0)))
                    rows.append(X[0])
                    tg.append(pred if pred + se < 0 else abs(pred) + 1e-12)
                    wv.append(sc["cost"])
            _ahu_mode()
            trees[j] = K._distil(np.array(rows), np.array(tg), np.array(wv)) if rows else \
                {"depth": 0, "id": 0, "gamma": 0.0, "on": False}
    finally:
        RL.NO = NO
    return RL.RoomFit(trees, bb)


def fit(kind, train, seed=0):
    """RoomFit with NO trees (FEATS_X indices). kind: "ahu-<k>" or "room:<k>|<rep>"."""
    trs = set(train)
    if kind == "ahu-lin3":
        f = fit_lin3(train, seed)
    elif kind == "ahu-lin3e":
        f = fit_lin3e(train, seed)
    elif kind == "ahu-c3fix":                     # candidate three, complete fix (review 2026-11-10)
        import geo5_r3_c3fix as C3
        C3.A.DATA = DATA                          # (when this file runs as __main__, C3 sees another module)
        f = C3.fit_tuned(list(train), seed)
    elif kind == "ahu-lin3g":
        f = fit_lin3(train, seed, AHU_FEATS_G)
    elif kind.startswith("ahu-"):
        _ahu_mode()
        try:
            f = K.fit_room_kind(kind[4:], [o for o in DATA.obs if o["s"] in trs], DATA, list(train), seed=seed)
        finally:
            RL.NO = NO
        f = RL.RoomFit(f.trees[:NO], f.beta_b)
    else:
        k, rep = kind[5:].split("|")
        _room_mode(rep)
        add = _AddData(DATA)
        f = K.fit_room_kind(k, [o for o in add.obs if o["s"] in trs], add, list(train), seed=seed)
    if GUARD:
        oats = [DATA.scen[s]["oat"] for s in train if s in DATA.scen]
        f = RL.RoomFit([guarded(t, min(oats), max(oats)) for t in f.trees], f.beta_b)
    return f


def decided(fit_, s):
    """Per option, per AHU: (eligible rooms, decided coverage) in scenario s."""
    if hasattr(fit_, "decided"):                 # a teacher deciding with AHU sums (geo5_r3_fixcheck)
        return fit_.decided(s)
    sc = DATA.scen[s]
    out = {}
    for j in range(NO):
        el = sc["elig"][j]
        X = RL.Data._rows(sc["rr"], np.arange(len(el)), sc["oat"])
        on = fit_.on_rows(j, X) & el
        out[j] = {int(g): (int((el & (sc["ahu"] == g)).sum()), float(on[el & (sc["ahu"] == g)].mean()))
                  for g in np.unique(sc["ahu"][el])}
    return out


def boot_buildings(test, rng):
    """A building-clustered bootstrap draw of a test scenario list (buildings with replacement)."""
    by = {}
    for s in test:
        by.setdefault(s.split("/")[-2], []).append(s)
    blds = sorted(by)
    out = []
    for i in rng.integers(0, len(blds), len(blds)):
        out += by[blds[i]]
    return out


def referee(fit_, test):
    from collections import Counter
    mult = Counter(test)                          # a resampled list may repeat a scenario
    tes = set(mult)
    obs = [o for o in DATA.obs if o["s"] in tes]
    dec = {s: decided(fit_, s) for s in tes}
    A = np.zeros((len(obs), RL.NB + 4 * NO))
    for q, o in enumerate(obs):
        A[q, :RL.NB] = o["x"]
        for j in range(NO):
            for g, (n, c) in o["cov"][j].items():
                grp = 0 if dec[o["s"]][j].get(g, (0, 0.0))[1] > 0 else 2
                A[q, RL.NB + 4 * j + grp] += n * c
                A[q, RL.NB + 4 * j + grp + 1] += n * c * c
    y = np.array([o["y"] for o in obs])
    w = np.array([o["w"] * mult[o["s"]] for o in obs])
    sid = np.unique([o["s"] for o in obs], return_inverse=True)[1]
    Ad = RL.demean(A, sid, w)
    keep = np.abs(Ad).sum(0) > 1e-12
    beta = np.zeros(A.shape[1])
    b, _, _ = RL.wls(Ad[:, keep], RL.demean(y, sid, w), w, (np.ones(keep.sum()), np.zeros(keep.sum())), 1e-6)
    beta[keep] = b
    num = den = 0.0
    for s in tes:
        tot = 0.0
        for j in range(NO):
            b1, b2 = beta[RL.NB + 4 * j], beta[RL.NB + 4 * j + 1]
            for g, (n, d) in dec[s][j].items():
                if d > 0:
                    tot += n * (b1 * d + b2 * d * d)
        num += DATA.scen[s]["cost"] * tot * mult[s]
        den += DATA.scen[s]["cost"] * mult[s]
    return num / den


def _cov_stats(fit_, test):
    part = full = el = 0.0
    for s in test:
        w = DATA.scen[s]["cost"]
        for j, by in decided(fit_, s).items():
            for g, (n, d) in by.items():
                el += w * n
                part += w * n * (0 < d < 1)
                full += w * n * (d >= 1)
    return part / max(el, 1e-12), full / max(el, 1e-12)


def _job(args):
    kind, vtype, fid, test, train = args
    t0 = time.time()
    f = fit(kind, train, seed=sum(map(ord, fid)) % 1000)
    part, full = _cov_stats(f, test)
    return {"kind": kind, "v": vtype, "fold": fid, "w": float(sum(DATA.scen[s]["cost"] for s in test)),
            "ref": referee(f, test), "part": part, "full": full, "sec": time.time() - t0}


def main():
    global DATA, DAGGER, OUT
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dagger", action="store_true", help="add the relabelling data (training only)")
    p.add_argument("--kinds", default="", help="comma list, e.g. ahu-bic1,ahu-tree (default: all)")
    a = p.parse_args()
    if a.dagger:
        DAGGER = True
        OUT = C.OUT / "kinds-ahu-dagger.json"
    t0 = time.time()
    K.setup_features()
    DATA = AhuData()
    K.DATA = DATA
    print(f"data: {len(DATA.scen)} scenarios, {len(DATA.obs)} episodes ({time.time() - t0:.0f} s)", flush=True)
    K.DATA = _AddData(DATA)
    K.prepare_scales()
    K.DATA = DATA
    kinds = [f"ahu-{k}" for k in AHU_KINDS] + [f"room:{k}|{r}" for k, r in ROOM_BASE]
    if a.kinds:
        kinds = a.kinds.split(",")
    folds = K._folds(DATA)
    jobs = [(k, v, f, te, tr) for k in kinds for (v, f, te, tr) in folds]
    if a.smoke:
        K.BAGS = 2
        jobs = [j for j in jobs if j[1] == "lobo" and j[2] == "0"]
    else:
        K.BAGS = 10
    jobs.sort(key=lambda j: "bag" not in j[0])
    res = []
    with mp.get_context("fork").Pool(a.procs) as pool:
        for r in pool.imap_unordered(_job, jobs):
            res.append(r)
            print(f"  {r['kind']:16s} {r['v']} {r['fold']:10s} referee {100 * r['ref']:+.2f}%  rooms in partly / "
                  f"fully covered AHUs {100 * r['part']:.0f}% / {100 * r['full']:.0f}%  ({r['sec']:.0f} s)", flush=True)
    out = {}
    for k in kinds:
        out[k] = {}
        for v in ("lobo", "loco"):
            rr = [r for r in res if r["kind"] == k and r["v"] == v]
            if rr:
                W_ = sum(r["w"] for r in rr)
                out[k][v] = sum(r["ref"] * r["w"] for r in rr) / W_
                out[k][v + "_folds"] = {r["fold"]: r["ref"] for r in rr}
                out[k][v + "_partly"] = sum(r["part"] * r["w"] for r in rr) / W_
    for k, v in out.items():
        print(f"{k:16s} " + "  ".join(f"{x} {100 * v[x]:+.2f}% (partly covered {100 * v[x + '_partly']:.0f}%)"
                                      for x in ("lobo", "loco") if x in v), flush=True)
    if not a.smoke and a.kinds:                   # a subset: merged into the existing file, its choice kept
        old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {"kinds": {}}
        old["kinds"].update(out)
        old.setdefault("added", {}).update({k: time.strftime("%Y-%m-%d %H:%M:%S") for k in out})
        OUT.write_text(json.dumps(old, indent=1, default=float), encoding="utf-8")
        print(f"merged {list(out)} into {OUT.name}")
    elif not a.smoke:
        ahu = {k: v for k, v in out.items() if k.startswith("ahu-") and "lobo" in v and "loco" in v}
        best = min(ahu, key=lambda k: (ahu[k]["lobo"] + ahu[k]["loco"]) / 2)
        cur = out.get(CURRENT, {})
        better = "lobo" in cur and ahu[best]["lobo"] < cur["lobo"] and ahu[best]["loco"] < cur["loco"]
        out_all = {"kinds": out, "best": best, "beats_current_both": better, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        OUT.write_text(json.dumps(out_all, indent=1, default=float), encoding="utf-8")
        print(f"best AHU kind {best}; better than {CURRENT} in both validations: {better}")
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
