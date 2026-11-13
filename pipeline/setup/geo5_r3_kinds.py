"""Stage 戊之三, tree kinds x geometry representations on the round's own data (2026-11-09 evening; spec
docs/plan/tree-round3-20261109.md §2d; notes docs/plan/tree-kinds-20261107.md and the dashboard paper page §10 戊):
no new simulation.
- Building-wide: all 15 kinds of geo5_tk_learn (戊's ten + linleaf, reem, rf, et, gb), leave-buildings-out and
  leave-one-climate-out on every real-building scenario of the round (train/..., the new cold ones included).
- Rooms (two-stage randomised exposures, no per-room labels): me (one switch per option), tree (the joint room tree
  of gt_r3a1), bic1 (looser split rule), d1 (depth 1), honest, bag (building-bootstrap room trees as teacher ->
  regret-weighted distillation to depth <= 2), lin (linear-exposure teacher: room effect = geometry x climate terms
  -> distillation).
- Geometry representations of a room (each + climate): basic (area, exterior wall, perimeter depth, AHU exterior
  share = gt_r3a1's), rich (+ wall per area, exterior sides, AHU size, shared interior wall, wall per orientation),
  graph (basic + the room's neighbours through shared walls: count, exterior share, wall per area), rot (basic
  geometry randomly rotated: the representation diagnostic of the notes).
- Room score = honest referee value: decisions from the training buildings; effects estimated on the test
  scenarios' own episodes (per option a column for decided-on exposed rooms and one for decided-off, scenario fixed
  effects), value = sum over options of effect x decided-on eligible rooms, cost-weighted.
Output data/runs/b2rl/geo5/r3/kinds.json.  python3 -B pipeline/launch/launch.py pipeline/setup/geo5_r3_kinds.py
[--procs 16] [--only rooms|whole] [--smoke]"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_r3_collect as C  # noqa: E402
import geo5_r3_learn as RL  # noqa: E402
import geo5_tk_room as RM  # noqa: E402

OUT = C.OUT / "kinds.json"
ROOM_KINDS = ("me", "tree", "bic1", "d1", "honest", "bag", "lin")
BASIC = ["area", "wall", "depth", "ahu_ext_share"]
REPS = {"basic": BASIC,
        "rich": ["area", "wall", "depth", "wall_per_area", "sides", "ahu_size", "ahu_ext_share", "shared_wall",
                 "wall_n", "wall_s", "wall_e", "wall_w"],
        "graph": BASIC + ["nb_n", "nb_ext", "nb_wall_pa"],
        "rot": ["rot1", "rot2", "rot3", "rot4"]}
COMBOS = [(k, "basic") for k in ROOM_KINDS] + [(k, r) for r in ("rich", "graph", "rot")
                                               for k in ("tree", "bic1", "bag", "lin")]
CURRENT = ("tree", "basic")
BAGS = 20
FEATS_X = tuple(RM.FEATS[:-1]) + tuple(RM.EXTRA) + ("rot1", "rot2", "rot3", "rot4", "oat")
DATA = None
_SCALE = {}
_ROT = None
_room_rows = RM.room_rows


def _station(s):
    return s.split("/")[2].rstrip("_0123456789")


# ---------------------------------------------------------------- extra room representations
def _graph(uf_b, rr):
    nums = [int(r) for r in uf_b["rooms"]]
    pos = {r: k for k, r in enumerate(nums)}
    n = len(nums)
    ext = (rr["wall"] > 0).astype(float)
    wpa = rr["wall_per_area"]
    cnt, le, e_, w_ = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for a, b, length in uf_b["shared"]:
        if int(a) not in pos or int(b) not in pos:
            continue
        i, j = pos[int(a)], pos[int(b)]
        for p, q in ((i, j), (j, i)):
            cnt[p] += 1
            le[p] += length
            e_[p] += length * ext[q]
            w_[p] += length * wpa[q]
    return {"nb_n": cnt, "nb_ext": e_ / np.maximum(le, 1e-9), "nb_wall_pa": w_ / np.maximum(le, 1e-9)}


def room_rows_x(uf_b):
    rr = _room_rows(uf_b)
    rr.update(_graph(uf_b, rr))
    if _ROT is not None:
        mu, sd, Q = _ROT
        Z = (np.column_stack([rr[f] for f in BASIC]) - mu) / sd
        R = Z @ Q
        for k in range(4):
            rr[f"rot{k + 1}"] = R[:, k]
    else:
        for k in range(4):
            rr[f"rot{k + 1}"] = np.zeros(len(rr["area"]))
    return rr


def setup_features():
    """Extended feature rows for every room (graph and rotated columns), installed before the data are read."""
    global _ROT
    feats = json.loads((C.G / "tree2/unit_features.json").read_text(encoding="utf-8"))
    st = json.loads((C.OUT / "pipeline.json").read_text(encoding="utf-8"))
    blds = {s.split("/")[1] for s in st["specs"]}
    Z = np.vstack([np.column_stack([_room_rows(feats[b])[f] for f in BASIC]) for b in blds if b in feats])
    Q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(4, 4)))
    _ROT = (Z.mean(0), Z.std(0) + 1e-9, Q)
    RM.room_rows = room_rows_x
    RL.RM.room_rows = room_rows_x
    RL.FEATS = FEATS_X


def use_rep(rep):
    RL.SPLIT_FEATS = [FEATS_X.index(f) for f in REPS[rep]] + [FEATS_X.index("oat")]


# ---------------------------------------------------------------- room kinds
def _distil(Xq, gam, wq, min_frac=0.04):
    """Regret-weighted distillation (VIPER-style): classify 'teacher effect < 0' with weight |effect| x scenario
    cost, depth <= 2, splits on RL.SPLIT_FEATS only; returns a tree in the room-tree format."""
    from sklearn.tree import DecisionTreeClassifier
    if len(Xq) == 0:
        return {"depth": 0, "id": 0, "gamma": 0.0, "on": False}
    feats = RL.SPLIT_FEATS
    y = (gam < 0).astype(int)
    sw = np.abs(gam) * wq
    if y.min() == y.max() or sw.sum() <= 0:
        g = float(np.average(gam, weights=wq))
        return {"depth": 0, "id": 0, "gamma": g, "on": bool(g < 0)}
    clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=max(5, int(min_frac * len(Xq))), random_state=0)
    clf.fit(Xq[:, feats], y, sample_weight=sw)
    tr = clf.tree_
    leaf_of = clf.apply(Xq[:, feats])

    def conv(i, d):
        if tr.children_left[i] == -1:
            m = leaf_of == i
            g = float(np.average(gam[m], weights=wq[m])) if m.any() else 0.0
            return {"depth": d, "gamma": g, "on": bool(g < 0)}
        return {"depth": d, "f": int(feats[tr.feature[i]]), "t": float(tr.threshold[i]),
                "l": conv(tr.children_left[i], d + 1), "r": conv(tr.children_right[i], d + 1)}
    t = conv(0, 0)
    RL._number(t)
    return t


def _query(data, scen):
    """Every eligible room of the training scenarios (the distillation query points), with the scenario weight."""
    Xs, ws = {j: [] for j in range(RL.NO)}, {j: [] for j in range(RL.NO)}
    for s in scen:
        sc = data.scen[s]
        for j in range(RL.NO):
            X = RL.Data._rows(sc["rr"], np.nonzero(sc["elig"][j])[0], sc["oat"])
            Xs[j].append(X)
            ws[j].append(np.full(len(X), sc["cost"]))
    return ({j: np.vstack(Xs[j]) if Xs[j] else np.zeros((0, len(RL.FEATS))) for j in Xs},
            {j: np.concatenate(ws[j]) if ws[j] else np.zeros(0) for j in ws})


def _phi(X):
    """Linear-exposure terms of a room: 1, the standardised representation, climate^2 and climate x geometry."""
    mu, sd = _SCALE[tuple(RL.SPLIT_FEATS)]
    z = (X[:, RL.SPLIT_FEATS] - mu) / sd
    g, o = z[:, :-1], z[:, -1:]
    return np.column_stack([np.ones(len(X)), z, o * o, g * o])


def fit_lin(obs, lam=5.0):
    pr = RL.Prep(obs)
    P = _phi(np.zeros((1, len(RL.FEATS)))).shape[1]
    A = np.zeros((pr.n, RL.NB + RL.NO * P))
    A[:, :RL.NB] = pr.xb
    for j in range(RL.NO):
        if len(pr.X[j]):
            blk = np.zeros((pr.n, P))
            np.add.at(blk, pr.q[j], _phi(pr.X[j]))
            A[:, RL.NB + j * P: RL.NB + (j + 1) * P] = blk
    Ad, yd = RL.demean(A, pr.sid, pr.w), RL.demean(pr.y, pr.sid, pr.w)
    pen = np.r_[np.zeros(RL.NB), np.ones(RL.NO * P)]
    beta, _, _ = RL.wls(Ad, yd, pr.w, (pen, np.zeros_like(pen)), lam * pr.n / A.shape[1])
    return beta[RL.NB:].reshape(RL.NO, P), beta[:RL.NB]


def fit_room_kind(kind, obs, data, scen, seed=0):
    """A RoomFit for the kind, learnt from obs (training buildings only); scen = the training scenarios (query)."""
    saved = RL.BIC_K
    try:
        if kind == "me":
            return RL.fit_rooms(obs, seed=seed, depth=0)
        if kind == "tree":
            return RL.fit_rooms(obs, seed=seed)
        if kind == "bic1":
            RL.BIC_K = 1.0
            return RL.fit_rooms(obs, seed=seed)
        if kind == "d1":
            return RL.fit_rooms(obs, seed=seed, depth=1)
        if kind == "honest":
            return RL.fit_rooms(obs, seed=seed, honest=True)
        Xq, wq = _query(data, scen)
        if kind == "bag":
            RL.BIC_K = 1.0
            rng = np.random.default_rng(seed)
            blds = sorted({o["b"] for o in obs})
            by_b = {}
            for o in obs:
                by_b.setdefault(o["b"], []).append(o)
            gam = {j: np.zeros(len(Xq[j])) for j in range(RL.NO)}
            bb = np.zeros(RL.NB)
            for k in range(BAGS):
                pick = rng.choice(len(blds), len(blds), replace=True)
                boot = []
                for n_, i in enumerate(pick):     # each draw its own scenario keys: fixed effects stay per draw
                    boot += [dict(o, s=f"{o['s']}#{n_}") for o in by_b[blds[i]]]
                f = RL.fit_rooms(boot, seed=seed + k)
                bb += f.beta_b / BAGS
                for j in range(RL.NO):
                    gam[j] += f.gamma_rows(j, Xq[j]) / BAGS
            return RL.RoomFit([_distil(Xq[j], gam[j], wq[j]) for j in range(RL.NO)], bb)
        if kind == "lin":
            th, bb = fit_lin(obs)
            return RL.RoomFit([_distil(Xq[j], _phi(Xq[j]) @ th[j], wq[j]) for j in range(RL.NO)], bb)
        raise ValueError(kind)
    finally:
        RL.BIC_K = saved


def referee(fit, obs_test, data, scen_test):
    """Honest value on test scenarios: per option, the effect of an exposed room the fit decides on (and one it decides
    off) estimated from the test episodes with scenario fixed effects; value = effect x decided-on eligible rooms."""
    n = len(obs_test)
    A = np.zeros((n, RL.NB + 2 * RL.NO))
    for q, o in enumerate(obs_test):
        A[q, :RL.NB] = o["x"]
        for j in range(RL.NO):
            X = o["rows"].get(j)
            if X is None or not len(X):
                continue
            on = fit.on_rows(j, X)
            A[q, RL.NB + 2 * j] = on.sum()
            A[q, RL.NB + 2 * j + 1] = (~on).sum()
    y = np.array([o["y"] for o in obs_test])
    w = np.array([o["w"] for o in obs_test])
    sid = np.unique([o["s"] for o in obs_test], return_inverse=True)[1]
    keep = np.abs(A).sum(0) > 0
    beta = np.zeros(A.shape[1])
    b, _, _ = RL.wls(RL.demean(A[:, keep], sid, w), RL.demean(y, sid, w), w, (np.ones(keep.sum()),
                                                                              np.zeros(keep.sum())), 1e-6)
    beta[keep] = b
    num = den = 0.0
    for s in scen_test:
        sc = data.scen[s]
        tot = 0.0
        for j in range(RL.NO):
            X = RL.Data._rows(sc["rr"], np.nonzero(sc["elig"][j])[0], sc["oat"])
            tot += beta[RL.NB + 2 * j] * float(fit.on_rows(j, X).sum())
        num += sc["cost"] * tot
        den += sc["cost"]
    return num / den


def _folds(data, seed=0, k=5):
    real = sorted(s for s in data.scen if s.startswith("train/"))
    blds = sorted({s.split("/")[1] for s in real})
    rng = np.random.default_rng(seed)
    fold = {b: i % k for i, b in enumerate(rng.permutation(blds))}
    out = []
    for f in range(k):
        test = [s for s in real if fold[s.split("/")[1]] == f]
        tb = {s.split("/")[1] for s in test}
        out.append(("lobo", str(f), test, [s for s in data.scen if s.split("/")[1] not in tb]))
    for stn in sorted({_station(s) for s in real}):
        test = [s for s in real if _station(s) == stn]
        out.append(("loco", stn, test, [s for s in data.scen if _station(s) != stn]))
    return out


def _job(args):
    kind, rep, vtype, fid, test, train = args
    t0 = time.time()
    use_rep(rep)
    data = DATA
    trs, tes = set(train), set(test)
    obs_tr = [o for o in data.obs if o["s"] in trs]
    obs_te = [o for o in data.obs if o["s"] in tes]
    fit = fit_room_kind(kind, obs_tr, data, train, seed=sum(map(ord, fid)) % 1000)
    wt = float(sum(data.scen[s]["cost"] for s in test))
    on = {C.DELTAS[j]["name"]: int(sum(fit.on_rows(j, RL.Data._rows(data.scen[s]["rr"], np.nonzero(
        data.scen[s]["elig"][j])[0], data.scen[s]["oat"])).sum() for s in test)) for j in range(RL.NO)}
    return {"kind": kind, "rep": rep, "v": vtype, "fold": fid, "w": wt, "ref": referee(fit, obs_te, data, test),
            "model": RL.room_value(fit, data, test), "on": on, "sec": time.time() - t0}


def prepare_scales():
    allX = np.vstack([RL.Data._rows(sc["rr"], np.arange(len(sc["rr"]["area"])), sc["oat"])
                      for sc in DATA.scen.values()])
    for rep in REPS:
        use_rep(rep)
        _SCALE[tuple(RL.SPLIT_FEATS)] = (allX[:, RL.SPLIT_FEATS].mean(0), allX[:, RL.SPLIT_FEATS].std(0) + 1e-9)


def rooms(procs, smoke=False):
    folds = _folds(DATA)
    prepare_scales()
    jobs = [(k, r, v, f, te, tr) for k, r in COMBOS for (v, f, te, tr) in folds]
    if smoke:
        jobs = [j for j in jobs if j[2] == "lobo" and j[3] == "0"]
    jobs.sort(key=lambda j: (j[0] != "bag", j[1] == "basic"))          # the long ones first
    res = []
    with mp.get_context("fork").Pool(procs) as pool:
        for r in pool.imap_unordered(_job, jobs):
            res.append(r)
            print(f"  {r['kind']:7s} {r['rep']:6s} {r['v']} {r['fold']:10s} referee {100 * r['ref']:+.2f}%  model "
                  f"{100 * r['model']:+.2f}%  ({r['sec']:.0f} s)", flush=True)
    out = {}
    for k, rep in COMBOS:
        key = f"{k}|{rep}"
        out[key] = {}
        for v in ("lobo", "loco"):
            rr = [r for r in res if r["kind"] == k and r["rep"] == rep and r["v"] == v]
            if rr:
                W_ = sum(r["w"] for r in rr)
                out[key][v] = sum(r["ref"] * r["w"] for r in rr) / W_
                out[key][v + "_model"] = sum(r["model"] * r["w"] for r in rr) / W_
                out[key][v + "_folds"] = {r["fold"]: r["ref"] for r in rr}
                out[key][v + "_on"] = {n: sum(r["on"][n] for r in rr) for n in rr[0]["on"]}
    return out


def whole():
    import geo5_tk_learn as TK
    import geo5_wu as W
    import geo5_wu_learn as WL
    per = RL.whole_effects(DATA)
    fac = [W.whole_factor(w) for w, _ in C.WHOLE]
    U = WL.Units(per, fac, list(range(len(fac))), weighted=True)
    real = [s for s in per if s.startswith("train/")]
    out = {}
    for k in TK.KINDS_ALL:
        t0 = time.time()
        try:
            r = TK.compare(U, real, WL.UPGRADE, kinds=(k,))[k]
            out[k] = {"lobo": r["lobo"], "loco": r["loco"], "loco_by_station": r["loco_by_station"]}
            print(f"  whole {k:8s} lobo {100 * r['lobo']:+.2f}%  loco {100 * r['loco']:+.2f}%  "
                  f"({time.time() - t0:.0f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            out[k] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  whole {k:8s} ERROR {type(e).__name__}: {e}", flush=True)
    return out


def choose(res, current):
    """Best mean of the two validations; replaces the current kind only when better in both."""
    ok = {k: v for k, v in res.items() if "lobo" in v and "loco" in v}
    best = min(ok, key=lambda k: (ok[k]["lobo"] + ok[k]["loco"]) / 2)
    cur = ok.get(current)
    if cur is None or best == current:
        return best, best
    better = ok[best]["lobo"] < cur["lobo"] and ok[best]["loco"] < cur["loco"]
    return best, (best if better else current)


def main():
    global DATA, BAGS
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--only", choices=("rooms", "whole"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        BAGS = 2
    t0 = time.time()
    setup_features()
    DATA = RL.Data()
    print(f"data: {len(DATA.scen)} scenarios, {len(DATA.obs)} episodes ({time.time() - t0:.0f} s)", flush=True)
    out_path = OUT.with_name(f"kinds-{a.only}.json") if a.only else OUT
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() and not a.smoke else {}
    if a.only != "rooms":
        out["whole"] = whole()
        out["whole_best"], out["whole_choice"] = choose(out["whole"], "figs")
        print(f"whole: best {out['whole_best']}, chosen {out['whole_choice']}", flush=True)
    if a.only != "whole":
        out["rooms"] = rooms(a.procs, a.smoke)
        for k, v in out["rooms"].items():
            print(f"room {k:14s} " + "  ".join(f"{x} {100 * v[x]:+.2f}% (model {100 * v[x + '_model']:+.2f}%)"
                                               for x in ("lobo", "loco") if x in v), flush=True)
        if not a.smoke:
            out["room_best"], out["room_choice"] = choose(out["rooms"], "|".join(CURRENT))
            print(f"rooms: best {out['room_best']}, chosen {out['room_choice']}", flush=True)
    if not a.smoke:
        out["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out_path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
