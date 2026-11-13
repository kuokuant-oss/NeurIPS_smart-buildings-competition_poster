"""Candidate three, complete fix (2026-11-10; docs/plan/review-candidate3-20261110.md §四 2-6). Candidate three's
method (linear exposure teacher on the rich room geometry -> regret-weighted shallow tree) with every rule it broke:
- additivity is tested, not assumed: per option the teacher has the room-sum terms (all exposures, not only fully
  covered AHUs) + a partial-coverage term (on-rooms of partly covered AHUs); an option whose partial term is
  significant (|t| > 2, building-clustered SE) is decided per AHU (all or none), the others room by room;
- conditional effects: each option's room effect interacts with the building-wide factors; it is evaluated at the
  candidate's own building-wide settings for the scenario's climate;
- evidence rule (notes: "效果比 0 低超過 1 個標準誤才開"): on only where prediction + 1 SE < 0, SE clustered by building;
- shrinkage toward the overall mean: the ridge penalises the geometry / climate slopes and the interactions, never an
  option's own mean effect; the ridge strength and the student's minimum leaf are chosen by an inner
  leave-buildings-out (3 folds) on the coverage-aware referee;
- climate support guard (no extrapolation outside the training climates).
Kind name for geo5_r3_ahu / the chain: "ahu-c3fix".  Smoke: python3 -B pipeline/launch/launch.py
pipeline/setup/geo5_r3_c3fix.py --smoke (planted non-additive, conditional and null effects)."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_r3_ahu as A  # noqa: E402
import geo5_r3_kinds as K  # noqa: E402
import geo5_r3_learn as RL  # noqa: E402

LAMBDAS = (0.0, 0.001, 0.01, 0.1, 1.0)    # x the median diagonal of the weighted normal matrix
MIN_LEAF = (0.02, 0.04, 0.08)
WHOLE_DEC = None                          # {scenario: x (NB,) of the candidate's building-wide settings}; set by caller


def _pairs():
    """r3 episodes as (additive obs, coverage obs) pairs (same order in geo5_r3_ahu.AhuData)."""
    n = len(A.DATA.obs_add)
    return list(zip(A.DATA.obs_add, A.DATA.obs[:n]))


def _phi_dim():
    return K._phi(np.zeros((1, len(RL.FEATS)))).shape[1]


def design(pairs, rr_cache=None):
    """Columns: x_b (NB) | per option j: room-sum phi (P) | partial on-rooms (1) | on-rooms x x_b (NB)."""
    P = _phi_dim()
    per = P + 1 + RL.NB
    A_ = np.zeros((len(pairs), RL.NB + A.NO * per))
    for q, (oa, oc) in enumerate(pairs):
        A_[q, :RL.NB] = oa["x"]
        sc = A.DATA.scen[oa["s"]]
        for j in range(A.NO):
            base = RL.NB + j * per
            X = oa["rows"].get(j)
            if X is not None and len(X):
                A_[q, base:base + P] = K._phi(X).sum(0)
                A_[q, base + P + 1: base + P + 1 + RL.NB] = len(X) * oa["x"]
            part = 0.0
            for g, (n, c) in oc["cov"][j].items():
                if 0.0 < c < 1.0:
                    part += round(n * c)
            A_[q, base + P] = part
    return A_, P, per


_CACHE = {}


def whole_dec(train):
    """The candidate's building-wide settings per scenario: random-forest teacher (as gt_r3c1) on the training
    scenarios' per-scenario effects, decision at the scenario's climate."""
    import geo5_tk_learn as TK
    import geo5_wu as W
    import geo5_wu_learn as WL
    add = A._AddData(A.DATA)
    trs = set(train)
    sub = type("D", (), {})()
    sub.obs = [o for o in add.obs if o["s"] in trs]
    sub.scen = {s: v for s, v in add.scen.items() if s in trs}
    per = RL.whole_effects(sub)
    fac = [W.whole_factor(w) for w, _ in A.C.WHOLE]
    U = WL.Units(per, fac, list(range(len(fac))), weighted=True)
    fit = TK.fit_kind("rf", U, U.scen, WL.UPGRADE, None, seed=0)
    dec = WL.decisions(fit, fac, list(range(len(fac))))
    out = {}
    for s, sc in A.DATA.scen.items():
        g = int(np.argmin(np.abs(WL.GRID - sc["oat"])))
        out[s] = np.array([1.0 if dec[b][g] else -1.0 for b in range(len(fac))])
    return out


def fit_teacher(pairs, lam_rel):
    key = tuple(sorted({oa["s"] for oa, _ in pairs}))
    if key not in _CACHE:
        if len(_CACHE) > 16:
            _CACHE.clear()
        _CACHE[key] = design(pairs)
    A_, P, per = _CACHE[key]
    y = np.array([oa["y"] for oa, _ in pairs])
    w = np.array([oa["w"] for oa, _ in pairs])
    sid = np.unique([oa["s"] for oa, _ in pairs], return_inverse=True)[1]
    bid = np.unique([oa["b"] for oa, _ in pairs], return_inverse=True)[1]
    Ad, yd = RL.demean(A_, sid, w), RL.demean(y, sid, w)
    ww = w / w.mean()
    M = Ad.T @ (Ad * ww[:, None])
    pen = np.zeros(A_.shape[1])
    for j in range(A.NO):
        base = RL.NB + j * per
        pen[base + 1: base + P] = 1.0                      # slopes of phi (not its constant = the option's mean)
        # interactions with the building-wide factors: NOT penalised (smoke 11-10: their columns ~ x * room count are
        # collinear across options; shrinking the true one leaked its effect into the null options)
    lam = lam_rel * float(np.median(np.diag(M)[pen > 0])) if (pen > 0).any() else 0.0
    Mr = M + lam * np.diag(pen) + 1e-9 * np.eye(len(pen))
    beta = np.linalg.solve(Mr, Ad.T @ (ww * yd))
    res = yd - Ad @ beta
    G = np.zeros((bid.max() + 1, A_.shape[1]))
    np.add.at(G, bid, Ad * (ww * res)[:, None])          # building-clustered sandwich
    Mi = np.linalg.inv(Mr)
    cov = Mi @ (G.T @ G) @ Mi
    return {"beta": beta, "cov": cov, "P": P, "per": per}


def _x_of(s):
    if WHOLE_DEC is not None and s in WHOLE_DEC:
        return np.asarray(WHOLE_DEC[s], float)
    return np.zeros(RL.NB)


def room_pred(t, j, X, x):
    """Per room: (prediction, SE) of switching option j on, at building-wide settings x."""
    P, per = t["P"], t["per"]
    base = RL.NB + j * per
    idx = list(range(base, base + P)) + list(range(base + P + 1, base + P + 1 + RL.NB))
    F = np.hstack([K._phi(X), np.tile(x, (len(X), 1))])
    b = t["beta"][idx]
    C = t["cov"][np.ix_(idx, idx)]
    return F @ b, np.sqrt(np.maximum(np.einsum("ip,pq,iq->i", F, C, F), 0.0)), F, b, C


def option_evidence(t, alpha=0.05):
    """Per option: are its effect terms jointly different from 0? (Wald, building-clustered covariance)."""
    from scipy.stats import chi2
    P, per = t["P"], t["per"]
    out = []
    for j in range(A.NO):
        base = RL.NB + j * per
        idx = list(range(base, base + per))
        b = t["beta"][idx]
        C = t["cov"][np.ix_(idx, idx)]
        keep = np.diag(C) > 0
        b, C = b[keep], C[np.ix_(keep, keep)]
        try:
            wald = float(b @ np.linalg.pinv(C) @ b)
        except np.linalg.LinAlgError:
            wald = 0.0
        out.append(bool(len(b) and chi2.sf(wald, np.linalg.matrix_rank(C)) < alpha))
    return out


def _room_on(t, j, s, is_ahu, rich):
    """Teacher decision for option j in scenario s: boolean per room (all rooms of the scenario)."""
    sc = A.DATA.scen[s]
    el = sc["elig"][j]
    on = np.zeros(len(el), bool)
    x = _x_of(s)
    RL.SPLIT_FEATS = rich
    if is_ahu:
        for g in np.unique(sc["ahu"][el]):
            idx = np.nonzero(el & (sc["ahu"] == g))[0]
            _, _, F, b, C = room_pred(t, j, RL.Data._rows(sc["rr"], idx, sc["oat"]), x)
            v = F.sum(0)
            on[idx] = float(v @ b) + float(np.sqrt(max(v @ C @ v, 0.0))) < 0
    else:
        idx = np.nonzero(el)[0]
        if len(idx):
            pred, se, _, _, _ = room_pred(t, j, RL.Data._rows(sc["rr"], idx, sc["oat"]), x)
            on[idx] = pred + se < 0
    return on


def option_evidence_xfit(train, lam, seed=0):
    """Per option: (effect per decided-on exposed room, SE) estimated on held-out halves; evidence = eff + SE < 0."""
    blds = sorted({s.split("/")[-2] for s in train})
    rng = np.random.default_rng(seed + 11)
    half = {b: i % 2 for i, b in enumerate(rng.permutation(blds))}
    rich = list(RL.SPLIT_FEATS)
    est = [[] for _ in range(A.NO)]
    for h in (0, 1):
        tr = [s for s in train if half[s.split("/")[-2]] != h]
        te = [s for s in train if half[s.split("/")[-2]] == h and not s.startswith("dg|")]
        trs, tes = set(tr), set(te)
        th = fit_teacher([(oa, oc) for oa, oc in _pairs() if oa["s"] in trs], lam)
        nadd = nonadditive(th)
        dec = {s: [_room_on(th, j, s, nadd[j], rich) for j in range(A.NO)] for s in te}
        pt = [(oa, oc) for oa, oc in _pairs() if oa["s"] in tes]
        per = 2 + RL.NB
        Z = np.zeros((len(pt), RL.NB + per * A.NO))
        for q, (oa, oc) in enumerate(pt):
            Z[q, :RL.NB] = oa["x"]
            sc = A.DATA.scen[oa["s"]]
            for j in range(A.NO):
                hit = ((oa["code"] >> j) & 1).astype(bool) & sc["elig"][j]     # rooms on in this episode
                n_on = float(np.sum(hit & dec[oa["s"]][j]))
                Z[q, RL.NB + per * j] = n_on
                Z[q, RL.NB + per * j + 1] = float(np.sum(hit & ~dec[oa["s"]][j]))
                Z[q, RL.NB + per * j + 2: RL.NB + per * (j + 1)] = n_on * oa["x"]
        y = np.array([oa["y"] for oa, _ in pt])
        w = np.array([oa["w"] for oa, _ in pt])
        sid = np.unique([oa["s"] for oa, _ in pt], return_inverse=True)[1]
        bid = np.unique([oa["b"] for oa, _ in pt], return_inverse=True)[1]
        Zd, yd = RL.demean(Z, sid, w), RL.demean(y, sid, w)
        keep = np.abs(Zd).sum(0) > 1e-12
        ww = w / w.mean()
        M = Zd[:, keep].T @ (Zd[:, keep] * ww[:, None]) + 1e-9 * np.eye(keep.sum())
        b = np.linalg.solve(M, Zd[:, keep].T @ (ww * yd))
        res = yd - Zd[:, keep] @ b
        G = np.zeros((bid.max() + 1, keep.sum()))
        np.add.at(G, bid, Zd[:, keep] * (ww * res)[:, None])
        Mi = np.linalg.inv(M)
        V = Mi @ (G.T @ G) @ Mi
        beta = np.full(Z.shape[1], np.nan)
        se = np.full(Z.shape[1], np.nan)
        beta[keep] = b
        se[keep] = np.sqrt(np.maximum(np.diag(V), 0))
        Vf = np.zeros((Z.shape[1], Z.shape[1]))
        Vf[np.ix_(keep, keep)] = V
        bf = np.where(np.isfinite(beta), beta, 0.0)
        for j in range(A.NO):
            k = RL.NB + per * j
            if not (np.isfinite(beta[k]) and se[k] > 0):
                continue
            # effect at the candidate's settings, averaged over the test scenarios (cost x decided-on rooms)
            g = np.zeros(Z.shape[1])
            wsum = 0.0
            for s in te:
                n_dec = float(np.sum(dec[s][j]))
                if n_dec == 0:
                    continue
                wt = A.DATA.scen[s]["cost"] * n_dec
                g[k] += wt
                g[k + 2: k + 2 + RL.NB] += wt * _x_of(s)
                wsum += wt
            if wsum == 0:
                continue
            g /= wsum
            est[j].append((float(g @ bf), float(np.sqrt(max(g @ Vf @ g, 0.0)))))
    out = []
    for j in range(A.NO):
        if not est[j]:
            out.append((float("nan"), float("nan"), False))
            continue
        wts = np.array([1 / s ** 2 for _, s in est[j]])
        e = float(np.sum(wts * np.array([v for v, _ in est[j]])) / wts.sum())
        s = float(np.sqrt(1 / wts.sum()))
        out.append((e, s, bool(e + s < 0)))
    RL.SPLIT_FEATS = rich
    return out


def nonadditive(t):
    P, per = t["P"], t["per"]
    out = []
    for j in range(A.NO):
        k = RL.NB + j * per + P
        se = float(np.sqrt(max(t["cov"][k, k], 0.0)))
        out.append(bool(se > 0 and abs(t["beta"][k]) > 2.0 * se))
    return out


def fit_c3(train, lam_rel=1.0, min_leaf=0.04, seed=0):
    trs = set(train)
    pairs = [(oa, oc) for oa, oc in _pairs() if oa["s"] in trs]
    A._room_mode("rich")
    t = fit_teacher(pairs, lam_rel)
    nadd = nonadditive(t)
    xf = option_evidence_xfit(train, lam_rel, seed)
    evid = [v[2] for v in xf]
    rich = list(RL.SPLIT_FEATS)
    trees = []
    for j in range(A.NO):
        rows, lab, wq, is_ahu = [], [], [], nadd[j]
        if not evid[j]:                              # no option-level evidence: off everywhere
            trees.append({"depth": 0, "id": 0, "gamma": 0.0, "on": False})
            continue
        for s in train:
            sc = A.DATA.scen.get(s)
            if sc is None or s.startswith("dg|"):
                continue
            el = sc["elig"][j]
            x = _x_of(s)
            RL.SPLIT_FEATS = rich
            if is_ahu:
                for g in np.unique(sc["ahu"][el]):
                    idx = np.nonzero(el & (sc["ahu"] == g))[0]
                    X = RL.Data._rows(sc["rr"], idx, sc["oat"])
                    _, _, F, b, C = room_pred(t, j, X, x)
                    v = F.sum(0)
                    pred, se = float(v @ b), float(np.sqrt(max(v @ C @ v, 0.0)))
                    rows.append(X[0])
                    lab.append(pred if pred + se < 0 else abs(pred) + 1e-12)
                    wq.append(sc["cost"])
            else:
                idx = np.nonzero(el)[0]
                if not len(idx):
                    continue
                X = RL.Data._rows(sc["rr"], idx, sc["oat"])
                pred, se, _, _, _ = room_pred(t, j, X, x)
                rows += list(X)
                lab += list(np.where(pred + se < 0, pred, np.abs(pred) + 1e-12))
                wq += [sc["cost"]] * len(X)
        if is_ahu:
            A._ahu_mode()
            RL.NO = A.NO
        else:
            RL.SPLIT_FEATS = rich
        trees.append(K._distil(np.array(rows), np.array(lab), np.array(wq), min_frac=min_leaf) if rows else
                     {"depth": 0, "id": 0, "gamma": 0.0, "on": False})
    RL.NO = A.NO
    f = RL.RoomFit(trees, t["beta"][:RL.NB])
    f.nonadditive = nadd
    f.evidence = evid
    f.evidence_detail = [(v[0], v[1]) for v in xf]
    return f


def fit_tuned(train, seed=0, inner=3, whole=None):
    """Ridge strength and minimum leaf by an inner leave-buildings-out on the coverage-aware referee; the
    candidate's building-wide settings from the training scenarios."""
    global WHOLE_DEC
    WHOLE_DEC = whole if whole is not None else whole_dec(train)
    real = [s for s in train if s.startswith("train/")]
    blds = sorted({s.split("/")[1] for s in real})
    rng = np.random.default_rng(seed)
    fold = {b: i % inner for i, b in enumerate(rng.permutation(blds))}
    splits = []
    for f in range(inner):
        te = [s for s in real if fold[s.split("/")[1]] == f]
        tb = {s.split("/")[1] for s in te}
        if te:
            splits.append((te, [s for s in train if s.split("/")[1] not in tb]))
    A._room_mode("rich")
    pe = {}
    for lam in LAMBDAS:                       # ridge: inner prediction error of the teacher
        err = wsum = 0.0
        for te, tr in splits:
            tes, trs = set(te), set(tr)
            t_ = fit_teacher([(oa, oc) for oa, oc in _pairs() if oa["s"] in trs], lam)
            pt = [(oa, oc) for oa, oc in _pairs() if oa["s"] in tes]
            At, _, _ = design(pt)
            y = np.array([oa["y"] for oa, _ in pt])
            w = np.array([oa["w"] for oa, _ in pt])
            sid = np.unique([oa["s"] for oa, _ in pt], return_inverse=True)[1]
            r = RL.demean(y, sid, w) - RL.demean(At, sid, w) @ t_["beta"]
            err += float(np.sum(w * r * r))
            wsum += float(np.sum(w))
        pe[lam] = err / wsum
    lam = min(LAMBDAS, key=lambda v: (pe[v], -v))
    scores = {}
    for ml in MIN_LEAF:                       # student's minimum leaf: inner decision value
        acc = wsum = 0.0
        for te, tr in splits:
            fit_ = fit_c3(tr, lam, ml, seed)
            wt = float(sum(A.DATA.scen[s]["cost"] for s in te))
            acc += A.referee(fit_, te) * wt
            wsum += wt
        scores[ml] = acc / wsum
    ml = min(MIN_LEAF, key=lambda v: scores[v])
    f = fit_c3(train, lam, ml, seed)
    f.tuned = {"lambda": lam, "min_leaf": ml, "pred_err": {str(k): v for k, v in pe.items()},
               "leaf_value": {str(k): v for k, v in scores.items()}}
    return f


# ---------------------------------------------------------------- planted-structure smoke (review §四 7)
def smoke():
    """Replace the outcomes by a planted truth on the real design: option 0 null; option 2 additive, on only in rooms
    with exterior wall > 15 m; option 3 non-additive (whole AHU -0.04 per room, partial +0.03 per on room);
    option 4 conditional (effect -0.02 per room only when building-wide factor 0 is on, +0.02 otherwise)."""
    rng = np.random.default_rng(3)
    wi = K.FEATS_X.index("wall")
    for oa, oc in _pairs():
        y = 0.0
        for j in range(A.NO):
            X = oa["rows"].get(j)
            n_on = 0 if X is None else len(X)
            if j == 2 and n_on:
                y += -0.01 * float(np.sum(X[:, wi] > 15.0)) + 0.005 * float(np.sum(X[:, wi] <= 15.0))
            if j == 3:
                for g, (n, c) in oc["cov"][j].items():
                    y += -0.04 * n if c >= 1.0 else (0.03 * round(n * c) if c > 0 else 0.0)
            if j == 4 and n_on:
                y += (-0.02 if oa["x"][0] > 0 else 0.02) * n_on
        y = y / 10.0 + rng.normal(0, 0.002)
        oa["y"] = y
        oc["y"] = y
    global WHOLE_DEC
    WHOLE_DEC = {s: np.array([1.0, -1.0, -1.0]) for s in A.DATA.scen}       # the candidate has factor 0 on
    train = [s for s in A.DATA.scen if not s.startswith("dg|")]
    f = fit_tuned(train, 0, 3, WHOLE_DEC)
    print("tuned:", {k: v for k, v in f.tuned.items() if k != "inner"})
    names = [d["name"] for d in A.C.DELTAS]
    print("non-additive flags:", dict(zip(names, f.nonadditive)))
    print("option-level evidence:", {n: (e, f"{100 * v[0]:+.4f}% per room (SE {100 * v[1]:.4f})")
                                      for n, e, v in zip(names, f.evidence, f.evidence_detail)})
    for j in range(A.NO):
        on = el_n = part = 0
        wall_ok = 0
        for s in train[:80]:
            sc = A.DATA.scen[s]
            el = sc["elig"][j]
            X = RL.Data._rows(sc["rr"], np.arange(len(el)), sc["oat"])
            o = f.on_rows(j, X) & el
            on += int(o.sum())
            el_n += int(el.sum())
            for g in np.unique(sc["ahu"][el]):
                m = el & (sc["ahu"] == g)
                part += int(0 < o[m].mean() < 1) * int(m.sum())
            if j == 2:
                wall_ok += int(np.sum((o == (X[:, wi] > 15.0)) & el))
        extra = f", rooms decided as planted (wall > 15 m) {wall_ok}/{el_n}" if j == 2 else ""
        print(f"  {names[j]:12s} on {on}/{el_n} eligible rooms, in partly covered AHUs {part}{extra}")
    ok = (not f.nonadditive[0] and f.nonadditive[3])
    print("C3FIX SMOKE " + ("PASSED (flags)" if ok else "CHECK THE FLAGS"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    K.setup_features()
    A.DATA = A.AhuData()
    K.DATA = A._AddData(A.DATA)
    K.prepare_scales()
    K.DATA = A.DATA
    print(f"data {time.time() - t0:.0f} s", flush=True)
    if a.smoke:
        smoke()
    return 0


if __name__ == "__main__":
    sys.exit(main())
