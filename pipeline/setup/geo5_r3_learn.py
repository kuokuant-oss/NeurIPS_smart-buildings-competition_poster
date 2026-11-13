"""Stage 戊之三 learning (2026-11-09; spec docs/plan/tree-round3-20261109.md).
Data: geo5_r3_collect (per scenario 16 episodes: whole plot 2^3 x 2 room randomisations) and, after the DAgger step,
extra observations (a candidate run on the selection set against the base gt_r3b).
- Building-wide: per scenario, least squares of y = cost / rule cost - 1 on (1, the three whole-plot columns +-1, each
  room option's share of eligible rooms on) -> each column's main effect (2 beta); then the stage's climate-only kinds
  (geo5_wu_learn / geo5_tk_learn, depth <= 1) on (scenario, factor) units.
- Room tree (point 5): one shallow tree per room option over the deployable room features and the outside air; all
  options and the whole-plot columns are fitted jointly with scenario fixed effects (every column demeaned within its
  scenario):  y(s, run) = a_s + sum_b beta_b x_b + sum_options sum_leaves gamma(option, leaf) n(s, run, option, leaf)
  where n counts the rooms of the leaf with the option on. Splits are grown on half of the buildings (cost-weighted
  SSE of the joint fit), the coefficients estimated on the other half (ridge toward the option's root coefficient); a
  room is on when its leaf's gamma + SE < 0.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_r3_collect as C  # noqa: E402
import geo5_record as R  # noqa: E402
import geo5_tk_room as RM  # noqa: E402
import geo5_wu as W  # noqa: E402
import group_stage as gs  # noqa: E402

FEATS = RM.FEATS
NB = len(C.WHOLE)
NO = len(C.DELTAS)
SPLIT_FEATS = [FEATS.index(f) for f in ("area", "wall", "depth", "ahu_ext_share", "oat")]
BIC_K = 2.0


def rule_cost(spec):
    for d in (C.G / "tree", C.OUT):
        p = d / "records" / R.key("gt_r5pa2", spec)
        if p.is_file():
            rec = json.loads(p.read_text(encoding="utf-8"))
            return float(rec["summary"]["cost"]), float(np.mean(rec["hourly"]["outside_t"]))
    return None, None


class Data:
    """Run-level observations of the design (+ extra DAgger observations)."""

    def __init__(self, specs=None):
        st = json.loads((C.OUT / "pipeline.json").read_text(encoding="utf-8"))
        self.specs = specs or st["specs"]
        feats = json.loads((C.G / "tree2/unit_features.json").read_text(encoding="utf-8"))
        self.obs = []          # dict(s, b (building), w (weight), oat, y, x (NB), rows: {option j: X rows of rooms on})
        self.scen = {}
        for i, s in enumerate(st["specs"]):
            if s not in set(self.specs):
                continue
            norm, oat = rule_cost(s)
            if norm is None:
                continue
            fb = feats[s.split("/")[1]]
            rr = RM.room_rows(fb)
            nums = [int(r) for r in fb["rooms"]]
            cells = [tuple(gs.cell_of(a, w)) for a, w in zip(rr["area"], rr["wall"])]
            elig = [np.array([d["cells"] is None or list(c) in d["cells"] for c in cells]) for d in C.DELTAS]
            got = []
            for r, bits in enumerate(st["rows"]):
                for k in range(st["reps"]):
                    name = f"r3r{r}k{k}s{i:03d}"
                    rp = R.record_path(C.OUT, name, s)
                    if not rp.is_file():
                        continue
                    cost = json.loads(rp.read_text(encoding="utf-8"))["summary"]["cost"]
                    asg = json.loads((C.OUT / "assign" / f"{name}.json").read_text(encoding="utf-8"))
                    code = np.array([int(asg["codes"].get(str(n), 0)) for n in nums])
                    rows = {}
                    for j in range(NO):
                        on = (code >> j & 1).astype(bool) & elig[j]
                        rows[j] = self._rows(rr, np.nonzero(on)[0], oat)
                    got.append({"s": s, "b": s.split("/")[1], "w": norm, "oat": oat, "y": cost / norm - 1.0,
                                "x": np.array([1.0 if v else -1.0 for v in bits]), "rows": rows,
                                "share": np.array([float(((code >> j & 1).astype(bool) & elig[j]).sum())
                                                   / max(1, elig[j].sum()) for j in range(NO)])})
            if len(got) >= 8:
                self.obs += got
                self.scen[s] = {"oat": oat, "cost": norm, "rr": rr, "elig": elig, "n": len(got)}

    @staticmethod
    def _rows(rr, idx, oat):
        """FEATS[:-1], oat, then the graph columns of geo5_tk_room.EXTRA not already in FEATS (2026-11-10: after oat,
        so every existing tree's feature index is unchanged)."""
        extra = [f for f in RM.EXTRA if f in rr and f not in FEATS]
        if not len(idx):
            return np.zeros((0, len(FEATS) + len(extra)))
        return np.column_stack([rr[f][idx] for f in FEATS[:-1]] + [np.full(len(idx), oat)] +
                               [rr[f][idx] for f in extra])

    def add(self, s, b, w, oat, y, x, rows):
        self.obs.append({"s": s, "b": b, "w": w, "oat": oat, "y": y, "x": np.asarray(x, float), "rows": rows,
                         "share": None})


# ---------------------------------------------------------------- building-wide effects per scenario
def whole_effects(data):
    """{spec: {"eff": [e_b...], "oat", "cost"}}: per scenario OLS on (1, x_b, room shares)."""
    per = {}
    by = {}
    for o in data.obs:
        if o["share"] is not None:
            by.setdefault(o["s"], []).append(o)
    for s, os_ in by.items():
        X = np.array([np.concatenate([[1.0], o["x"], o["share"]]) for o in os_])
        y = np.array([o["y"] for o in os_])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        per[s] = {"eff": [2.0 * float(beta[1 + b]) for b in range(NB)], "oat": data.scen[s]["oat"],
                  "cost": data.scen[s]["cost"]}
    return per


# ---------------------------------------------------------------- joint room tree with scenario fixed effects
def _leaf_id(node, X):
    out = np.zeros(len(X), int)

    def rec(n, rows):
        if "f" not in n:
            out[rows] = n["id"]
            return
        m = X[rows, n["f"]] <= n["t"]
        rec(n["l"], rows[m])
        rec(n["r"], rows[~m])
    if len(X):
        rec(node, np.arange(len(X)))
    return out


def _number(node, start=0):
    if "f" not in node:
        node["id"] = start
        return start + 1
    return _number(node["r"], _number(node["l"], start))


def _leaves(node):
    return [node] if "f" not in node else _leaves(node["l"]) + _leaves(node["r"])


class Prep:
    """Stacked arrays of a list of observations (fast designs)."""

    def __init__(self, obs):
        self.obs = obs
        self.n = len(obs)
        self.xb = np.array([o["x"] for o in obs]) if obs else np.zeros((0, NB))
        self.y = np.array([o["y"] for o in obs])
        self.w = np.array([o["w"] for o in obs])
        self.sid = np.unique([o["s"] for o in obs], return_inverse=True)[1] if obs else np.zeros(0, int)
        self.X, self.q = {}, {}
        for j in range(NO):
            parts = [(q, o["rows"][j]) for q, o in enumerate(obs) if o["rows"].get(j) is not None and len(o["rows"][j])]
            self.X[j] = np.vstack([x for _, x in parts]) if parts else np.zeros((0, len(FEATS)))
            self.q[j] = np.concatenate([np.full(len(x), q) for q, x in parts]) if parts else np.zeros(0, int)


def _prep(obs):
    return obs if isinstance(obs, Prep) else Prep(obs)


def design(obs, trees):
    """Columns: x_b, then for every option its leaf counts. Returns A, y, w, scenario index, starts, sizes."""
    pr = _prep(obs)
    sizes = [_number(t) for t in trees]
    A = np.zeros((pr.n, NB + sum(sizes)))
    A[:, :NB] = pr.xb
    starts, off = [], NB
    for j, sz in enumerate(sizes):
        starts.append(off)
        if len(pr.X[j]):
            np.add.at(A, (pr.q[j], off + _leaf_id(trees[j], pr.X[j])), 1.0)
        off += sz
    return A, pr.y, pr.w, pr.sid, starts, sizes


def demean(M, sid, w):
    out = M.astype(float).copy()
    for g in np.unique(sid):
        m = sid == g
        ww = w[m] / w[m].sum()
        out[m] -= (ww[:, None] * M[m]).sum(0) if M.ndim == 2 else (ww * M[m]).sum()
    return out


def wls(A, y, w, ridge=None, lam=0.0):
    ww = w / w.mean()
    M = A.T @ (A * ww[:, None])
    v = A.T @ (ww * y)
    if ridge is not None and lam > 0:
        M = M + lam * np.diag(ridge[0])
        v = v + lam * ridge[0] * ridge[1]
    beta = np.linalg.solve(M + 1e-12 * np.eye(A.shape[1]), v)
    res = y - A @ beta
    s2 = float(np.sum(ww * res ** 2) / max(1, len(y) - A.shape[1]))
    Minv = np.linalg.pinv(M)
    cov = s2 * Minv @ (A.T @ (A * (ww ** 2)[:, None])) @ Minv
    return beta, cov, float(np.sum(ww * res ** 2))


def fit_joint(obs, trees):
    A, y, w, sid, starts, sizes = design(obs, trees)
    return wls(demean(A, sid, w), demean(y, sid, w), w), (starts, sizes)


def grow_joint(obs, depth=2, min_scen=6, nq=7):
    """Greedy over every option's leaves: the split that most lowers the joint fixed-effects SSE; each leaf keeps rooms
    in >= min_scen scenarios."""
    obs = _prep(obs)
    trees = [{"depth": 0} for _ in range(NO)]
    allx = obs.X
    for _ in range(NO * (2 ** depth - 1)):
        (b0, _, sse0), _ = fit_joint(obs, trees)
        s2 = sse0 / max(1, obs.n - len(np.unique(obs.sid)) - len(b0))
        best = None
        for j, t in enumerate(trees):
            if j not in allx or not len(allx[j]):
                continue
            for lf in _leaves(t):
                if lf.get("depth", 0) >= depth:
                    continue
                for fidx in SPLIT_FEATS:
                    for thr in np.unique(np.quantile(allx[j][:, fidx], np.linspace(0.15, 0.85, nq))):
                        lf.update(f=fidx, t=float(thr), l={"depth": lf.get("depth", 0) + 1},
                                  r={"depth": lf.get("depth", 0) + 1})
                        A, _, _, sid, starts, sizes = design(obs, trees)
                        sup_ok = True
                        for jj, (st0, sz) in enumerate(zip(starts, sizes)):
                            cnt = A[:, st0:st0 + sz] > 0
                            n_s = [len(np.unique(sid[cnt[:, c]])) for c in range(sz)]
                            if min(n_s) < min_scen:
                                sup_ok = False
                                break
                        if sup_ok:
                            (_, _, sse), _ = fit_joint(obs, trees)
                            if best is None or sse0 - sse > best[0]:
                                best = (sse0 - sse, j, lf, fidx, float(thr))
                        for key in ("f", "t", "l", "r"):
                            lf.pop(key)
        if best is None or best[0] <= BIC_K * s2 * np.log(max(obs.n, 2)):     # BIC-sized gain
            break
        _, j, lf, fidx, thr = best
        d = lf.get("depth", 0)
        lf.update(f=fidx, t=thr, l={"depth": d + 1}, r={"depth": d + 1})
    return trees


class RoomFit:
    def __init__(self, trees, beta_b):
        self.trees, self.beta_b = trees, beta_b

    def on_rows(self, j, X):
        out = np.zeros(len(X), bool)
        if not len(X):
            return out
        ids = _leaf_id(self.trees[j], X)
        leaves = {lf["id"]: lf for lf in _leaves(self.trees[j])}
        for k, i in enumerate(ids):
            out[k] = leaves[i].get("on", False)
        return out

    def gamma_rows(self, j, X):
        if not len(X):
            return np.zeros(0)
        ids = _leaf_id(self.trees[j], X)
        leaves = {lf["id"]: lf for lf in _leaves(self.trees[j])}
        return np.array([leaves[i].get("gamma", 0.0) for i in ids])


def fit_rooms(obs, seed=0, depth=2, lam=2.0, honest=False):
    """Joint fit. honest=True: structure on half of the buildings, coefficients on the other half; default: both on
    every building (BIC-stopped growth; the honest check is the selection set)."""
    blds = sorted({o["b"] for o in obs})
    half = set(np.random.default_rng(seed).permutation(blds)[: max(1, len(blds) // 2)].tolist())
    if honest:
        oa = Prep([o for o in obs if o["b"] in half])
        ob = Prep([o for o in obs if o["b"] not in half])
    else:
        oa = ob = Prep(list(obs))
    trees = grow_joint(oa, depth)
    root = [{"depth": 0} for _ in range(NO)]
    (rb, _, _), (rs, _) = fit_joint(ob, root)
    A, y, w, sid, starts, sizes = design(ob, trees)
    prior = np.zeros(A.shape[1])
    pen = np.zeros(A.shape[1])
    for j, (st0, sz) in enumerate(zip(starts, sizes)):
        prior[st0:st0 + sz] = rb[rs[j]]
        pen[st0:st0 + sz] = 1.0
    Ad, yd = demean(A, sid, w), demean(y, sid, w)
    beta, cov, _ = wls(Ad, yd, w, (pen, prior), lam * ob.n / max(1, A.shape[1]))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    for j, (st0, sz) in enumerate(zip(starts, sizes)):
        cnt = A[:, st0:st0 + sz] > 0
        for lf in _leaves(trees[j]):
            c = st0 + lf["id"]
            n_s = len(np.unique(sid[cnt[:, lf["id"]]]))
            lf.update(gamma=float(beta[c]), se=float(se[c]), n=int(n_s),
                      on=bool(n_s >= 6 and beta[c] + se[c] < 0.0))
    return RoomFit(trees, beta[:NB])


def room_value(fit, data, scenarios):
    """Model-based cost-weighted change (share of the rule cost) if each scenario's rooms follow the fit."""
    num = den = 0.0
    for s in scenarios:
        sc = data.scen.get(s)
        if sc is None:
            continue
        tot = 0.0
        for j in range(NO):
            idx = np.nonzero(sc["elig"][j])[0]
            X = Data._rows(sc["rr"], idx, sc["oat"])
            on = fit.on_rows(j, X)
            tot += float(np.sum(fit.gamma_rows(j, X) * on))
        num += sc["cost"] * tot
        den += sc["cost"]
    return num / den if den else 0.0


def calibration(fit, obs):
    """Out-of-sample check on held-out observations: within-scenario demeaned prediction vs actual (corr, slope)."""
    A, y, w, sid, starts, sizes = design(obs, fit.trees)
    coef = np.concatenate([fit.beta_b] + [np.array([lf["gamma"] for lf in sorted(_leaves(t), key=lambda q: q["id"])])
                                          for t in fit.trees])
    pred = demean(A, sid, w) @ coef
    act = demean(y, sid, w)
    if np.std(pred) == 0:
        return {"corr": 0.0, "slope": 0.0}
    return {"corr": float(np.corrcoef(pred, act)[0, 1]),
            "slope": float(np.sum(pred * act) / np.sum(pred * pred))}
