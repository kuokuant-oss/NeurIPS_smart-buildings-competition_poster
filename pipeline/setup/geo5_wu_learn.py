"""Stage 戊 learning (2026-11-06; spec docs/plan/tree-stage-wu-20261106.md §3): round 4's tree step
(group_stage4.py: Units of (scenario, factor) effects, honest shallow trees, the four ways me / tree / bag / reg,
leave-buildings-out on the real buildings only, trained with every generated scenario) plus appendix A:
- cost-weighted everywhere (weight = the scenario's paper-rule cost; a leaf's value = the weighted mean over scenarios
  of the summed effects of the leaf's units), so the ranking matches the score;
- leaf effects shrunk toward the root (hierarchical shrinkage: each step down the path divided by 1 + lambda / n);
- depth and shrinkage chosen by an inner leave-buildings-out ("tuned");
- structures 甲 joint (one tree, leaves give the combined action of every option type), 乙 per-type in sequence
  (building-wide climate stumps, then the room trees = "tuned"), 丙 a smooth teacher (weighted ridge on climate,
  cell, base-map context and AHU / neighbour features with physically motivated cross terms) distilled into shallow
  trees over the deployable features;
- merges into one shallow tree: born-again (the bag's vote function, compressed to depth <= 2, fidelity reported),
  RuleFit (rules of the bag + weighted lasso), FIGS (a sum of small trees);
- building-wide (whole-plot) factors: climate only, depth <= 1 (lit-two-level §4 step 4);
- reports only: learning curve over the number of generated buildings, physical vs adjacency-graph representation.
Deployable features: area band, wall band of the factor's cells, the episode running-mean outside air."""
import math

import numpy as np

import group_stage4 as g4

GRID = g4.GRID
FEAT = ("area", "wall", "oat")
OATF = 2
ROUND4 = {"weighted": False, "agg": "mean", "lam": 0.0}       # round 4 exactly (regression check)
UPGRADE = {"weighted": True, "agg": "sum", "lam": 0.0}


# ---------------------------------------------------------------- units
class Units:
    """(scenario, factor) units: S scenario, B building, I factor, T type, Y effect, X deployable features
    (area, wall, oat; building-wide factors -1, -1, oat), WS the scenario weights, Z extra features (dict)."""

    def __init__(self, per, factors, active, weighted=True, extra=None):
        self.factors, self.active = factors, list(active)
        self.scen = sorted(per)
        self.sid = {s: k for k, s in enumerate(self.scen)}
        blds = sorted({s.split("/")[1] for s in self.scen})
        self.bid = {b: k for k, b in enumerate(blds)}
        self.SB = np.array([self.bid[s.split("/")[1]] for s in self.scen])
        self.WS = np.array([per[s]["cost"] if weighted else 1.0 for s in self.scen], float)
        self.OAT = np.array([per[s]["oat"] for s in self.scen], float)
        self.types = sorted({factors[i]["type"] for i in self.active})
        self.whole_types = {t for t in self.types if t.startswith("bldg:")}
        rows, zs = [], []
        for s in self.scen:
            for i in self.active:
                e = per[s]["eff"][i]
                if e is None:
                    continue
                f = factors[i]
                wh = f.get("whole", False)
                rows.append((self.sid[s], self.bid[s.split("/")[1]], i, e, -1.0 if wh else f["area"],
                             -1.0 if wh else f["wall"], per[s]["oat"]))
                if extra is not None:
                    zs.append(extra(s, i))
        arr = np.array(rows, float).reshape(-1, 7)
        self.S, self.B, self.I = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2].astype(int)
        self.Y, self.X = arr[:, 3], arr[:, 4:7]
        self.T = np.array([self.types.index(factors[i]["type"]) for i in self.I], int)
        self.Z = {k: np.array([z[k] for z in zs], float) for k in (zs[0] if zs else {})}

    def mask(self, scenarios):
        return np.isin(self.S, [self.sid[s] for s in scenarios if s in self.sid])

    def is_whole(self, t):
        return self.types[t] in self.whole_types


# ---------------------------------------------------------------- weighted, shrunk honest trees
def scen_totals(U, idx, agg):
    uniq, inv = np.unique(U.S[idx], return_inverse=True)
    tot = np.bincount(inv, U.Y[idx])
    if agg == "mean":
        tot = tot / np.bincount(inv)
    return uniq, tot


def node_stats(U, idx, cfg):
    if len(idx) == 0:
        return {"n": 0, "mean": 0.0, "se": float("inf"), "wsum": 0.0}
    uniq, tot = scen_totals(U, idx, cfg["agg"])
    w = U.WS[uniq] if cfg["weighted"] else np.ones(len(uniq))
    n = len(uniq)
    mean = float(np.sum(w * tot) / np.sum(w))
    if n < 2:
        return {"n": n, "mean": mean, "se": float("inf"), "wsum": float(np.sum(w))}
    if cfg["weighted"]:
        se = float(math.sqrt(n / (n - 1) * np.sum((w * (tot - mean)) ** 2)) / np.sum(w))
    else:
        se = float(np.std(tot, ddof=1) / math.sqrt(n))
    return {"n": n, "mean": mean, "se": se, "wsum": float(np.sum(w))}


def _best_split(U, idx, min_n, allowed, cfg):
    y, s = U.Y[idx], U.S[idx]
    w = U.WS[s] if cfg["weighted"] else np.ones(len(idx))
    W = w.sum()
    base_sse = float(np.sum(w * (y - np.sum(w * y) / W) ** 2))
    best = None
    for j in allowed:
        x = U.X[idx, j]
        order = np.argsort(x, kind="stable")
        xs, ys, ss, ws = x[order], y[order], s[order], w[order]
        n = len(ys)
        if n < 2:
            continue
        cw, cy, cy2 = np.cumsum(ws), np.cumsum(ws * ys), np.cumsum(ws * ys * ys)
        pl, pr = np.empty(n, int), np.empty(n, int)
        seen = set()
        for k in range(n):
            seen.add(ss[k])
            pl[k] = len(seen)
        seen = set()
        for k in range(n - 1, -1, -1):
            seen.add(ss[k])
            pr[k] = len(seen)
        k = np.arange(n - 1)
        wl, wr = cw[k], cw[-1] - cw[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            sse = (cy2[k] - cy[k] ** 2 / wl) + ((cy2[-1] - cy2[k]) - (cy[-1] - cy[k]) ** 2 / wr)
        gain = base_sse - sse
        gain[~((xs[k] < xs[k + 1]) & (pl[k] >= min_n) & (pr[k + 1] >= min_n))] = -np.inf
        if not np.isfinite(gain).any():
            continue
        b = int(np.argmax(gain))
        if gain[b] > 1e-12 * (W / len(idx) if cfg["weighted"] else 1.0) and (best is None or gain[b] > best[0]):
            best = (float(gain[b]), j, float((xs[b] + xs[b + 1]) / 2.0))
    return best


def _grow(U, idx, levels, min_n, cfg, used_oat=False, whole=False):
    if not levels or len(idx) < 2:
        return {}
    allowed = [j for j in levels[0] if not (j == OATF and (used_oat and (whole or cfg.get("no_oat_twice"))))]
    if whole:
        allowed = [j for j in allowed if j == OATF]
    split = _best_split(U, idx, min_n, allowed, cfg)
    if split is None:
        return {}
    _, j, t = split
    m = U.X[idx, j] <= t
    return {"f": j, "t": t, "l": _grow(U, idx[m], levels[1:], min_n, cfg, used_oat or j == OATF, whole),
            "r": _grow(U, idx[~m], levels[1:], min_n, cfg, used_oat or j == OATF, whole)}


def _fill(node, U, idx, cfg, min_leaf, parent=None):
    """Estimation-half statistics at every node; leaf value shrunk along the path (Agarwal et al. 2022)."""
    st = node_stats(U, idx, cfg)
    node["stat"] = st
    if parent is None:
        node["shrunk"] = st["mean"]
    else:
        pst = parent["stat"]
        node["shrunk"] = parent["shrunk"] + (st["mean"] - pst["mean"]) / (1.0 + cfg["lam"] / max(1, pst["n"]))
    if "f" not in node:
        node["on"] = bool(st["n"] >= min_leaf and node["shrunk"] + st["se"] < 0.0)
        return
    m = U.X[idx, node["f"]] <= node["t"]
    _fill(node["l"], U, idx[m], cfg, min_leaf, node)
    _fill(node["r"], U, idx[~m], cfg, min_leaf, node)


def walk(node, X, key="on"):
    if "f" not in node:
        v = node[key] if key == "on" else node.get(key, node["stat"]["mean"])
        return np.full(len(X), float(v))
    out = np.zeros(len(X))
    m = X[:, node["f"]] <= node["t"]
    if m.any():
        out[m] = walk(node["l"], X[m], key)
    if (~m).any():
        out[~m] = walk(node["r"], X[~m], key)
    return out


def honest_tree(U, idx, min_leaf, levels, seed, cfg, whole=False):
    b = np.unique(U.B[idx])
    rng = np.random.default_rng(seed)
    half = set(rng.permutation(b)[: max(1, len(b) // 2)].tolist())
    in_a = np.array([x in half for x in U.B[idx]])
    node = _grow(U, idx[in_a], levels, max(3, min_leaf // 2), cfg, whole=whole)
    _fill(node, U, idx[~in_a], cfg, min_leaf)
    return node


def depth_levels(levels, depth):
    return [list(x) for x in levels[:depth]]


# ---------------------------------------------------------------- fits
class Fit:
    """A decision per unit: kind me / tree / bag / reg / tuned / joint / teacher / born / rulefit / figs."""

    def __init__(self, kind, U, models, info=None, cfg=None):
        self.kind, self.U, self.models, self.info, self.cfg = kind, U, models, info or {}, cfg

    def on(self, X, I):
        X, I = np.asarray(X, float).reshape(-1, 3), np.asarray(I, int)
        if self.kind == "me":
            return np.array([bool(self.models.get(int(i), False)) for i in I])
        types = np.array([self.U.types.index(self.U.factors[int(i)]["type"]) for i in I])
        out = np.zeros(len(X), bool)
        if self.kind == "joint":
            node = self.models["joint"]
            for t in np.unique(types):
                m = types == t
                out[m] = _walk_joint(node, X[m], t) > 0.5
            return out
        for t in np.unique(types):
            m = types == t
            model = self.models.get(int(t))
            if model is None:
                continue
            if self.kind == "bag":
                out[m] = np.mean([walk(n, X[m]) for n in model], axis=0) >= 0.5
            elif callable(model):
                out[m] = model(X[m])
            else:
                out[m] = walk(model, X[m]) > 0.5
        return out


def score(U, fit, scenarios, cfg):
    """Cost-weighted (or equal-weight) mean over scenarios of the summed effects of the units switched on."""
    ids = [U.sid[s] for s in scenarios if s in U.sid]
    if not ids:
        return 0.0
    m = np.isin(U.S, ids)
    on = fit.on(U.X[m], U.I[m])
    tot = np.bincount(U.S[m], U.Y[m] * on, minlength=len(U.scen))[ids]
    w = U.WS[ids] if cfg["weighted"] else np.ones(len(ids))
    return float(np.sum(w * tot) / np.sum(w))


def type_idx(U, mask, t):
    return np.where(mask & (U.T == t))[0]


def inner_folds(U, idx, seed, k=3):
    b = np.unique(U.B[idx])
    fold = {x: j % k for j, x in enumerate(np.random.default_rng(seed).permutation(b))}
    return np.array([fold[x] for x in U.B[idx]])


def _inner_score(U, idx, fold_of, f, node_fn):
    node = node_fn(idx[fold_of != f])
    test = idx[fold_of == f]
    on = walk(node, U.X[test]) > 0.5
    uniq, inv = np.unique(U.S[test], return_inverse=True)
    tot = np.bincount(inv, U.Y[test] * on)
    w = U.WS[uniq]
    return float(np.sum(w * tot) / np.sum(w))


def fit_kind(kind, U, scenarios, cfg, levels=None, min_leaf=6, bags=50, seed=0):
    levels = levels or g4.DEFAULT_LEVELS
    mask = U.mask(scenarios)
    c = dict(cfg)
    if kind == "me":
        models = {}
        for i in U.active:
            st = node_stats(U, np.where(mask & (U.I == i))[0], c)
            models[i] = st["n"] >= 2 and st["mean"] + st["se"] < 0.0
        return Fit("me", U, models, cfg=c)
    if kind == "joint":
        return fit_joint(U, mask, c, levels, min_leaf, seed)
    if kind == "teacher":
        return fit_teacher(U, scenarios, c, seed)
    if kind in TERM_SETS:                      # representation comparison (paper): the same teacher, other features
        f = fit_teacher(U, scenarios, c, seed, TERM_SETS[kind])
        f.kind = kind
        return f
    if kind in ("born", "rulefit", "figs"):
        return fit_merge(kind, U, scenarios, c, levels, min_leaf, bags, seed)
    models, info = {}, {}
    for t in range(len(U.types)):
        idx = type_idx(U, mask, t)
        if len(idx) < 4:
            continue
        wh = U.is_whole(t)
        lv = [[OATF]] if wh else levels
        if kind == "tree":
            models[t] = honest_tree(U, idx, min_leaf, lv, seed + t, c, wh)
        elif kind == "bag":
            b = np.unique(U.B[idx])
            rng = np.random.default_rng(seed + 100 + t)
            trees = []
            for r in range(bags):
                draw = rng.choice(b, size=len(b), replace=True)
                sub = idx[np.isin(U.B[idx], np.unique(draw))]
                trees.append(honest_tree(U, sub, min_leaf, lv, seed + 1000 * (r + 1) + t, c, wh))
            models[t] = trees
        elif kind == "reg":
            fold_of = inner_folds(U, idx, seed + 7)
            sc = {}
            for ml in (6, 10, 16, 24):
                sc[ml] = float(np.mean([_inner_score(U, idx, fold_of, f, lambda tr, ml=ml, f=f: honest_tree(
                    U, tr, ml, lv, seed + 31 * f + t, c, wh)) for f in range(3)])) if cfg["weighted"] else \
                    _r4_reg_inner(U, idx, fold_of, ml, lv, seed, t, c, wh)
            best = min(sc, key=lambda m: (round(sc[m], 9), -m))
            models[t] = honest_tree(U, idx, best, lv, seed + t, c, wh)
            info[U.types[t]] = {"min_leaf": best}
        elif kind == "tuned":
            fold_of = inner_folds(U, idx, seed + 7)
            sc = {}
            depths = (0, 1) if wh else (0, 1, 2)
            for d in depths:
                for lam in (0.0, 10.0, 40.0):
                    if d == 0 and lam > 0:
                        continue
                    cc = dict(c, lam=lam)
                    sc[(d, lam)] = float(np.mean([_inner_score(U, idx, fold_of, f, lambda tr, d=d, cc=cc, f=f:
                                                               honest_tree(U, tr, min_leaf, depth_levels(lv, d),
                                                                           seed + 31 * f + t, cc, wh))
                                                  for f in range(3)]))
            best = min(sc, key=lambda k: (round(sc[k], 9), k[0], -k[1]))
            models[t] = honest_tree(U, idx, min_leaf, depth_levels(lv, best[0]), seed + t, dict(c, lam=best[1]), wh)
            info[U.types[t]] = {"depth": best[0], "lam": best[1]}
        else:
            raise ValueError(kind)
    return Fit(kind, U, models, info, c)


def _r4_reg_inner(U, idx, fold_of, ml, lv, seed, t, c, wh):
    """Round 4's inner score exactly (equal weights, divided by the test scenario count)."""
    tot = []
    for f in range(3):
        node = honest_tree(U, idx[fold_of != f], ml, lv, seed + 31 * f + t, c, wh)
        test = idx[fold_of == f]
        on = walk(node, U.X[test]) > 0.5
        tot.append(float(np.sum(U.Y[test] * on)) / max(1, len(np.unique(U.S[test]))))
    return float(np.mean(tot))


# ---------------------------------------------------------------- 甲 joint tree (combined action in the leaves)
def _joint_value(U, idx, cfg, ntypes):
    """Decision value of a node: the sum over types of min(0, weighted leaf mean) (the saving of its best action)."""
    v = 0.0
    for t in range(ntypes):
        st = node_stats(U, idx[U.T[idx] == t], cfg)
        if st["n"]:
            v += min(0.0, st["mean"]) * st["wsum"]
    return v


def _joint_grow(U, idx, levels, min_n, cfg, used_oat=False):
    if not levels or len(idx) < 2:
        return {}
    ntypes = len(U.types)
    best, base = None, _joint_value(U, idx, cfg, ntypes)
    for j in levels[0]:
        if j == OATF and used_oat:
            continue
        xs = np.unique(U.X[idx, j][U.X[idx, j] >= 0] if j != OATF else U.X[idx, j])
        for a, b in zip(xs[:-1], xs[1:]):
            t = (a + b) / 2.0
            m = U.X[idx, j] <= t
            nl, nr = len(np.unique(U.S[idx[m]])), len(np.unique(U.S[idx[~m]]))
            if nl < min_n or nr < min_n:
                continue
            gain = base - (_joint_value(U, idx[m], cfg, ntypes) + _joint_value(U, idx[~m], cfg, ntypes))
            if gain > 1e-12 and (best is None or gain > best[0]):
                best = (gain, j, float(t))
    if best is None:
        return {}
    _, j, t = best
    m = U.X[idx, j] <= t
    return {"f": j, "t": t, "l": _joint_grow(U, idx[m], levels[1:], min_n, cfg, used_oat or j == OATF),
            "r": _joint_grow(U, idx[~m], levels[1:], min_n, cfg, used_oat or j == OATF)}


def _joint_fill(node, U, idx, cfg, min_leaf, parent=None):
    node["stat"] = [node_stats(U, idx[U.T[idx] == t], cfg) for t in range(len(U.types))]
    if parent is None:
        node["shrunk"] = [s["mean"] for s in node["stat"]]
    else:
        node["shrunk"] = [p + (s["mean"] - ps["mean"]) / (1.0 + cfg["lam"] / max(1, ps["n"]))
                          for p, s, ps in zip(parent["shrunk"], node["stat"], parent["stat"])]
    if "f" not in node:
        node["on"] = [bool(s["n"] >= min_leaf and v + s["se"] < 0.0) for s, v in zip(node["stat"], node["shrunk"])]
        return
    m = U.X[idx, node["f"]] <= node["t"]
    _joint_fill(node["l"], U, idx[m], cfg, min_leaf, node)
    _joint_fill(node["r"], U, idx[~m], cfg, min_leaf, node)


def _walk_joint(node, X, t):
    if "f" not in node:
        return np.full(len(X), float(node["on"][t]))
    out = np.zeros(len(X))
    m = X[:, node["f"]] <= node["t"]
    if m.any():
        out[m] = _walk_joint(node["l"], X[m], t)
    if (~m).any():
        out[~m] = _walk_joint(node["r"], X[~m], t)
    return out


def fit_joint(U, mask, cfg, levels, min_leaf, seed):
    """One tree over every type's units; whole-plot units have area = wall = -1, so they follow one side of every
    cell split and see at most one climate split (no climate split twice on a path)."""
    idx = np.where(mask)[0]
    fold_of = inner_folds(U, idx, seed + 7)
    sc = {}
    for d in (1, 2):
        for lam in (0.0, 10.0, 40.0):
            cc = dict(cfg, lam=lam)
            vals = []
            for f in range(3):
                node = _joint_honest(U, idx[fold_of != f], min_leaf, depth_levels(levels, d), seed + 31 * f, cc)
                fit = Fit("joint", U, {"joint": node}, cfg=cc)
                test = [U.scen[k] for k in np.unique(U.S[idx[fold_of == f]])]
                vals.append(score(U, fit, test, cc))
            sc[(d, lam)] = float(np.mean(vals))
    best = min(sc, key=lambda k: (round(sc[k], 9), k[0], -k[1]))
    cc = dict(cfg, lam=best[1])
    node = _joint_honest(U, idx, min_leaf, depth_levels(levels, best[0]), seed, cc)
    return Fit("joint", U, {"joint": node}, {"depth": best[0], "lam": best[1]}, cc)


def _joint_honest(U, idx, min_leaf, levels, seed, cfg):
    b = np.unique(U.B[idx])
    half = set(np.random.default_rng(seed).permutation(b)[: max(1, len(b) // 2)].tolist())
    in_a = np.array([x in half for x in U.B[idx]])
    node = _joint_grow(U, idx[in_a], levels, max(3, min_leaf // 2), cfg)
    _joint_fill(node, U, idx[~in_a], cfg, min_leaf)
    return node


# ---------------------------------------------------------------- 丙 smooth teacher distilled into shallow trees
TEACHER_TERMS = ("oat", "oat2", "area", "wall", "oat_area", "oat_wall", "has_min", "base_ratio", "base_cool",
                 "base_heat", "oat_has_min", "oat_base_ratio", "ahu_rooms", "ahu_ext", "nb_same", "depth")


BASE_CONTEXT = ("oat", "oat2", "area", "wall", "oat_area", "oat_wall", "has_min", "base_ratio", "base_cool",
                "base_heat", "oat_has_min", "oat_base_ratio")
TERM_SETS = {"teacher_phys": BASE_CONTEXT + ("ahu_rooms", "ahu_ext", "depth", "wall_ns", "wall_ew"),
             "teacher_graph": BASE_CONTEXT + ("nb_count", "nb_wall", "nb_on", "nb_same")}


def teacher_design(U, rows, terms):
    o = (U.X[rows, OATF] - 15.0) / 12.0
    cols = {"oat": o, "oat2": o * o, "area": U.X[rows, 0], "wall": U.X[rows, 1],
            "oat_area": o * U.X[rows, 0], "oat_wall": o * U.X[rows, 1]}
    for k in ("has_min", "base_ratio", "base_cool", "base_heat", "ahu_rooms", "ahu_ext", "nb_same", "depth",
              "nb_count", "nb_wall", "nb_on", "wall_ns", "wall_ew"):
        if k in U.Z:
            cols[k] = U.Z[k][rows]
    if "has_min" in cols:
        cols["oat_has_min"] = o * cols["has_min"]
    if "base_ratio" in cols:
        cols["oat_base_ratio"] = o * cols["base_ratio"]
    use = [t for t in terms if t in cols]
    return np.column_stack([np.ones(len(rows))] + [cols[t] for t in use]), use


def ridge(X, y, w, lam):
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    pen = lam * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.solve(Xw.T @ Xw + pen, Xw.T @ yw)


def _standardise(X):
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def fit_teacher(U, scenarios, cfg, seed, terms=TEACHER_TERMS):
    mask = U.mask(scenarios)
    models, info = {}, {}
    for t in range(len(U.types)):
        idx = type_idx(U, mask, t)
        if len(idx) < 8:
            continue
        wh = U.is_whole(t)
        tt = ("oat", "oat2") if wh else terms
        X, use = teacher_design(U, idx, tt)
        mu, sd = _standardise(X)
        Xs = X.copy()
        Xs[:, 1:] = (X[:, 1:] - mu) / sd
        w = U.WS[U.S[idx]]
        fold_of = inner_folds(U, idx, seed + 11)
        sc = {}
        for lam in (1.0, 10.0, 100.0, 1000.0):
            err = 0.0
            for f in range(3):
                tr, te = fold_of != f, fold_of == f
                beta = ridge(Xs[tr], U.Y[idx][tr], w[tr], lam * w[tr].mean())
                err += float(np.sum(w[te] * (U.Y[idx][te] - Xs[te] @ beta) ** 2))
            sc[lam] = err
        lam = min(sc, key=sc.get)
        beta = ridge(Xs, U.Y[idx], w, lam * w.mean())
        pred = Xs @ beta
        # distil: a shallow tree on the deployable features, regression on the teacher's predictions (all units of
        # the training scenarios, real and generated), a leaf is on when its weighted mean prediction is < 0
        student = _distil(U, idx, pred, w, 1 if wh else 2, wh)
        models[t] = student
        info[U.types[t]] = {"ridge_lambda": lam, "terms": use}
    return Fit("teacher", U, models, info, cfg)


def _distil(U, idx, target, w, depth, whole, min_n=6):
    """Greedy weighted regression tree (depth <= depth) of `target` on the deployable features; leaves on when the
    weighted mean target < 0. Returns a predicate over X."""
    def grow(rows, d, used_oat):
        if d == 0 or len(rows) < 2 * min_n:
            return {"v": float(np.sum(w[rows] * target[rows]) / np.sum(w[rows]))}
        best = None
        tot = np.sum(w[rows] * (target[rows] - np.sum(w[rows] * target[rows]) / np.sum(w[rows])) ** 2)
        for j in ((OATF,) if whole else (0, 1, OATF)):
            if j == OATF and used_oat:
                continue
            x = U.X[idx[rows], j]
            for t in np.unique(x)[:-1]:
                m = x <= t
                if m.sum() < min_n or (~m).sum() < min_n:
                    continue
                sse = 0.0
                for part in (rows[m], rows[~m]):
                    mu = np.sum(w[part] * target[part]) / np.sum(w[part])
                    sse += np.sum(w[part] * (target[part] - mu) ** 2)
                if best is None or tot - sse > best[0]:
                    nxt = np.min(x[x > t])
                    best = (tot - sse, j, float((t + nxt) / 2.0))
        if best is None or best[0] <= 1e-15:
            return {"v": float(np.sum(w[rows] * target[rows]) / np.sum(w[rows]))}
        _, j, t = best
        m = U.X[idx[rows], j] <= t
        return {"f": j, "t": t, "l": grow(rows[m], d - 1, used_oat or j == OATF),
                "r": grow(rows[~m], d - 1, used_oat or j == OATF)}

    node = grow(np.arange(len(idx)), depth, False)

    def pred(X):
        out = np.zeros(len(X), bool)
        for k in range(len(X)):
            n = node
            while "f" in n:
                n = n["l"] if X[k, n["f"]] <= n["t"] else n["r"]
            out[k] = n["v"] < 0.0
        return out
    pred.tree = node
    return pred


# ---------------------------------------------------------------- merges into one shallow tree
def _query_points(U, idx):
    """The units themselves (their own feature rows) plus each of their factors on a climate grid over the observed
    range: the bag / rule / FIGS decision functions are compressed where data exist."""
    lo, hi = np.percentile(U.X[idx, OATF], [1, 99])
    grid = np.arange(math.floor(lo), math.ceil(hi) + 0.01, 0.5)
    cells = np.unique(U.X[idx, :2], axis=0)
    q = [np.column_stack([np.full(len(grid), a), np.full(len(grid), b), grid]) for a, b in cells]
    return np.vstack([U.X[idx]] + q)


def _compress(U, idx, decide, depth, whole):
    """A depth <= depth classification tree over the deployable features reproducing decide() on the query points
    (weighted misclassification); returns the predicate and its fidelity."""
    Q = _query_points(U, idx)
    lab = decide(Q).astype(float)
    w = np.ones(len(Q))
    rows = np.arange(len(Q))

    def grow(rows, d, used_oat):
        p = float(np.sum(w[rows] * lab[rows]) / np.sum(w[rows]))
        if d == 0 or len(rows) < 2:
            return {"v": p >= 0.5}
        err0 = min(p, 1 - p) * np.sum(w[rows])
        best = None
        for j in ((OATF,) if whole else (0, 1, OATF)):
            if j == OATF and used_oat:
                continue
            x = Q[rows, j]
            for t in np.unique(x)[:-1]:
                m = x <= t
                e = 0.0
                for part in (rows[m], rows[~m]):
                    q = np.sum(w[part] * lab[part]) / np.sum(w[part])
                    e += min(q, 1 - q) * np.sum(w[part])
                if best is None or e < best[0] - 1e-12:
                    best = (e, j, float((t + np.min(x[x > t])) / 2.0))
        if best is None or best[0] >= err0 - 1e-12:
            return {"v": p >= 0.5}
        _, j, t = best
        m = Q[rows, j] <= t
        return {"f": j, "t": t, "l": grow(rows[m], d - 1, used_oat or j == OATF),
                "r": grow(rows[~m], d - 1, used_oat or j == OATF)}

    node = grow(rows, depth, False)

    def pred(X):
        out = np.zeros(len(X), bool)
        for k in range(len(X)):
            n = node
            while "f" in n:
                n = n["l"] if X[k, n["f"]] <= n["t"] else n["r"]
            out[k] = bool(n["v"])
        return out
    pred.tree = node
    fidelity = float(np.mean(pred(Q) == lab.astype(bool)))
    return pred, fidelity


def _rules_of(node, path=()):
    out = [path] if path else []
    if "f" in node:
        out += _rules_of(node["l"], path + ((node["f"], node["t"], "<="),))
        out += _rules_of(node["r"], path + ((node["f"], node["t"], ">"),))
    return out


def _rule_matrix(X, rules):
    M = np.ones((len(X), len(rules)))
    for k, r in enumerate(rules):
        for j, t, op in r:
            M[:, k] *= (X[:, j] <= t) if op == "<=" else (X[:, j] > t)
    return M


def lasso(X, y, w, alpha, iters=300):
    """Weighted lasso by coordinate descent; column 0 (intercept) unpenalised."""
    beta = np.zeros(X.shape[1])
    wn = w / w.sum()
    r = y.copy()
    col = np.sum(wn[:, None] * X * X, axis=0)
    for _ in range(iters):
        for k in range(X.shape[1]):
            if col[k] == 0:
                continue
            r += X[:, k] * beta[k]
            rho = float(np.sum(wn * X[:, k] * r))
            beta[k] = rho / col[k] if k == 0 else np.sign(rho) * max(abs(rho) - alpha, 0.0) / col[k]
            r -= X[:, k] * beta[k]
    return beta


def _figs(U, idx, w, whole, max_splits=4, min_n=6):
    """FIGS (Tan et al. 2025): grow several trees together, each split chosen greedily on the current residual;
    returns the summed prediction function."""
    y = U.Y[idx]
    trees = []

    def predict_all(X):
        return sum(_leafval(t, X) for t in trees) if trees else np.zeros(len(X))

    def _leafval(node, X):
        if "f" not in node:
            return np.full(len(X), node["v"])
        out = np.zeros(len(X))
        m = X[:, node["f"]] <= node["t"]
        out[m] = _leafval(node["l"], X[m])
        out[~m] = _leafval(node["r"], X[~m])
        return out

    feats = (OATF,) if whole else (0, 1, OATF)
    for _ in range(max_splits):
        resid = y - predict_all(U.X[idx])
        best = None
        # candidate: a new stump (new tree) or a split of an existing leaf
        cands = [(None, None, np.arange(len(idx)))]
        for ti, t in enumerate(trees):
            for leaf, rows in _leaves(t, U.X[idx]):
                cands.append((ti, leaf, rows))
        for ti, leaf, rows in cands:
            if len(rows) < 2 * min_n:
                continue
            for j in feats:
                x = U.X[idx[rows], j]
                for thr in np.unique(x)[:-1]:
                    m = x <= thr
                    if m.sum() < min_n or (~m).sum() < min_n:
                        continue
                    gain = -float(np.sum(w[rows] * resid[rows])) ** 2 / float(np.sum(w[rows]))
                    for part in (rows[m], rows[~m]):
                        mu = np.sum(w[part] * resid[part]) / np.sum(w[part])
                        gain += mu * mu * np.sum(w[part])
                    if best is None or gain > best[0]:
                        best = (gain, ti, leaf, rows, j, float((thr + np.min(x[x > thr])) / 2.0))
        if best is None:
            break
        _, ti, leaf, rows, j, thr = best
        m = U.X[idx[rows], j] <= thr
        vals = [float(np.sum(w[p] * resid[p]) / np.sum(w[p])) for p in (rows[m], rows[~m])]
        new = {"f": j, "t": thr, "l": {"v": vals[0]}, "r": {"v": vals[1]}}
        if ti is None:
            trees.append(new)
        else:
            base = leaf["v"]
            leaf.clear()
            leaf.update({"f": j, "t": thr, "l": {"v": base + vals[0]}, "r": {"v": base + vals[1]}})
    return predict_all


def _leaves(node, X, rows=None):
    rows = np.arange(len(X)) if rows is None else rows
    if "f" not in node:
        return [(node, rows)]
    m = X[rows, node["f"]] <= node["t"]
    return _leaves(node["l"], X, rows[m]) + _leaves(node["r"], X, rows[~m])


_BAG_CACHE = {}


def fit_merge(kind, U, scenarios, cfg, levels, min_leaf, bags, seed):
    mask = U.mask(scenarios)
    key = (id(U), tuple(sorted(scenarios)), seed, tuple(sorted(cfg.items())), min_leaf, bags, str(levels))
    if key not in _BAG_CACHE:
        if len(_BAG_CACHE) > 64:
            _BAG_CACHE.clear()
        _BAG_CACHE[key] = fit_kind("bag", U, scenarios, cfg, levels, min_leaf, bags, seed)
    bag = _BAG_CACHE[key]
    models, info = {}, {}
    for t in range(len(U.types)):
        idx = type_idx(U, mask, t)
        if len(idx) < 8 or t not in bag.models:
            continue
        wh = U.is_whole(t)
        depth = 1 if wh else 2
        w = U.WS[U.S[idx]]
        if kind == "born":
            def decide(X, trees=bag.models[t]):
                return np.mean([walk(n, X) for n in trees], axis=0) >= 0.5
        elif kind == "rulefit":
            rules = []
            for n in bag.models[t]:
                rules += _rules_of(n)
            rules = list(dict.fromkeys(rules))
            M = np.column_stack([np.ones(len(idx)), _rule_matrix(U.X[idx], rules)])
            fold_of = inner_folds(U, idx, seed + 13)
            ys = U.Y[idx]
            sc = {}
            for alpha in (1e-4, 3e-4, 1e-3, 3e-3):
                err = 0.0
                for f in range(3):
                    tr, te = fold_of != f, fold_of == f
                    beta = lasso(M[tr], ys[tr], w[tr], alpha)
                    err += float(np.sum(w[te] * (ys[te] - M[te] @ beta) ** 2))
                sc[alpha] = err
            alpha = min(sc, key=sc.get)
            beta = lasso(M, ys, w, alpha)
            keep = [k for k in range(len(rules)) if abs(beta[k + 1]) > 0]
            info.setdefault(U.types[t], {})["rulefit"] = {"alpha": alpha, "rules_kept": len(keep),
                                                          "rules": len(rules)}

            def decide(X, beta=beta, rules=rules):
                return (np.column_stack([np.ones(len(X)), _rule_matrix(X, rules)]) @ beta) < 0.0
        else:
            f = _figs(U, idx, w, wh)

            def decide(X, f=f):
                return f(X) < 0.0
        pred, fid = _compress(U, idx, decide, depth, wh)
        models[t] = pred
        info.setdefault(U.types[t], {})["fidelity"] = round(fid, 4)
    return Fit(kind, U, models, info, cfg)


# ---------------------------------------------------------------- leave-buildings-out on the real buildings
KINDS_R4 = ("me", "tree", "bag", "reg")
KINDS_ALL = KINDS_R4 + ("tuned", "joint", "teacher", "born", "rulefit", "figs")


def lobo_real(U, real, cfg, kinds=KINDS_ALL, levels=None, folds=5, seed=0, gen=None, variants=("all", "real")):
    """Round 4's comparison: test scenarios real, training = the other folds' real + (variant "all") every generated
    scenario (or `gen`); the predicted (cost-weighted) change on the held-out real scenarios."""
    real = [s for s in real if s in U.sid]
    gen = [s for s in U.scen if s not in set(real)] if gen is None else gen
    buildings = sorted({s.split("/")[1] for s in real})
    rng = np.random.default_rng(seed)
    fold = {b: i % folds for i, b in enumerate(rng.permutation(buildings))}
    acc = {k: {v: [] for v in variants} for k in kinds}
    wsum = 0.0
    for f in range(folds):
        test = [s for s in real if fold[s.split("/")[1]] == f]
        train_real = [s for s in real if fold[s.split("/")[1]] != f]
        if not test:
            continue
        wt = float(sum(U.WS[U.sid[s]] for s in test)) if cfg["weighted"] else float(len(test))
        wsum += wt
        for label in variants:
            train = train_real + gen if label == "all" else train_real
            for kind in kinds:
                fit = fit_kind(kind, U, train, cfg, levels, seed=seed + f)
                acc[kind][label].append(score(U, fit, test, cfg) * wt)
    out = {k: {lab: float(np.sum(v)) / wsum for lab, v in d.items() if v} for k, d in acc.items()}
    ids = [U.sid[s] for s in real]
    best = [float(np.minimum(np.bincount(U.I[U.S == k], U.Y[U.S == k]), 0.0).sum()) for k in ids]
    w = U.WS[ids] if cfg["weighted"] else np.ones(len(ids))
    out["oracle"] = {"all": float(np.sum(w * np.array(best)) / np.sum(w))}
    return out


def decisions(fit, factors, active):
    """dec[i] = on-vector over GRID for factor i."""
    out = {}
    for i in active:
        f = factors[i]
        wh = f.get("whole", False)
        X = np.column_stack([np.full(len(GRID), -1.0 if wh else f["area"]), np.full(len(GRID), -1.0 if wh else
                                                                                       f["wall"]), GRID])
        out[i] = fit.on(X, np.full(len(GRID), i))
    return out
