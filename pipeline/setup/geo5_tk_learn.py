"""Stage 戊之二 tree kinds (2026-11-07; spec docs/plan/tree-kinds-20261107.md): the kinds the notes planned that stage
戊 did not run, on the same (scenario, factor) units as geo5_wu_learn (cost-weighted, honest halves, building-wide
types climate only and depth <= 1), plus the leave-one-climate-out check.
- linleaf (K1, model-based recursive partitioning / local linear, lit-ml-trees C10, C11): honest tree, every leaf's
  effect = a + b x outside air (weighted least squares on the estimation half, ridge on the slope); a unit is on where
  the leaf's prediction + its standard error < 0;
- reem (K2, building random effects, RE-EM style; dashboard 2-2 GPBoost class, lit-two-level §0.6): alternate the
  honest tree on (effect - building intercept) and shrunk building intercepts; decisions from the tree (population);
- rf / et / gb (K4, strong teacher distilled VIPER-style, lit-ml-trees D1 and §8.6): a random forest, extremely
  randomised trees or gradient boosting (scikit-learn) on the unit's deployable features + the stage's context and
  AHU / neighbour features; queried on the units and on each unit's factor over the climate grid; the student is a
  depth <= 2 (building-wide: 1) tree on the deployable features fitted to the teacher's predictions with regret weights
  (cost x |predicted effect|), a leaf on when its weighted mean prediction < 0.
Every other kind is geo5_wu_learn's, unchanged.
Deployable features stay (area band, wall band, episode running-mean outside air)."""
import copy

import numpy as np

import geo5_wu_learn as WL

OATF = WL.OATF
KINDS_NEW = ("linleaf", "reem", "rf", "et", "gb")
KINDS_ALL = WL.KINDS_ALL + KINDS_NEW


def fit_kind(kind, U, scenarios, cfg, levels=None, min_leaf=6, bags=50, seed=0):
    if kind == "linleaf":
        return fit_linleaf(U, scenarios, cfg, levels, min_leaf, seed)
    if kind == "reem":
        return fit_reem(U, scenarios, cfg, levels, min_leaf, seed)
    if kind in ("rf", "et", "gb"):
        return fit_forest(kind, U, scenarios, cfg, seed)
    return WL.fit_kind(kind, U, scenarios, cfg, levels, min_leaf, bags, seed)


# ---------------------------------------------------------------- K1 leaf-linear honest tree
def _o(oat):
    return (np.asarray(oat, float) - 15.0) / 12.0


def _lin_fill(node, U, idx, cfg, min_leaf, slope_ridge):
    """Leaves: weighted least squares of the scenario totals on (1, o); stores a, b, covariance, n."""
    if "f" in node:
        m = U.X[idx, node["f"]] <= node["t"]
        _lin_fill(node["l"], U, idx[m], cfg, min_leaf, slope_ridge)
        _lin_fill(node["r"], U, idx[~m], cfg, min_leaf, slope_ridge)
        return
    uniq, tot = WL.scen_totals(U, idx, cfg["agg"]) if len(idx) else (np.array([], int), np.array([]))
    n = len(uniq)
    node["n"] = n
    if n < max(3, min_leaf):
        node.update(a=0.0, b=0.0, cov=np.zeros((2, 2)), ok=False)
        return
    w = U.WS[uniq] if cfg["weighted"] else np.ones(n)
    w = w / w.mean()
    X = np.column_stack([np.ones(n), _o(U.OAT[uniq])])
    pen = np.diag([0.0, slope_ridge * n])
    A = X.T @ (X * w[:, None]) + pen
    beta = np.linalg.solve(A, X.T @ (w * tot))
    res = tot - X @ beta
    s2 = float(np.sum(w * res ** 2) / max(1, n - 2))
    Ainv = np.linalg.inv(A)
    cov = s2 * Ainv @ (X.T @ (X * (w ** 2)[:, None])) @ Ainv
    node.update(a=float(beta[0]), b=float(beta[1]), cov=cov, ok=True)


def _leaf(node, x):
    while "f" in node:
        node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
    return node


def _lin_predicate(node):
    def on(X):
        out = np.zeros(len(X), bool)
        for k, x in enumerate(np.asarray(X, float)):
            lf = _leaf(node, x)
            if not lf.get("ok"):
                continue
            v = np.array([1.0, float(_o(x[OATF]))])
            pred = lf["a"] + lf["b"] * v[1]
            se = float(np.sqrt(max(v @ lf["cov"] @ v, 0.0)))
            out[k] = pred + se < 0.0
        return out
    on.tree = node
    return on


def fit_linleaf(U, scenarios, cfg, levels, min_leaf, seed, slope_ridge=0.05):
    levels = levels or WL.g4.DEFAULT_LEVELS
    mask = U.mask(scenarios)
    models = {}
    for t in range(len(U.types)):
        idx = WL.type_idx(U, mask, t)
        if len(idx) < 4:
            continue
        wh = U.is_whole(t)
        lv = [[OATF]] if wh else levels
        b = np.unique(U.B[idx])
        rng = np.random.default_rng(seed + t)
        half = set(rng.permutation(b)[: max(1, len(b) // 2)].tolist())
        in_a = np.array([x in half for x in U.B[idx]])
        node = WL._grow(U, idx[in_a], lv, max(3, min_leaf // 2), cfg, whole=wh)
        _lin_fill(node, U, idx[~in_a], cfg, min_leaf, slope_ridge)
        models[t] = _lin_predicate(node)
    return WL.Fit("linleaf", U, models, {}, cfg)


# ---------------------------------------------------------------- K2 building random effects (RE-EM)
def fit_reem(U, scenarios, cfg, levels, min_leaf, seed, iters=4, lam_b=3.0):
    levels = levels or WL.g4.DEFAULT_LEVELS
    mask = U.mask(scenarios)
    models, info = {}, {}
    for t in range(len(U.types)):
        idx = WL.type_idx(U, mask, t)
        if len(idx) < 4:
            continue
        wh = U.is_whole(t)
        lv = [[OATF]] if wh else levels
        u = {}
        Uc = copy.copy(U)
        w = U.WS[U.S[idx]] if cfg["weighted"] else np.ones(len(idx))
        for _ in range(iters):
            Y = U.Y.copy()
            Y[idx] = U.Y[idx] - np.array([u.get(int(b), 0.0) for b in U.B[idx]])
            Uc.Y = Y
            node = WL.honest_tree(Uc, idx, min_leaf, lv, seed + t, cfg, wh)
            # per-unit fitted value: the leaf's scenario-total mean spread over the scenario's units in that leaf
            leafv = WL.walk(node, U.X[idx], "shrunk")
            lid = np.array([id(_leaf(node, x)) for x in U.X[idx]])
            key = np.array([f"{int(s)}:{int(i)}" for s, i in zip(U.S[idx], lid)])
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            fitted = leafv / cnt[inv]
            r = U.Y[idx] - fitted
            u = {}
            for b in np.unique(U.B[idx]):
                m = U.B[idx] == b
                u[int(b)] = float(np.sum(w[m] * r[m]) / (np.sum(w[m]) + lam_b * w.mean()))
        models[t] = node
        info[U.types[t]] = {"intercept_sd": float(np.std(list(u.values())))}
    return WL.Fit("reem", U, models, info, cfg)


# ---------------------------------------------------------------- K4 strong teacher, VIPER-style distillation
Z_KEYS = ("has_min", "base_ratio", "base_cool", "base_heat", "ahu_rooms", "ahu_ext", "nb_same", "depth", "wall_ns",
          "wall_ew", "nb_count", "nb_wall", "nb_on")


def _teacher_X(U, rows, X=None):
    X = U.X[rows] if X is None else X
    cols = [X[:, 0], X[:, 1], X[:, OATF]]
    cols += [U.Z[k][rows] for k in Z_KEYS if k in U.Z]
    return np.column_stack(cols)


def _model(kind, seed):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=seed,
                                     n_jobs=1)
    if kind == "et":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=seed, n_jobs=1)
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10,
                                         l2_regularization=1.0, random_state=seed)


def _distil_rows(Xq, target, w, depth, whole, min_n=6):
    """Greedy weighted regression tree on the deployable columns (0 area, 1 wall, 2 oat) of the query rows."""
    def grow(rows, d, used_oat):
        mu = float(np.sum(w[rows] * target[rows]) / np.sum(w[rows]))
        if d == 0 or len(rows) < 2 * min_n:
            return {"v": mu}
        tot = float(np.sum(w[rows] * (target[rows] - mu) ** 2))
        best = None
        for j in ((OATF,) if whole else (0, 1, OATF)):
            if j == OATF and used_oat:
                continue
            x = Xq[rows, j]
            vals = np.unique(x)
            if len(vals) > 64:
                vals = np.unique(np.quantile(x, np.linspace(0.02, 0.98, 49)))
            for t in vals[:-1]:
                m = x <= t
                if m.sum() < min_n or (~m).sum() < min_n:
                    continue
                sse = 0.0
                for part in (rows[m], rows[~m]):
                    pm = np.sum(w[part] * target[part]) / np.sum(w[part])
                    sse += np.sum(w[part] * (target[part] - pm) ** 2)
                if best is None or tot - sse > best[0]:
                    nxt = np.min(x[x > t])
                    best = (tot - sse, j, float((t + nxt) / 2.0))
        if best is None or best[0] <= 1e-15:
            return {"v": mu}
        _, j, t = best
        m = Xq[rows, j] <= t
        return {"f": j, "t": t, "l": grow(rows[m], d - 1, used_oat or j == OATF),
                "r": grow(rows[~m], d - 1, used_oat or j == OATF)}

    node = grow(np.arange(len(Xq)), depth, False)

    def on(X):
        return np.array([_leaf(node, x)["v"] < 0.0 for x in np.asarray(X, float)], bool)
    on.tree = node
    return on


def fit_forest(kind, U, scenarios, cfg, seed):
    mask = U.mask(scenarios)
    models, info = {}, {}
    for t in range(len(U.types)):
        idx = WL.type_idx(U, mask, t)
        if len(idx) < 8:
            continue
        wh = U.is_whole(t)
        w = U.WS[U.S[idx]] if cfg["weighted"] else np.ones(len(idx))
        Xt = U.X[idx][:, [OATF]] if wh else _teacher_X(U, idx)
        model = _model(kind, seed + t)
        model.fit(Xt, U.Y[idx], sample_weight=w / w.mean())
        # query points: the units, and every unit with its climate replaced by a grid over the observed range
        lo, hi = np.percentile(U.X[idx, OATF], [1, 99])
        grid = np.arange(np.floor(lo), np.ceil(hi) + 0.01, 1.0)
        reps = [U.X[idx].copy()]
        for g in grid:
            Xg = U.X[idx].copy()
            Xg[:, OATF] = g
            reps.append(Xg)
        Xq = np.vstack(reps)
        rows = np.tile(idx, len(reps))
        if wh:
            pred = model.predict(Xq[:, [OATF]])
        else:
            pred = model.predict(_teacher_X(U, rows, Xq))
        # review fix 2026-11-10: cost weights only, so a leaf mean is the cost-weighted mean effect (its sign = the
        # decision); weighting by |pred| as well made the sign that of sum(cost * pred * |pred|)
        wq = np.tile(w, len(reps))
        wq = np.where(wq > 0, wq, 1e-12)
        models[t] = _distil_rows(Xq, pred, wq, 1 if wh else 2, wh)
        info[U.types[t]] = {"teacher": kind, "query_rows": int(len(Xq))}
    return WL.Fit(kind, U, models, info, cfg)


# ---------------------------------------------------------------- validation: leave buildings out, leave a climate out
def _station(s):
    return s.split("/")[2].rstrip("_0123456789")


def compare(U, real, cfg, kinds=KINDS_ALL, levels=None, folds=5, seed=0):
    """Leave-buildings-out exactly as geo5_wu_learn.lobo_real (variant "all") and leave-one-climate-out: each station
    in turn, test = its real scenarios, train = every scenario (real and generated) of the other stations."""
    real = [s for s in real if s in U.sid]
    gen = [s for s in U.scen if s not in set(real)]
    buildings = sorted({s.split("/")[1] for s in real})
    rng = np.random.default_rng(seed)
    fold = {b: i % folds for i, b in enumerate(rng.permutation(buildings))}
    out = {k: {} for k in kinds}
    acc = {k: 0.0 for k in kinds}
    wsum = 0.0
    for f in range(folds):
        test = [s for s in real if fold[s.split("/")[1]] == f]
        train = [s for s in real if fold[s.split("/")[1]] != f] + gen
        if not test:
            continue
        wt = float(sum(U.WS[U.sid[s]] for s in test))
        wsum += wt
        for k in kinds:
            acc[k] += WL.score(U, fit_kind(k, U, train, cfg, levels, seed=seed + f), test, cfg) * wt
    for k in kinds:
        out[k]["lobo"] = acc[k] / wsum
    acc = {k: 0.0 for k in kinds}
    wsum = 0.0
    per_station = {k: {} for k in kinds}
    for stn in sorted({_station(s) for s in real}):
        test = [s for s in real if _station(s) == stn]
        train = [s for s in U.scen if _station(s) != stn]
        wt = float(sum(U.WS[U.sid[s]] for s in test))
        wsum += wt
        for k in kinds:
            v = WL.score(U, fit_kind(k, U, train, cfg, levels, seed=seed), test, cfg)
            per_station[k][stn] = v
            acc[k] += v * wt
    for k in kinds:
        out[k]["loco"] = acc[k] / wsum
        out[k]["loco_by_station"] = per_station[k]
    return out
