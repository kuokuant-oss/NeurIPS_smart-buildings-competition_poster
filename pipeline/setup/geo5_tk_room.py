"""Stage 戊之二 K3, the room tree by rooms-in-leaf regression (2026-11-07; spec docs/plan/tree-kinds-20261107.md;
lit-two-level §0.2 and §4 step 4; dashboard 2-1 "replace the hand-made 12 classes"). For each room option type, the
whole-building effect of switching a factor's rooms is modelled as the sum over the affected rooms of a per-room
coefficient that depends on the room's leaf in a shallow tree over room geometry and the episode outside air:
    e(s, f) = sum over rooms r of f in s of beta(leaf(x_r, oat_s))
Splits are grown greedily on one half of the buildings (cost-weighted least squares of the scenario effects on the
leaf room counts), the coefficients are estimated on the other half (weighted ridge toward the root coefficient); a
room is on when its leaf's beta + standard error < 0. The features are the deployable ones of
packages/smart_building/g36/rooms.py room_features, computed here from tree2/unit_features.json with the same
definitions.
Scores on held-out scenarios (compare): "model" = the training model's predicted change of the rooms switched on;
"observed" (conservative, comparable with the unit kinds) = the observed effect of the factor when all its rooms are on,
0 when none, and the observed effect times the share of rooms on otherwise."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo5_wu as W  # noqa: E402
import group_stage as gs  # noqa: E402

CV_M = 0.2
FEATS = ("area", "wall", "depth", "wall_per_area", "sides", "ahu_size", "ahu_ext_share", "shared_wall", "wall_n",
         "wall_s", "wall_e", "wall_w", "oat")
# 戊之三 (2026-11-10): the graph representation (the room's neighbours through shared walls), after "oat" so that every
# existing tree's feature index is unchanged; computed at run time by rooms.room_features (package patch 2)
AHU_GEOM = ("ahu_m_area", "ahu_wall_pa", "ahu_depth", "ahu_m_sides", "ahu_wn_pa", "ahu_ws_pa", "ahu_we_pa", "ahu_ww_pa",
            "ahu_m_nb_ext")
EXTRA = ("nb_n", "nb_ext", "nb_wall_pa") + AHU_GEOM
FEATS_ALL = FEATS + EXTRA


def ahu_geometry(group, area, wall, sides, wall_n, wall_s, wall_e, wall_w, nb_ext):
    """候選三修正版二 (2026-11-10): the room geometry of the whole AHU the room is on (identical for its rooms, so a
    decision on them is all-or-none per AHU): mean room area, exterior wall per floor area, perimeter depth (floor area
    / exterior wall), mean exterior sides, exterior wall per area by orientation, mean neighbours' exterior share.
    Same code as rooms.ahu_geometry (package patch 3)."""
    g = np.asarray(group).astype(int)
    k = int(g.max()) + 1
    cnt = np.bincount(g, minlength=k).astype(float)
    a_sum = np.bincount(g, weights=area, minlength=k)
    w_sum = np.bincount(g, weights=wall, minlength=k)
    per = {"ahu_m_area": a_sum / np.maximum(cnt, 1.0), "ahu_wall_pa": w_sum / np.maximum(a_sum, 1e-9),
           "ahu_depth": np.minimum(a_sum / np.maximum(w_sum, CV_M), 1000.0),
           "ahu_m_sides": np.bincount(g, weights=sides, minlength=k) / np.maximum(cnt, 1.0),
           "ahu_wn_pa": np.bincount(g, weights=wall_n, minlength=k) / np.maximum(a_sum, 1e-9),
           "ahu_ws_pa": np.bincount(g, weights=wall_s, minlength=k) / np.maximum(a_sum, 1e-9),
           "ahu_we_pa": np.bincount(g, weights=wall_e, minlength=k) / np.maximum(a_sum, 1e-9),
           "ahu_ww_pa": np.bincount(g, weights=wall_w, minlength=k) / np.maximum(a_sum, 1e-9),
           "ahu_m_nb_ext": np.bincount(g, weights=nb_ext, minlength=k) / np.maximum(cnt, 1.0)}
    return {name: v[g] for name, v in per.items()}


def neighbours(uf_b, area, wall):
    """nb_n, nb_ext, nb_wall_pa of every room from the training features' shared-wall pairs (= rooms.room_features)."""
    nums = [int(r) for r in uf_b["rooms"]]
    pos = {r: k for k, r in enumerate(nums)}
    n = len(nums)
    ext = (wall > 0).astype(float)
    wpa = wall / np.maximum(area, 1e-9)
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


def room_rows(uf_b):
    """Per-room deployable features of one building (same definitions as rooms.room_features)."""
    area, wall = np.asarray(uf_b["area"], float), np.asarray(uf_b["wall"], float)
    ahu = [str(int(a)) for a in uf_b["ahu"]]
    shared = {}
    for a, b, length in uf_b["shared"]:
        shared[int(a)] = shared.get(int(a), 0.0) + float(length)
        shared[int(b)] = shared.get(int(b), 0.0) + float(length)
    out = {"area": area, "wall": wall, "depth": np.minimum(area / np.maximum(wall, CV_M), 1000.0),
           "wall_per_area": wall / np.maximum(area, 1e-9), "sides": np.asarray(uf_b["sides"], float),
           "ahu_size": np.array([float(uf_b["ahu_rooms"][a]) for a in ahu]),
           "ahu_ext_share": np.array([float(uf_b["ahu_exterior_share"][a]) for a in ahu]),
           "shared_wall": np.array([shared.get(int(r), 0.0) for r in uf_b["rooms"]]),
           "wall_n": np.asarray(uf_b["wall_n"], float), "wall_s": np.asarray(uf_b["wall_s"], float),
           "wall_e": np.asarray(uf_b["wall_e"], float), "wall_w": np.asarray(uf_b["wall_w"], float),
           **neighbours(uf_b, area, wall)}
    out.update(ahu_geometry(uf_b["ahu"], area, wall, out["sides"], out["wall_n"], out["wall_s"], out["wall_e"],
                            out["wall_w"], out["nb_ext"]))
    return out


class RoomData:
    """Observations (scenario, factor) of the room factors with their affected rooms' feature rows."""

    def __init__(self, per, uf):
        self.obs = []                         # (scenario, factor index, effect, weight, rows array)
        for s, v in per.items():
            rr = room_rows(uf[s.split("/")[1]])
            cells = [gs.cell_of(a, w) for a, w in zip(rr["area"], rr["wall"])]
            for j, fac in enumerate(W.ROOM):
                i = W.NW + j
                e = v["eff"][i]
                if e is None:
                    continue
                want = {tuple(c) for c in fac["cells"]}
                sel = [k for k, c in enumerate(cells) if tuple(c) in want]
                if not sel:
                    continue
                X = np.column_stack([rr[f][sel] for f in FEATS[:-1]] + [np.full(len(sel), v["oat"])])
                self.obs.append((s, i, float(e), float(v["cost"]), X))
        self.types = sorted({W.FACTORS[o[1]]["type"] for o in self.obs})


def _leaf_ids(node, X):
    """Leaf id of each row (vectorised)."""
    out = np.zeros(len(X), int)

    def rec(n, rows):
        if "f" not in n:
            out[rows] = n["id"]
            return
        m = X[rows, n["f"]] <= n["t"]
        rec(n["l"], rows[m])
        rec(n["r"], rows[~m])
    rec(node, np.arange(len(X)))
    return out


def _number(node, start=0):
    if "f" not in node:
        node["id"] = start
        return start + 1
    start = _number(node["l"], start)
    return _number(node["r"], start)


_STACK = {}


def _stack(obs):
    """Stacked rows of the observations; cached for the list object itself (the cache holds it, so no id reuse)."""
    if _STACK.get("obj") is not obs:
        _STACK.update(obj=obs, val=(np.vstack([o[4] for o in obs]),
                                    np.concatenate([np.full(len(o[4]), q) for q, o in enumerate(obs)])))
    return _STACK["val"]


def _design(node, obs):
    nl = _number(node)
    X, q = _stack(obs)
    A = np.zeros((len(obs), nl))
    np.add.at(A, (q, _leaf_ids(node, X)), 1.0)
    return A


def _wls(A, y, w, ridge_to=None, lam=0.0):
    w = w / w.mean()
    M = A.T @ (A * w[:, None])
    b = A.T @ (w * y)
    if ridge_to is not None and lam > 0:
        M = M + lam * np.eye(A.shape[1])
        b = b + lam * np.full(A.shape[1], ridge_to)
    beta = np.linalg.solve(M + 1e-12 * np.eye(A.shape[1]), b)
    res = y - A @ beta
    s2 = float(np.sum(w * res ** 2) / max(1, len(y) - A.shape[1]))
    Minv = np.linalg.pinv(M)
    cov = s2 * Minv @ (A.T @ (A * (w ** 2)[:, None])) @ Minv
    return beta, cov, float(np.sum(w * res ** 2))


def _leaves(node):
    if "f" not in node:
        return [node]
    return _leaves(node["l"]) + _leaves(node["r"])


def grow(obs, depth=2, min_scen=6, nq=9, feats=range(len(FEATS))):
    """Greedy: at each step split the leaf / feature / threshold that most reduces the cost-weighted SSE of the joint
    least squares, while every leaf keeps rooms in >= min_scen observations."""
    root = {}
    y = np.array([o[2] for o in obs])
    w = np.array([o[3] for o in obs])
    allx = np.vstack([o[4] for o in obs])
    for _ in range(2 ** depth - 1):
        A = _design(root, obs)
        _, _, sse0 = _wls(A, y, w)
        best = None
        for lf in _leaves(root):
            dep = lf.get("depth", 0)
            if dep >= depth:
                continue
            for j in feats:
                vals = np.unique(np.quantile(allx[:, j], np.linspace(0.1, 0.9, nq)))
                for t in vals:
                    lf.update(f=j, t=float(t), l={"depth": dep + 1}, r={"depth": dep + 1})
                    A2 = _design(root, obs)
                    support = (A2 > 0).sum(0)
                    if support.min() >= min_scen:
                        _, _, sse = _wls(A2, y, w)
                        if best is None or sse0 - sse > best[0]:
                            best = (sse0 - sse, lf, j, float(t))
                    for k in ("f", "t", "l", "r"):
                        lf.pop(k)
        if best is None or best[0] <= 1e-15:
            break
        _, lf, j, t = best
        dep = lf.get("depth", 0)
        lf.update(f=j, t=t, l={"depth": dep + 1}, r={"depth": dep + 1})
    return root


def fit_type(obs, seed, depth=2, lam=2.0):
    """Honest: structure on half the buildings, coefficients on the other half (ridge toward the root coefficient)."""
    b = sorted({o[0].split("/")[1] for o in obs})
    half = set(np.random.default_rng(seed).permutation(b)[: max(1, len(b) // 2)].tolist())
    oa = [o for o in obs if o[0].split("/")[1] in half]
    ob = [o for o in obs if o[0].split("/")[1] not in half]
    node = grow(oa, depth) if len(oa) >= 12 else {}
    y = np.array([o[2] for o in ob])
    w = np.array([o[3] for o in ob])
    root_beta = _wls(_design({}, ob), y, w)[0][0] if ob else 0.0
    A = _design(node, ob)
    beta, cov, _ = _wls(A, y, w, root_beta, lam * len(ob) / max(1, A.shape[1]))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    support = (A > 0).sum(0)
    for lf in _leaves(node):
        k = lf["id"]
        lf.update(beta=float(beta[k]), se=float(se[k]), n=int(support[k]),
                  on=bool(support[k] >= 6 and beta[k] + se[k] < 0.0))
    return node


class RoomFit:
    def __init__(self, models):
        self.models = models                 # type -> tree

    def on_rows(self, ftype, X):
        node = self.models.get(ftype)
        if node is None:
            return np.zeros(len(X), bool)
        out = np.zeros(len(X), bool)
        for k, x in enumerate(X):
            n = node
            while "f" in n:
                n = n["l"] if x[n["f"]] <= n["t"] else n["r"]
            out[k] = n.get("on", False)
        return out

    def beta_rows(self, ftype, X):
        node = self.models.get(ftype)
        out = np.zeros(len(X))
        if node is None:
            return out
        for k, x in enumerate(X):
            n = node
            while "f" in n:
                n = n["l"] if x[n["f"]] <= n["t"] else n["r"]
            out[k] = n.get("beta", 0.0)
        return out


def fit(data, scenarios, seed=0, depth=2):
    keep = set(scenarios)
    models = {}
    for t, ft in enumerate(data.types):
        obs = [o for o in data.obs if o[0] in keep and W.FACTORS[o[1]]["type"] == ft]
        if len(obs) >= 12:
            models[ft] = fit_type(obs, seed + t, depth)
    return RoomFit(models)


def score(data, rf, test, weights):
    """(observed-anchored, model-based) cost-weighted change of the room factors on the test scenarios."""
    keep = set(test)
    obs_sum = mod_sum = 0.0
    partial = total = 0
    for s, i, e, c, X in data.obs:
        if s not in keep:
            continue
        on = rf.on_rows(W.FACTORS[i]["type"], X)
        share = float(on.mean())
        total += 1
        if 0 < share < 1:
            partial += 1
        obs_sum += c * e * share
        mod_sum += c * float(np.sum(rf.beta_rows(W.FACTORS[i]["type"], X) * on))
    wsum = sum(weights[s] for s in test)
    return obs_sum / wsum, mod_sum / wsum, partial / max(1, total)


# ---------------------------------------------------------------- deployment (correction 1 of 戊之二, 2026-11-09)
def compose_policy(bmap, rf):
    """Room policy tree: the stage base map's climate bins -> area -> wall (group_stage4._cell_tree4 layout, one
    period), then, inside each cell, every room factor covering that cell in W.ROOM order: its type's room tree
    (splits on room geometry / outside air); an "on" leaf applies the factor's delta (dict update, as
    group_stage4.compose_one). Leaves are indices into the options list."""
    options = [{}]

    def idx(opt):
        if opt not in options:
            options.append(opt)
        return options.index(opt)

    def chain(fs, opt):
        if not fs:
            return {"leaf": idx(opt)}
        f, rest = fs[0], fs[1:]
        tree = rf.models.get(f["type"])
        if not tree:
            return chain(rest, opt)

        def conv(n):
            if "f" not in n:
                return chain(rest, {**opt, **f["delta"]} if n.get("on") else dict(opt))
            return {"feature": FEATS_ALL[n["f"]], "threshold": float(n["t"]), "left": conv(n["l"]),
                    "right": conv(n["r"])}
        return conv(tree)

    def cell_node(m, cell):
        base = dict(m[g4_key(cell)])
        fs = [f for f in W.ROOM if list(cell) in f["cells"]]
        return chain(fs, base)

    def wall_node(m, a):
        return {"feature": "wall", "threshold": 0.1, "left": cell_node(m, (a, "core")),
                "right": {"feature": "wall", "threshold": 20.0, "left": cell_node(m, (a, "perim")),
                          "right": cell_node(m, (a, "long"))}}

    def bin_tree(m):
        tree = wall_node(m, gs.AREAS[-1])
        for i in range(len(gs.AREA_EDGES) - 1, -1, -1):
            tree = {"feature": "area", "threshold": gs.AREA_EDGES[i], "left": wall_node(m, gs.AREAS[i]), "right": tree}
        return tree

    trees = [bin_tree(m) for m in bmap["maps"]]
    tree = trees[-1]
    for i in range(len(bmap["oat_cuts"]) - 1, -1, -1):
        tree = {"feature": "oat", "threshold": float(bmap["oat_cuts"][i]), "left": trees[i], "right": tree}
    return {"mode": "tree", "tree": tree, "options": options, "oat_mode": "episode"}


def g4_key(cell):
    return f"{cell[0]}|{cell[1]}|0"


def intended_option(bmap, rf, rows_b, k, oat):
    """The option a room should get: base map cell option at the climate bin, plus every covering factor's delta where
    its room tree is on."""
    j = sum(oat > c for c in bmap["oat_cuts"])
    cell = gs.cell_of(rows_b["area"][k], rows_b["wall"][k])
    opt = dict(bmap["maps"][j][g4_key(cell)])
    x = np.array([[rows_b[f][k] for f in FEATS[:-1]] + [oat] + [rows_b[f][k] for f in EXTRA if f in rows_b]])
    for f in W.ROOM:
        if list(cell) in f["cells"] and rf.on_rows(f["type"], x)[0]:
            opt.update(f["delta"])
    return opt


def predicted_room(data, rf, scenarios, weights):
    """Model-based in-sample change of the room factors (cost-weighted mean over the scenarios)."""
    keep = set(scenarios)
    num = 0.0
    for s, i, e, c, X in data.obs:
        if s in keep:
            num += c * float(np.sum(rf.beta_rows(W.FACTORS[i]["type"], X) * rf.on_rows(W.FACTORS[i]["type"], X)))
    return num / sum(weights[s] for s in scenarios)
