"""Per-room outline Fourier descriptors after Zahn & Roskies (1972), "Fourier descriptors for plane closed curves",
IEEE Transactions on Computers C-21(3):269-281 (spec docs/plan/geo-reps-literature-audit-20261110.md §四).

Each room's outer boundary on the plan grid (0.2 m cells; holes filled, the largest 4-connected part) is traced
clockwise along the cell edges as a rectilinear polygon in (x, y) = (column, -row) cell units, collinear points
merged. With edge lengths dl_i (edge V_{i-1} -> V_i, i = 1..m, V_m = V_0), bends dphi_i at V_i, l_k = sum_{i<=k} dl_i
and L = l_m, the cumulative angular bend function phi*(t) = phi(Lt / 2 pi) + t has the closed-form coefficients of the
paper's eqs. 6-8:
    mu0 = -pi - (1/L) sum_k l_k dphi_k
    a_n = -(1/(n pi)) sum_k dphi_k sin(2 pi n l_k / L)
    b_n =  (1/(n pi)) sum_k dphi_k cos(2 pi n l_k / L)
and the harmonic amplitudes A_n = sqrt(a_n^2 + b_n^2) are invariant to translation, rotation, size and starting point.
reconstruct() is the paper's eq. 5 (used only to choose the number of harmonics). Used by policies/b2rl_shape.py only
when a model's room_policy.shape has "outline_harmonics"; nothing in the submission path imports this module."""
import math

import numpy as np

CELL = 0.2


def _room_cells(rid, n):
    """(rows, cols) of every room, in room order, from one pass over the grid."""
    rows, cols = np.nonzero(rid >= 0)
    ids = rid[rows, cols]
    order = np.argsort(ids, kind="stable")
    rows, cols, ids = rows[order], cols[order], ids[order]
    cuts = np.searchsorted(ids, np.arange(n + 1))
    return [(rows[cuts[i]:cuts[i + 1]], cols[cuts[i]:cuts[i + 1]]) for i in range(n)]


def _mask(r, c):
    """Cropped boolean mask of the cells (one-cell empty border), the largest 4-connected part, holes filled."""
    from scipy import ndimage as nd
    r0, c0 = r.min(), c.min()
    m = np.zeros((r.max() - r0 + 3, c.max() - c0 + 3), dtype=bool)
    m[r - r0 + 1, c - c0 + 1] = True
    lab, k = nd.label(m)                                   # 4-connectivity (cross structure)
    if k > 1:
        sizes = np.bincount(lab.ravel())[1:]
        m = lab == (1 + int(np.argmax(sizes)))
    return nd.binary_fill_holes(m), int(k)


def boundary_polygon(mask):
    """Clockwise vertices (m, 2) in (x, y) = (column, -row) of the outer boundary of a hole-free 4-connected mask
    (with an empty border), starting at the top-left corner of the top-most, then left-most cell."""
    m = mask
    up = m & ~np.roll(m, 1, axis=0)
    down = m & ~np.roll(m, -1, axis=0)
    left = m & ~np.roll(m, 1, axis=1)
    right = m & ~np.roll(m, -1, axis=1)
    nxt = {}
    # cell (r, c) corners in (x, y): TL (c, -r), TR (c+1, -r), BR (c+1, -r-1), BL (c, -r-1); interior on the right
    for r, c in zip(*np.nonzero(up)):
        nxt[(c, -r)] = (c + 1, -r)
    for r, c in zip(*np.nonzero(right)):
        nxt[(c + 1, -r)] = (c + 1, -r - 1)
    for r, c in zip(*np.nonzero(down)):
        nxt[(c + 1, -r - 1)] = (c, -r - 1)
    for r, c in zip(*np.nonzero(left)):
        nxt[(c, -r - 1)] = (c, -r)
    rs, cs = np.nonzero(m)
    top = rs.min()
    start = (int(cs[rs == top].min()), -int(top))
    pts = [start]
    p = nxt.pop(start)
    while p != start:
        pts.append(p)
        p = nxt.pop(p)
    if nxt:
        raise ValueError(f"boundary has {len(nxt)} edges outside the outer loop")
    P = np.asarray(pts, dtype=float)
    d = np.roll(P, -1, axis=0) - P                          # unit edge directions
    turn = np.any(d != np.roll(d, 1, axis=0), axis=1)       # a vertex where the direction changes
    return P[turn]


def coefficients(V, n_max):
    """Zahn-Roskies eqs. 6-8 for the clockwise polygon V (m, 2): dict with L, delta0, mu0, a, b, A, alpha
    (a, b, A, alpha indexed by n = 1..n_max)."""
    E = np.roll(V, -1, axis=0) - V                          # edge V_{i-1} -> V_i is E[i-1]
    dl = np.hypot(E[:, 0], E[:, 1])
    th = np.arctan2(E[:, 1], E[:, 0])
    dphi = np.angle(np.exp(1j * (np.roll(th, -1) - th)))   # bend at V_i between edge i and edge i+1, i = 1..m
    lk = np.cumsum(dl)                                      # l_k at V_k, k = 1..m (V_m = V_0)
    L = float(lk[-1])
    if abs(dphi.sum() + 2 * math.pi) > 1e-9:
        raise ValueError(f"not a clockwise simple curve: total bend {dphi.sum():.6f}")
    n = np.arange(1, n_max + 1)[:, None]
    arg = 2 * math.pi * n * lk[None, :] / L
    a = -(1 / (n[:, 0] * math.pi)) * (dphi[None, :] * np.sin(arg)).sum(1)
    b = (1 / (n[:, 0] * math.pi)) * (dphi[None, :] * np.cos(arg)).sum(1)
    mu0 = -math.pi - float((lk * dphi).sum()) / L
    return {"L": L, "delta0": float(th[0]), "mu0": mu0, "a": a, "b": b, "A": np.hypot(a, b),
            "alpha": np.arctan2(b, a), "start": V[0].copy()}


def reconstruct(co, N, samples=4096):
    """Paper eq. 5 with the first N harmonics: points (samples, 2) along the reconstructed curve (cell units)."""
    t = (np.arange(samples) + 0.5) * (2 * math.pi / samples)
    k = np.arange(1, N + 1)[:, None]
    phi = co["mu0"] + (co["a"][:N, None] * np.cos(k * t) + co["b"][:N, None] * np.sin(k * t)).sum(0) if N else co["mu0"]
    z = np.cumsum(np.exp(1j * (-t + co["delta0"] + phi))) * (co["L"] / samples)
    z = complex(*co["start"]) + np.concatenate([[0], z])
    return np.column_stack([z.real, z.imag])


def dense_outline(V, step=0.05):
    """Points every `step` cells along the closed polygon V."""
    out = []
    for p, q in zip(V, np.roll(V, -1, axis=0)):
        k = max(1, int(math.ceil(np.hypot(*(q - p)) / step)))
        out.append(p + (q - p) * (np.arange(k)[:, None] / k))
    return np.vstack(out)


def outline_coefficients(rid, n, n_max):
    """Per room: (coefficients dict, number of 4-connected parts) for the rooms 0..n-1 of the grid."""
    out = []
    for r, c in _room_cells(rid, n):
        m, parts = _mask(r, c)
        out.append((coefficients(boundary_polygon(m), n_max), parts))
    return out


def outline_features(rid, n, harmonics):
    """(n, harmonics) amplitudes A_1..A_harmonics of every room."""
    co = outline_coefficients(rid, n, harmonics)
    return np.vstack([c["A"][:harmonics] for c, _ in co]) if n else np.zeros((0, harmonics))


def names(harmonics):
    return tuple(f"zr{k}" for k in range(1, harmonics + 1))
