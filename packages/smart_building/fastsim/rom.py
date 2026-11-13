"""Reduced thermal model of one scenario: Galerkin projection of the official discretisation on a rational Krylov
basis, diagonalised, cached on disk (docs/plan/fastsim.md).

The full system (core.py) is symmetric: E = diag(C, C_mass) and K = conduction + exposure + mass coupling are
symmetric positive definite, and the zone-mean outputs are the transposes of the zone-power inputs. A one-sided
Galerkin projection x = V z therefore keeps a symmetric positive definite reduced pair, so the reduced backward
Euler is unconditionally stable, and it matches the transfer moments of the shifts the basis is built at. Rows of
V are the solves (K + s E)^-1 r for the inputs r (zone powers, diffuser shares, ambient forcing) and the initial
field, at the shifts of PLAN, with a few higher moments (K + s E)^-1 E v. The shift 1/dt is the official step
matrix itself, so the first step responses of the reduced model are those of the official one; 0 fixes the
steady state; the shifts between cover the walls (about an hour), the slab (hours) and the facade (a day).

The interior mass node of every occupied CV has the same capacity and coupling (one material, one CV size), so
the mass part of every Krylov vector is a masked combination of the field parts of its own shift; only the field
parts are stored (V_T) with the combination matrix Xi, which halves the memory of the build.

In modal coordinates (K_r phi = lambda E_r phi, phi' E_r phi = 1, E in units of E/dt) one step is
    w' = (w + Phi' B u + Phi' P q) / (1 + lambda),
diagonal; the implicit VAV coupling is solved exactly as the official _add_air_side does, with the reduced
response H = S Phi diag(1 / (1 + lambda)) Phi' P.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import gc
import hashlib
import logging
import os
from pathlib import Path
import time

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/runs/fastsim/rom-cache"
VERSION = 1
# (shift x dt, moments, inputs): "fast" = zone powers, diffuser shares and ambient; "slow" = zone powers and ambient
PLAN = ((1.0, 3, "fast"), (0.1, 2, "slow"), (0.01, 2, "slow"), (0.0, 2, "slow"))
TOLERANCE = 1e-12
STORAGE_BYTES = 2.0e9
# a larger basis goes to a scratch file; SB_FASTSIM_IN_MEMORY_BYTES raises the limit for a builder whose memory
# a scheduler has set aside (the file's dirty pages are written to disk for nothing and throttle the build)
IN_MEMORY_BYTES = float(os.environ.get("SB_FASTSIM_IN_MEMORY_BYTES", "1e9"))
SCRATCH = ROOT / ".tmp/fastsim-build"
# threads for the triangular solves of a build (SuperLU releases the GIL and solves each column on its own, so the
# columns can be split without changing the result); a builder's BLAS thread count unless set
SOLVE_THREADS = max(1, int(os.environ.get("SB_FASTSIM_SOLVE_THREADS") or os.environ.get("OPENBLAS_NUM_THREADS")
                           or 1))


def _solve(factor, rhs):
    """factor.solve(rhs) with the columns of rhs split over SOLVE_THREADS threads."""
    rhs = np.asarray(rhs, dtype=np.float64)
    if SOLVE_THREADS <= 1 or rhs.ndim == 1 or rhs.shape[1] < 2 * SOLVE_THREADS:
        return factor.solve(rhs)
    parts = np.array_split(np.arange(rhs.shape[1]), SOLVE_THREADS)
    with ThreadPoolExecutor(SOLVE_THREADS) as pool:
        return np.hstack(list(pool.map(lambda c: factor.solve(rhs[:, c]), parts)))
_ROWS = 32768
_COLUMNS = 32


@dataclass
class RomModel:
    h: float
    dt: float
    lam: np.ndarray            # (r,) generalized eigenvalues of (K, E/dt)
    zone_input: np.ndarray     # (r, n_zones) Phi' B for zone powers in W
    ambient_input: np.ndarray  # (r,) Phi' (dirichlet + exposure)
    applied: np.ndarray        # (r, n_vav) Phi' P
    zone: np.ndarray           # (n_zones, r) zone means
    sensed: np.ndarray         # (n_vav, r) VAV sensed means
    field: np.ndarray          # (r,) sum over solved CVs / all cells
    w0: np.ndarray             # (r,) the reset field
    info: dict = field(default_factory=dict)

    @property
    def decay(self):
        return 1.0 / (1.0 + self.lam)

    @property
    def beta(self):
        return 1.0 / (1.0 + self.lam)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, h=self.h, dt=self.dt, lam=self.lam, zone_input=self.zone_input,
                 ambient_input=self.ambient_input, applied=self.applied, zone=self.zone, sensed=self.sensed,
                 field=self.field, w0=self.w0, info=np.array(repr(self.info)))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(h=float(z["h"]), dt=float(z["dt"]), lam=z["lam"], zone_input=z["zone_input"],
                       ambient_input=z["ambient_input"], applied=z["applied"], zone=z["zone"],
                       sensed=z["sensed"], field=z["field"], w0=z["w0"], info={"cached": str(path)})


def mass_parameters(data):
    """(capacity rate C_m/dt, coupling G) of the interior mass nodes, which must be uniform over the mask."""
    mask = data.mass
    if not mask.any():
        return 0.0, 0.0
    capacity_rate = data.mass_coupling[mask] / data.mass_gained[mask]
    coupling = data.mass_coupling[mask] / data.mass_retained[mask]
    for name, values in (("capacity", capacity_rate), ("coupling", coupling)):
        if np.ptp(values) > 1e-9 * abs(values.mean()):
            raise ValueError(f"interior mass {name} is not uniform; the reduced model needs one mass material")
    return float(capacity_rate.mean()), float(coupling.mean())


def raw_count(data, plan=PLAN):
    """Number of raw basis vectors: the reset field, its free step, and moments x inputs per shift."""
    inputs = {"fast": len(data.zones) + data.n_vav + 1, "slow": len(data.zones) + 1}
    return 2 + sum(moments * inputs[kind] for _, moments, kind in plan)


def storage_dtype(data, plan=PLAN):
    """float64 basis storage up to STORAGE_BYTES, float32 above (the largest buildings; every product with the
    basis is still accumulated in float64, so this only rounds the basis, which stays a basis)."""
    return np.float64 if data.n * raw_count(data, plan) * 8 <= STORAGE_BYTES else np.float32


def cache_key(data, h, plan=PLAN, tolerance=TOLERANCE):
    digest = hashlib.sha256()
    digest.update(repr((VERSION, plan, tolerance, float(h), data.dt, data.shape, data.n)).encode())
    if storage_dtype(data, plan) is np.float32:
        digest.update(b"float32")
    conduction = data.conduction.tocsc()
    for array in (conduction.data, conduction.indices, conduction.indptr, data.capacitance_rate,
                  data.mass_coupling, data.mass_retained, data.mass_gained, data.dirichlet, data.exposure(h),
                  data.applied.data, data.applied.indices, data.applied.indptr, data.sensed.data,
                  data.sensed.indices, data.sensed.indptr, data.zone_mean.data, data.zone_mean.indices,
                  data.zone_mean.indptr, data.initial_temp):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()[:24]


def _weighted_gram(V, weight):
    """V' diag(weight) V in row chunks (float64 accumulation)."""
    out = np.zeros((V.shape[1], V.shape[1]))
    for start in range(0, V.shape[0], _ROWS):
        block = np.asarray(V[start:start + _ROWS], dtype=np.float64)
        out += (block * weight[start:start + _ROWS, np.newaxis]).T @ block
    return 0.5 * (out + out.T)


def _transpose_product(V, M):
    """V' M in float64 for M (n x k) dense or sparse, in row chunks."""
    sparse = scipy.sparse.issparse(M)
    M = M.tocsr() if sparse else np.asarray(M, dtype=np.float64)
    out = np.zeros((V.shape[1], M.shape[1]))
    for start in range(0, V.shape[0], _ROWS):
        block = np.asarray(V[start:start + _ROWS], dtype=np.float64)
        part = M[start:start + _ROWS]
        out += (part.T @ block).T if sparse else block.T @ part
    return out


def _columns_product(V, columns, coefficients):
    """V[:, columns] @ coefficients in float64, in row chunks."""
    out = np.empty((V.shape[0], coefficients.shape[1]))
    for start in range(0, V.shape[0], _ROWS):
        out[start:start + _ROWS] = np.asarray(V[start:start + _ROWS, columns], dtype=np.float64) @ coefficients
    return out


def _stiffness_gram(V, stiffness):
    """V' K V for a sparse symmetric K whose rows reach a narrow band of columns, in row chunks."""
    K = stiffness.tocsr()
    out = np.zeros((V.shape[1], V.shape[1]))
    for start in range(0, V.shape[0], _ROWS):
        rows = K[start:start + _ROWS]
        if rows.nnz == 0:
            continue
        lo, hi = int(rows.indices.min()), int(rows.indices.max()) + 1
        product = rows[:, lo:hi] @ np.asarray(V[lo:hi], dtype=np.float64)
        out += np.asarray(V[start:start + _ROWS], dtype=np.float64).T @ product
    return 0.5 * (out + out.T)


def build(data, h, plan=PLAN, tolerance=TOLERANCE, log=None):
    """The reduced model of data (core.ThermalData) at convection coefficient h."""
    started = time.time()
    say = log or (lambda message: None)
    n = data.n
    capacity_rate, coupling = mass_parameters(data)
    mask = data.mass.astype(np.float64)
    exposure = data.exposure(h)
    # K without the step's capacitance: conduction (with the dirichlet diagonal) + exposure
    stiffness = (data.conduction - scipy.sparse.diags(data.capacitance_rate + data.mass_coupling)
                 + scipy.sparse.diags(exposure)).tocsc()
    zone_rhs = data.zone_mean.T.tocsc()                       # B for zone powers: 1/n_zone on the zone's CVs
    ambient_rhs = (data.dirichlet + exposure)[:, np.newaxis]
    initial = np.asarray(data.initial_temp, dtype=np.float64).reshape(-1)[data.solved]
    inputs = {"fast": scipy.sparse.hstack([zone_rhs, data.applied, scipy.sparse.csc_matrix(ambient_rhs)]).tocsc(),
              "slow": scipy.sparse.hstack([zone_rhs, scipy.sparse.csc_matrix(ambient_rhs)]).tocsc()}
    total = raw_count(data, plan)
    storage = storage_dtype(data, plan)
    scratch = None
    if n * total * np.dtype(storage).itemsize > IN_MEMORY_BYTES:
        # a large basis lives in a scratch file inside the project (the page cache keeps what fits), so a builder
        # needs little more memory than the simulator itself; the arithmetic is the same
        folder = Path(os.environ.get("SB_FASTSIM_SCRATCH") or SCRATCH)
        folder.mkdir(parents=True, exist_ok=True)
        scratch = folder / f"basis-{os.getpid()}-{time.time_ns()}.dat"
        V = np.memmap(scratch, dtype=storage, mode="w+", shape=(n, total), order="F")
    else:
        V = np.zeros((n, total), dtype=storage, order="F")
    try:
        return _assemble(data, h, plan, tolerance, say, started, V, total, storage, capacity_rate, coupling, mask,
                         exposure, stiffness, ambient_rhs, initial, inputs)
    finally:
        if scratch is not None:
            V = None
            gc.collect()
            try:
                scratch.unlink()
            except OSError:
                pass


def _assemble(data, h, plan, tolerance, say, started, V, total, storage, capacity_rate, coupling, mask, exposure,
              stiffness, ambient_rhs, initial, inputs):
    """build() after the basis storage V (n x total) is allocated."""
    n, dt = data.n, data.dt
    Xi = np.zeros((total, total))
    used = 0

    def add(vector, mass_combination):
        nonlocal used
        V[:, used] = vector
        Xi[:, used] = mass_combination
        used += 1
        return used - 1

    def mass_of(columns):
        """Mass parts mask * (V Xi[:, columns]) through the nonzero rows of Xi only."""
        rows = np.flatnonzero(np.any(Xi[:, columns] != 0.0, axis=1))
        return mask[:, np.newaxis] * _columns_product(V, rows, Xi[np.ix_(rows, columns)])

    def next_moment(factor, sigma, gamma, columns):
        """(K + s E)^-1 (E/dt) v for the stored vectors v = columns; returns their indices.

        With v = (T, m): T' = A(s)^-1 (C/dt T + gamma C_m/dt m) and m' = (C_m/dt m + G T') / (s C_m/dt + G).
        Stored as T'/scale, m' is a combination of the chain's vectors: Xi[:, new] = c Xi[:, v] / scale with
        c = (C_m/dt) / (s C_m/dt + G), and gamma on the new vector itself."""
        block = _solve(factor, data.capacitance_rate[:, np.newaxis] * np.asarray(V[:, columns], dtype=np.float64)
                       + gamma * capacity_rate * mass_of(columns))
        carried = capacity_rate / (sigma * capacity_rate + coupling) if coupling > 0 else 0.0
        out = []
        for j, column in zip(columns, block.T):
            scale = np.abs(column).max()
            combination = carried * Xi[:, j] / scale
            combination[used] = gamma
            out.append(add(column / scale, combination))
        return out

    # the reset field (T and the mass nodes at the same temperature)
    x0_index = add(initial, np.zeros(total))
    Xi[x0_index, x0_index] = 1.0
    for sigma, moments, kind in plan:
        t = time.time()
        gamma = coupling / (sigma * capacity_rate + coupling) if coupling > 0 else 0.0
        mass_shift = (sigma * capacity_rate * coupling / (sigma * capacity_rate + coupling)) if coupling > 0 else 0.0
        shifted = (stiffness + scipy.sparse.diags(sigma * data.capacitance_rate + mask * mass_shift)).tocsc()
        factor = scipy.sparse.linalg.splu(shifted, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                          options={"SymmetricMode": True})
        rhs = inputs[kind]
        previous = []
        for start in range(0, rhs.shape[1], _COLUMNS):
            block = _solve(factor, rhs[:, start:start + _COLUMNS].toarray())
            for column in block.T:
                combination = np.zeros(total)
                combination[used] = gamma
                previous.append(add(column / np.abs(column).max(), combination))
        if sigma == 1.0:   # one free step of the reset field
            next_moment(factor, sigma, gamma, [x0_index])
        for _ in range(moments - 1):
            following = []
            for start in range(0, len(previous), _COLUMNS):
                following += next_moment(factor, sigma, gamma, previous[start:start + _COLUMNS])
            previous = following
        del factor
        say(f"shift {sigma}: {moments} moments x {rhs.shape[1]} inputs, {time.time() - t:.1f}s")
    V = V[:, :used]
    Xi = Xi[:used, :used]
    # Gram matrices of the stored field parts
    t = time.time()
    g_capacity = _weighted_gram(V, data.capacitance_rate)
    g_mask = _weighted_gram(V, mask)
    g_stiff = _stiffness_gram(V, stiffness)
    say(f"gram matrices {used}x{used} ({np.dtype(storage).name} basis): {time.time() - t:.1f}s")
    identity = np.eye(used)
    E = g_capacity + capacity_rate * Xi.T @ g_mask @ Xi
    K = g_stiff + coupling * (identity - Xi).T @ g_mask @ (identity - Xi)
    scale = 1.0 / np.sqrt(np.diag(E))
    E_n = E * scale[:, np.newaxis] * scale[np.newaxis, :]
    s, u = np.linalg.eigh(0.5 * (E_n + E_n.T))
    keep = s > tolerance * s.max()
    basis = (u[:, keep] / np.sqrt(s[keep])) * scale[:, np.newaxis]     # raw coefficients of an E-orthonormal basis
    K_b = basis.T @ K @ basis
    lam, q = np.linalg.eigh(0.5 * (K_b + K_b.T))
    coeff = basis @ q                                                    # raw coefficients of the modes
    zone_out = _transpose_product(V, data.zone_mean.T).T @ coeff
    sensed = _transpose_product(V, data.sensed.T).T @ coeff
    applied = coeff.T @ _transpose_product(V, data.applied)
    ambient_input = coeff.T @ _transpose_product(V, ambient_rhs)[:, 0]
    field_row = coeff.T @ _transpose_product(V, np.ones((n, 1)))[:, 0] / data.n_cells
    initial_parts = _transpose_product(V, np.stack([data.capacitance_rate * initial, mask * initial], axis=1))
    e_x0 = initial_parts[:, 0] + capacity_rate * Xi.T @ initial_parts[:, 1]
    w0 = coeff.T @ e_x0
    info = {"raw": used, "kept": int(keep.sum()), "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
            "seconds": round(time.time() - started, 1), "plan": plan, "tolerance": tolerance,
            "storage": np.dtype(storage).name}
    say(f"reduced model: {used} raw vectors, {int(keep.sum())} kept, lambda {lam.min():.3g}..{lam.max():.3g}, "
        f"{time.time() - started:.0f}s")
    return RomModel(h=float(h), dt=dt, lam=lam, zone_input=zone_out.T.copy(), ambient_input=ambient_input,
                    applied=applied, zone=zone_out, sensed=sensed, field=field_row, w0=w0, info=info)


def cached(data, h, plan=PLAN, tolerance=TOLERANCE, cache=None, log=None):
    """The reduced model from the cache, built and stored on a miss."""
    folder = Path(cache or os.environ.get("SB_FASTSIM_CACHE") or CACHE)
    path = folder / f"rom-{cache_key(data, h, plan, tolerance)}.npz"
    if path.is_file():
        return RomModel.load(path)
    if os.environ.get("SB_FASTSIM_NO_BUILD") == "1":      # 2026-11-11: big-building builds crashed WSL (> 17 GB)
        raise RuntimeError(f"fastsim: no cached reduced model {path.name} and SB_FASTSIM_NO_BUILD=1")
    logging.getLogger(__name__).warning("fastsim: no cached reduced model %s; building it (once per scenario, about "
                                        "0.5-10 minutes and 1-5 GB)", path.name)
    model = build(data, h, plan, tolerance, log=log)
    model.save(path)
    model.info["cached"] = str(path)
    return model
