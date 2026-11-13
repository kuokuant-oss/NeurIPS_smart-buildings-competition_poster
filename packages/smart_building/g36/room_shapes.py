"""Per-room shape features for the geometry representation study in closed loop (2026-11-10, spec
docs/plan/geo-reps-closed-loop-20261110.md): the polygon and raster representations of
.tmp/setup/analysis/reps.py, computed at run time from the floor plan the controller sees (building_utils rooms in
BuildingIndex.vav_ids order and the exterior-wall shell, as rooms.room_features). Used only by
policies/b2rl_shape.py when a model's room_policy carries a "shape" block; nothing else imports this module.

polygon  pg_scale (mean centroid distance, m), pg_fd1..pg_fd6 (magnitudes of the Fourier coefficients 1..6 of the
         centroid-distance function r(theta) at 32 angles, relative to the mean; rotation invariant), pg_ex0..pg_ex3
         (magnitudes 0..3 of the Fourier series of the exterior-wall indicator along theta);
raster   rpc1..rpcK: a 48 x 48 window of 0.6 m cells centred on the room, three channels (this room, other rooms,
         exterior shell) as bits, projected on the model's principal components (shape["pca_mean"],
         shape["pca_components"], fitted on the training buildings only)."""
import numpy as np

CELL = 0.2
WIN, RES = 48, 3
POLY_NAMES = ("pg_scale",) + tuple(f"pg_fd{k}" for k in range(1, 7)) + tuple(f"pg_ex{k}" for k in range(4))


def _shift(a, dr, dc):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(dr, 0):h + min(dr, 0), max(dc, 0):w + min(dc, 0)] = a[max(-dr, 0):h + min(-dr, 0), max(-dc, 0):w + min(-dc, 0)]
    return out


def grid(metadata, index):
    """room_id (zone index in index.vav_ids order, -1 elsewhere) and the exterior-wall shell mask."""
    from smart_buildings.smart_control.simulator import building_utils, constants
    rooms, exterior_walls, _, _, _ = building_utils.construct_building_data_types(
        floor_plan=np.asarray(metadata["floor_plan"]), zone_map=np.asarray(metadata["zone_map"]))
    room_id = np.full(np.shape(exterior_walls), -1, dtype=np.int32)
    for z, vav in enumerate(index.vav_ids):
        cells = np.asarray(rooms[vav.replace("VAV_", "room_")])
        room_id[cells[:, 0], cells[:, 1]] = z
    return room_id, np.asarray(exterior_walls) == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION


def shapes_from_grid(rid, shell, n):
    """(poly n x 11, raster bits n x 3 x 48 x 48) exactly as .tmp/setup/analysis/reps.py building()."""
    rid = rid.astype(np.int32)
    shell = shell.astype(bool)
    rows, cols = np.nonzero(rid >= 0)
    ids = rid[rows, cols]
    cnt = np.maximum(np.bincount(ids, minlength=n), 1)
    cr = np.bincount(ids, weights=rows, minlength=n) / cnt
    cc = np.bincount(ids, weights=cols, minlength=n) / cnt
    nb = 32
    ang = np.arctan2(-(rows - cr[ids]), cols - cc[ids])
    rad = np.hypot(rows - cr[ids], cols - cc[ids]) * CELL
    b = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    rmaxb = np.zeros((n, nb))
    np.maximum.at(rmaxb, (ids, b), rad)
    extb = np.zeros((n, nb))
    shell_nb = np.zeros_like(shell)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shell_nb |= _shift(shell, dr, dc)
    em = shell_nb[rows, cols]
    np.maximum.at(extb, (ids[em], b[em]), 1.0)
    for i in range(n):                                # empty bins (thin rooms): carry the previous radius
        for j in range(nb):
            if rmaxb[i, j] == 0:
                rmaxb[i, j] = rmaxb[i, j - 1]
    mean_r = rmaxb.mean(1)
    fr = np.abs(np.fft.rfft(rmaxb / np.maximum(mean_r[:, None], 1e-9), axis=1))[:, 1:7] / nb
    fe = np.abs(np.fft.rfft(extb, axis=1))[:, 0:4] / nb
    poly = np.column_stack([mean_r, fr, fe])
    h, w = rid.shape
    half = WIN * RES // 2
    pad = np.full((h + 2 * half, w + 2 * half), -2, dtype=np.int32)
    pad[half:half + h, half:half + w] = np.where(shell, -3, rid)
    raster = np.zeros((n, 3, WIN, WIN), dtype=bool)
    for i in range(n):
        r0, c0 = int(round(cr[i])), int(round(cc[i]))
        win = pad[r0:r0 + 2 * half, c0:c0 + 2 * half].reshape(WIN, RES, WIN, RES)
        raster[i, 0] = (win == i).mean(axis=(1, 3)) >= 0.5
        raster[i, 1] = ((win >= 0) & (win != i)).mean(axis=(1, 3)) >= 0.5
        raster[i, 2] = (win == -3).any(axis=(1, 3))
    return poly, raster


def named(poly, raster, shape):
    """Feature dict: the polygon columns and the raster principal-component scores of the shape block."""
    out = {name: poly[:, k].astype(float) for k, name in enumerate(POLY_NAMES)}
    comp = shape.get("pca_components") if shape else None
    if comp is not None and len(comp):
        mean = np.asarray(shape["pca_mean"], dtype=np.float64)
        comp = np.asarray(comp, dtype=np.float64)                               # (3*48*48, K)
        z = (raster.reshape(len(raster), -1).astype(np.float64) - mean) @ comp
        for k in range(comp.shape[1]):
            out[f"rpc{k + 1}"] = z[:, k]
    return out


def shape_features(metadata, index, shape):
    rid, shell = grid(metadata, index)
    poly, raster = shapes_from_grid(rid, shell, int(index.n_zones))
    return named(poly, raster, shape)
