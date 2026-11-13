"""Whole-floor raster per room after the RPLAN image layout (Wu et al. 2019, "Data-driven interior plan generation for
residential buildings", ACM TOG 38(6); a multi-channel 256 x 256 image of the whole floor, channel layout known from
secondary sources only), reduced by principal components (eigen-images, Turk & Pentland 1991, J. Cognitive
Neuroscience 3(1):71-86) because the literature's convolutional encoder cannot be trained on our ~100 buildings
(spec docs/plan/geo-reps-literature-audit-20261110.md §四, raster variant A).

A floor is one connected building outline on the plan grid (the exterior shell with everything it encloses). Its
bounding box is centred in a square of the longer side S (>= 256 cells in all 239 buildings) and reduced to 256 x 256
pixels; pixel i covers source cells [floor(i S / 256), floor((i + 1) S / 256)). Channels:
    0  exterior shell: any shell cell in the pixel (thin walls survive the reduction);
    1  floor interior: at least half of the pixel's cells inside the outline and not shell;
    2  this room: at least half of the pixel's cells in the room.
A room's image is x = [shell_f, interior_f, room] (3 x 65536 bits); its scores are (x - mean) . V, computed as sums
of the rows of mean-free loadings at the set pixels. Used by policies/b2rl_shape.py only when a model's
room_policy.shape has "floor_pca_components"; nothing in the submission path imports this module."""
import numpy as np

SIZE = 256
NPIX = SIZE * SIZE


def floors(shell):
    """Floor label grid (0 outside, 1..k) and k: 8-connected components of the shell with what it encloses."""
    from scipy import ndimage as nd
    bld = nd.binary_fill_holes(shell)
    lab, k = nd.label(bld, structure=np.ones((3, 3), dtype=bool))
    return lab, int(k)


def _pix(rows, cols, box):
    """Pixel index (row * SIZE + col) of grid cells inside a floor box (r0, c0, S) and the per-pixel cell counts."""
    r0, c0, S = box
    e = (np.arange(SIZE + 1) * S) // SIZE
    pr = np.searchsorted(e, rows - r0, side="right") - 1
    pc = np.searchsorted(e, cols - c0, side="right") - 1
    return pr * SIZE + pc, np.outer(np.diff(e), np.diff(e)).ravel()


def floor_pixels(rid, shell, n):
    """Per floor: (shell pixel indices, interior pixel indices); per room: (floor number, room pixel indices)."""
    from scipy import ndimage as nd
    lab, k = floors(shell)
    objs = nd.find_objects(lab)
    boxes, fl = [], []
    for f, o in enumerate(objs):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        S = max(h, w)
        if S < SIZE:
            raise ValueError(f"floor {f} is {S} cells, smaller than the {SIZE}-pixel image")
        boxes.append((o[0].start - (S - h) // 2, o[1].start - (S - w) // 2, S))
        m = lab == f + 1
        rs, cs = np.nonzero(m & shell)
        p, area = _pix(rs, cs, boxes[-1])
        sh = np.nonzero(np.bincount(p, minlength=NPIX) > 0)[0]
        rs, cs = np.nonzero(m & ~shell)
        p, _ = _pix(rs, cs, boxes[-1])
        inside = np.nonzero(2 * np.bincount(p, minlength=NPIX) >= area)[0]
        fl.append((sh, inside))
    rooms = []
    rows, cols = np.nonzero(rid >= 0)
    ids = rid[rows, cols]
    order = np.argsort(ids, kind="stable")
    rows, cols, ids = rows[order], cols[order], ids[order]
    cuts = np.searchsorted(ids, np.arange(n + 1))
    for i in range(n):
        r, c = rows[cuts[i]:cuts[i + 1]], cols[cuts[i]:cuts[i + 1]]
        f = int(np.bincount(lab[r, c]).argmax()) - 1
        p, area = _pix(r, c, boxes[f])
        rooms.append((f, np.nonzero(2 * np.bincount(p, minlength=NPIX) >= area)[0]))
    return fl, rooms


def scores(fl, rooms, mean, comp):
    """(n, K) principal-component scores; mean (3 * NPIX,), comp (3 * NPIX, K)."""
    mean = np.asarray(mean, dtype=np.float64)
    comp = np.asarray(comp, dtype=np.float64)
    base = mean @ comp
    blocks = [comp[:NPIX], comp[NPIX:2 * NPIX], comp[2 * NPIX:]]
    fz = [blocks[0][sh].sum(0) + blocks[1][ins].sum(0) for sh, ins in fl]
    out = np.zeros((len(rooms), comp.shape[1]))
    for i, (f, px) in enumerate(rooms):
        out[i] = fz[f] + blocks[2][px].sum(0) - base
    return out


def names(k):
    return tuple(f"fpc{j}" for j in range(1, k + 1))
