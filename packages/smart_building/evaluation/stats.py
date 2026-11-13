"""Statistics of the validation protocol (D-E02, D-E08, D-E09; notes/07 P0, P3, P5).

Unit (D-E01): per-scenario mean loss over 8640 steps, loss = -total_reward / 8640.

Main metric: S4 mean loss and the paired difference to pid_water. Intervals use
the product bootstrap over building x station x window (notes/07 §Z, P5.3):
every factor's levels are resampled with replacement, a scenario's weight is the
product of its levels' counts, and the statistic is the weighted mean; window
ids are unique within a station, so this equals nesting windows in stations.
95% percentile intervals from 50,000 draws (rliable defaults, notes/07 §H).
Draws whose total weight is zero carry no scenario and are drawn again.
The cross-structure SE (D-E09) is the standard deviation of the bootstrap means.

Secondary (D-E08): IQM (scipy trim_mean 0.25, rliable aggregate_iqm), probability
of improvement over pid_water (per scenario 1 if the candidate's loss is lower,
0.5 if equal, else 0 - the Mann-Whitney statistic with one run per side - then
averaged, with the product-bootstrap interval), worst scenario, P90 loss
(numpy percentile, linear interpolation), tables by station, season preset and
size stratum.
"""
import math

import numpy as np

STEPS = 8640  # D-E01
REPS = 50_000  # D-E08
LEVEL = 0.95
FACTORS = ("building", "station", "window")
CHUNK = 2_000


def loss(row, steps=STEPS):
    """Mean loss per step of one result row.

    Tracker-path rows (S1 reproduction): -total / 8640 (D-E01). Gateway-path rows
    (selection and S4 from J1b on, reviewer 2026-10-16): the gateway's own mean,
    total over the steps actually run (8641 for a full scenario).
    """
    if row.get("path") == "gateway":
        return -row["mean_reward"]
    return -row["total_reward"] / steps


def classify(new_building, held_out_station):
    """S1-S4 (D-E02; notes/07 P0.2): building new to training x station held out of training."""
    if new_building:
        return "S4" if held_out_station else "S2"
    return "S3" if held_out_station else "S1"


def _codes(labels):
    index = {}
    return np.array([index.setdefault(label, len(index)) for label in labels], dtype=np.int64)


def factor_codes(records, factors=FACTORS):
    return [_codes([r[f] for r in records]) for f in factors]


def product_bootstrap(values, codes, reps=REPS, seed=0):
    """Weighted means of `reps` product-bootstrap draws (notes/07 §Z)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    filled = 0
    while filled < reps:
        size = min(CHUNK, reps - filled)
        weights = np.ones((size, values.size))
        for code in codes:
            levels = int(code.max()) + 1
            counts = rng.multinomial(levels, np.full(levels, 1.0 / levels), size=size)
            weights *= counts[:, code]
        total = weights.sum(axis=1)
        keep = total > 0
        # Elementwise sum rather than matmul: no BLAS, the same result on every machine.
        means = (weights[keep] * values).sum(axis=1) / total[keep]
        if not np.isfinite(means).all():
            raise FloatingPointError("non-finite bootstrap mean")
        take = min(means.size, reps - filled)
        out[filled:filled + take] = means[:take]
        filled += take
    return out


def naive_bootstrap(values, reps=REPS, seed=0):
    """Scenario-level iid bootstrap means (only for comparison; not used for decisions)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for start in range(0, reps, CHUNK):
        size = min(CHUNK, reps - start)
        picks = rng.integers(0, values.size, size=(size, values.size))
        out[start:start + size] = values[picks].mean(axis=1)
    return out


def interval(draws, level=LEVEL):
    """Percentile interval of bootstrap draws."""
    tail = (1.0 - level) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def iqm(values):
    from scipy.stats import trim_mean
    return float(trim_mean(np.asarray(values, dtype=float), 0.25, axis=None))


def improvement_indicator(candidate_loss, baseline_loss):
    """Mann-Whitney statistic with one run each: candidate better (lower loss) 1, tie 0.5."""
    if candidate_loss < baseline_loss:
        return 1.0
    if candidate_loss == baseline_loss:
        return 0.5
    return 0.0


def summary(records, key="loss", reps=REPS, seed=0, factors=FACTORS):
    """Mean, product-bootstrap CI and SE, naive SE, IQM, worst, P90 of records[key]."""
    values = np.array([r[key] for r in records], dtype=float)
    draws = product_bootstrap(values, factor_codes(records, factors), reps, seed)
    naive = naive_bootstrap(values, reps, seed)
    worst = max(records, key=lambda r: r[key])
    low, high = interval(draws)
    return {"n": int(values.size), "mean": float(values.mean()), "ci95": [low, high],
            "se_cross": float(draws.std(ddof=1)), "se_naive": float(naive.std(ddof=1)),
            "iqm": iqm(values), "p90": float(np.percentile(values, 90)),
            "worst": {"value": float(worst[key]), "scenario": worst.get("scenario")},
            "bootstrap": {"reps": reps, "seed": seed, "factors": list(factors), "level": LEVEL}}


def paired(candidate, baseline, reps=REPS, seed=0, factors=FACTORS):
    """Paired difference candidate - baseline on the same scenarios, with PoI (D-E08)."""
    base = {r["scenario"]: r for r in baseline}
    missing = [r["scenario"] for r in candidate if r["scenario"] not in base]
    if missing or len(base) != len(candidate):
        raise ValueError(f"candidate and baseline must cover the same scenarios (missing {missing[:3]})")
    records = []
    for r in candidate:
        b = base[r["scenario"]]
        records.append(dict(r, diff=r["loss"] - b["loss"],
                            improvement=improvement_indicator(r["loss"], b["loss"])))
    result = summary(records, "diff", reps, seed, factors)
    poi = product_bootstrap([r["improvement"] for r in records], factor_codes(records, factors), reps, seed)
    result["probability_of_improvement"] = {
        "value": float(np.mean([r["improvement"] for r in records])), "ci95": list(interval(poi)),
        "criterion": "CI lower > 0.5 and upper > 0.75 (notes/07 P5.4)"}
    result["candidate_better_count"] = int(sum(r["improvement"] == 1.0 for r in records))
    return result, records


def table(records, by, key="loss"):
    """Mean of records[key] per value of `by` (diagnostic tables, D-E08); for paired differences
    (key "diff") also the count of scenarios where the candidate is better (diff < 0)."""
    groups = {}
    for r in records:
        groups.setdefault(r[by], []).append(r[key])
    out = {}
    for k, v in sorted(groups.items(), key=lambda item: str(item[0])):
        out[str(k)] = {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))}
        if key == "diff":
            out[str(k)]["better"] = int(sum(x < 0 for x in v))
    return out


def cross_se_rule(candidate_mean, champion_mean, se_cross):
    """D-E09: replace the champion only if the candidate's CV loss is lower by more than 1 cross SE.

    se_cross is the product-bootstrap SE of the paired difference candidate - champion
    on the same scenarios (notes/07 P3.2: §O's s/sqrt(n) of the paired loss differences,
    with the cross-structure SE in its place).
    """
    return bool(champion_mean - candidate_mean > se_cross) if math.isfinite(se_cross) else False
