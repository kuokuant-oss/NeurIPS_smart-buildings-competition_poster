"""Pipeline 5 stage 乙, single-option screening (2026-11-04, docs/plan/pipeline5-plan-20261104.md §二乙): the
stratified 30-scenario set, the per-level report against the base gt_u1b and the harm filter.

Screen set (screen30.json): from the 66 design scenarios (group_stage4.scenario_sets(): real 40 + one per generated
building 26) every cold scenario (scenario mean outside air below COLD_C, from the base record) and at least one of
every city x season stratum (season from the scenario's start month; the non-cold part of a stratum is its own
stratum); the remaining places go to the strata with the largest base-cost share. Each scenario carries the weight
N_h / n_h of its stratum, so weighted sums estimate the 66-scenario totals ("population-weighted"); the plain sums
over the 30 are reported too ("sample").

Report per level (cost-weighted = sum(option - base) / sum(base), the leaderboard's mean cost):
cost-weighted and equal-weighted change (population-weighted and sample), by city, by season, by cost part
(comfort, energy, carbon, gas, cooling electricity, fan, pump; shares of the base total, they add up), the
per-scenario best (on only where it helps: the upper bound), better / worse counts, and the billing-corrected values
(the gas part scaled to the heat the boiler had to make, options/opt_summary.py "honest" cost).

Harm filter (plan, corrected by the option review 2026-11-05, docs/plan/option-review-20261105.md §二):
- a stacked level (compare_with, e.g. on o546_1) is judged against that level, not against the base (S1);
- better / worse count only changes of at least MATERIAL (0.05 %) of the scenario's cost (S2);
- "no effect" when |cost-weighted| < 0.03 % and the upper bound is above -0.1 % (S4);
- keep only when the cost-weighted change is below 0 and not (more than half the scenarios worse and the upper
  bound above -0.3 %) (S3); everything else is dropped unless "conditional";
- conditional: a city or season stratum inside the level's declared conditions (when it declares any) with >= 3
  scenarios, more better than worse, and a gain <= -0.3 % (S5; for the tree stage, not the rule design).
The paper track judges the billing-corrected values (physics levels only); the competition track the nominal ones.
The design stage takes its factors from the reviewed list (data/runs/b2rl/geo5/review.json), not from these verdicts.
"""
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from geo5_lib import ROOT, read_json
import geo5_record as rec

COLD_C = 5.0
SEASON_OF = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer",
             8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
DEVICES = ("gas", "ac_cool", "blower", "pump")
PARTS = ("comfort", "energy", "carbon") + DEVICES
BASE_SOURCES = [("options", ROOT / "data/runs/b2rl/options", "gt_u1b"),            # full month, with billing
                ("geo4", ROOT / "data/runs/b2rl/geo4/stage4", "gt_u2a_pb11")]        # same behaviour, no billing
DROP_CW, DROP_ORACLE, COND_GAIN = 0.5, -0.3, -0.3
MATERIAL = 0.0005                      # a scenario counts as better / worse from 0.05 % of its cost (review S2)
NO_EFFECT_CW, NO_EFFECT_ORACLE = 0.03, -0.1
COND_MIN_N = 3


def city(spec):
    return spec.split("/")[2].rsplit("_", 1)[0]


def season(spec):
    sys.path.insert(0, str(ROOT / "packages"))
    from smart_building.learned.b2rl_env import spec_dir
    gin = (spec_dir(spec) / "config.gin").read_text(encoding="utf-8")
    m = re.search(r"start_timestamp\s*=\s*'(\d+)-(\d+)-", gin)
    return SEASON_OF[int(m.group(2))], int(m.group(2))


def design_specs():
    import group_stage4 as g4
    real, gen1, _ = g4.scenario_sets()
    return real, gen1


def base_full_record(spec):
    for label, d, name in BASE_SOURCES:
        r = read_json(d / "records" / rec.key(name, spec))
        if r is not None:
            return label, r
    return None, None


def build_screen30(path, n=30):
    """Write the stratified set (path) and return it."""
    real, gen1 = design_specs()
    specs = real + gen1
    meta = {}
    for s in specs:
        _, r = base_full_record(s)
        if r is None:
            raise RuntimeError(f"no base record for {s}")
        se, month = season(s)
        meta[s] = {"city": city(s), "season": se, "month": month, "real": s in set(real),
                   "oat_c": round(float(np.mean(r["hourly"]["outside_t"])), 2), "base_cost": r["summary"]["cost"]}
    for s in specs:
        meta[s]["stratum"] = "cold" if meta[s]["oat_c"] < COLD_C else f"{meta[s]['city']}|{meta[s]['season']}"
    strata = {}
    for s in specs:
        strata.setdefault(meta[s]["stratum"], []).append(s)
    chosen = list(strata.get("cold", []))

    def pick(h):
        rest = [s for s in strata[h] if s not in chosen]
        med = float(np.median([meta[s]["base_cost"] for s in rest]))
        # a real scenario first (the gates are real), then the one nearest the stratum's median cost, then the name
        return min(rest, key=lambda s: (not meta[s]["real"], abs(meta[s]["base_cost"] - med), s))

    for h in sorted(x for x in strata if x != "cold"):
        chosen.append(pick(h))
    if len(chosen) > n:
        raise RuntimeError(f"{len(chosen)} scenarios needed for the cold ones and one per stratum (> {n})")
    total = sum(meta[s]["base_cost"] for s in specs)
    while len(chosen) < n:
        open_ = [h for h in strata if h != "cold" and any(s not in chosen for s in strata[h])]
        h = max(open_, key=lambda h: (sum(meta[s]["base_cost"] for s in strata[h]) / total * n
                                      - sum(s in chosen for s in strata[h]), h))
        chosen.append(pick(h))
    weights = {}
    for h, members in strata.items():
        k = sum(s in chosen for s in members)
        for s in members:
            if s in chosen:
                weights[s] = len(members) / k
    out = {"made": __import__("time").strftime("%Y-%m-%d %H:%M:%S"), "n": len(chosen), "cold_c": COLD_C,
           "rule": "all cold (mean outside air < 5 C) + one per city x season stratum (non-cold part) + the rest by "
                   "base-cost share; weight = stratum size / picks",
           "specs": chosen, "weights": weights, "meta": {s: meta[s] for s in chosen},
           "strata": {h: {"size": len(m), "picked": [s for s in m if s in chosen]} for h, m in sorted(strata.items())},
           "population": len(specs)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- report
def entry(r, bill):
    s = r["summary"]
    return {"cost": s["cost"], "comfort": s["comfort"], "energy": s["energy"], "carbon": s["carbon"],
            **{d: s["parts"][d]["energy"] + s["parts"][d]["carbon"] for d in DEVICES}, "billing": bill}


def honest(row):
    bill, gas, cost = row["billing"], row["gas"], row["cost"]
    if not bill:
        return None
    billed = bill["billed_j"]
    need = bill["delivered_j"] + bill["loss_j"] + bill["stored_j"]
    if billed <= 0.0 or gas <= 0.0 or need <= 0.0:
        return cost
    return cost - gas + gas * need / billed


def _cw(o, b, w, idx=None):
    idx = np.arange(len(b)) if idx is None else np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return None
    return 100.0 * float(np.sum(w[idx] * (o[idx] - b[idx])) / np.sum(w[idx] * b[idx]))


def compare(base, opt, specs, weights, meta, conditions=None):
    """One level against the base on the scenarios both have; see the module text."""
    both = [s for s in specs if base.get(s) and opt.get(s)]
    if not both:
        return None
    b = np.array([base[s]["cost"] for s in both])
    o = np.array([opt[s]["cost"] for s in both])
    w = np.array([weights.get(s, 1.0) for s in both])
    one = np.ones(len(both))
    rel = o / b - 1.0
    out = {"n": len(both), "cw_pct": _cw(o, b, w), "cw_sample_pct": _cw(o, b, one),
           "ew_pct": 100.0 * float(np.sum(w * rel) / np.sum(w)), "ew_sample_pct": 100.0 * float(rel.mean()),
           "oracle_pct": _cw(np.minimum(o, b), b, w), "wins": int(np.sum(o < b * (1.0 - MATERIAL))),
           "losses": int(np.sum(o > b * (1.0 + MATERIAL))), "best_pct": 100.0 * float(rel.min()),
           "worst_pct": 100.0 * float(rel.max())}
    tot = float(np.sum(w * b))
    out["parts_pct"] = {k: 100.0 * float(sum(wi * (opt[s][k] - base[s][k]) for s, wi in zip(both, w))) / tot
                        for k in PARTS}

    def groups(field):
        res = {}
        for g in sorted({meta[s][field] for s in both}):
            idx = [i for i, s in enumerate(both) if meta[s][field] == g]
            res[g] = {"n": len(idx), "cw_pct": _cw(o, b, w, idx), "oracle_pct": _cw(np.minimum(o, b), b, w, idx),
                      "wins": int(np.sum(o[idx] < b[idx] * (1.0 - MATERIAL))),
                      "losses": int(np.sum(o[idx] > b[idx] * (1.0 + MATERIAL)))}
        return res

    out["city"], out["season"] = groups("city"), groups("season")
    hb = [honest(base[s]) for s in both]
    ho = [honest(opt[s]) for s in both]
    if all(x is not None for x in hb + ho):
        hb, ho = np.array(hb), np.array(ho)

        def share(rows):
            billed = sum(r["billed_j"] for r in rows)
            need = sum(r["delivered_j"] + r["loss_j"] + r["stored_j"] for r in rows)
            return 100.0 * (1.0 - billed / need) if need > 0 else 0.0

        out["billing"] = {"unbilled_base_pct": share([base[s]["billing"] for s in both]),
                          "unbilled_option_pct": share([opt[s]["billing"] for s in both]),
                          "cw_pct": _cw(ho, hb, w), "cw_sample_pct": _cw(ho, hb, one),
                          "oracle_pct": _cw(np.minimum(ho, hb), hb, w),
                          "wins": int(np.sum(ho < hb * (1.0 - MATERIAL))),
                          "losses": int(np.sum(ho > hb * (1.0 + MATERIAL)))}
        for field in ("city", "season"):
            for g, row in out[field].items():
                idx = [i for i, s in enumerate(both) if meta[s][field] == g]
                row["billing_cw_pct"] = _cw(ho, hb, w, idx)
                row["billing_wins"] = int(np.sum(ho[idx] < hb[idx] * (1.0 - MATERIAL)))
                row["billing_losses"] = int(np.sum(ho[idx] > hb[idx] * (1.0 + MATERIAL)))
        out["honest_cost"] = {s: [float(x), float(y)] for s, x, y in zip(both, hb, ho)}
    if conditions:
        idx = [i for i, s in enumerate(both) if (not conditions.get("cities") or meta[s]["city"] in conditions["cities"])
               and (not conditions.get("seasons") or meta[s]["season"] in conditions["seasons"])
               and (not conditions.get("oat_c") or (meta[s].get("oat_c") is not None
                                                    and conditions["oat_c"][0] <= meta[s]["oat_c"] <= conditions["oat_c"][1]))]
        out["conditions_view"] = {"n": len(idx), "cw_pct": _cw(o, b, w, idx),
                                  "wins": int(np.sum(o[idx] < b[idx] * (1.0 - MATERIAL))) if idx else 0,
                                  "losses": int(np.sum(o[idx] > b[idx] * (1.0 + MATERIAL))) if idx else 0}
    out["per_scenario_pct"] = {s: 100.0 * float(r) for s, r in zip(both, rel)}
    out["cost"] = {s: [float(x), float(y)] for s, x, y in zip(both, b, o)}
    return out


def verdict(r, level, track):
    """Harm filter of one level on one track: keep / drop / conditional (+ why)."""
    if track == "paper" and level["category"] != "physics":
        return {"keep": False, "status": "not on the paper track", "metric": None}
    use_bill = track == "paper" and "billing" in r
    m = r["billing"] if use_bill else r
    cw, oracle = m["cw_pct"], m["oracle_pct"]
    wins, losses, n = m["wins"], m["losses"], r["n"]
    why = []
    metric = "billing-corrected" if use_bill else "nominal"
    if abs(cw) < NO_EFFECT_CW and oracle > NO_EFFECT_ORACLE:         # review S4
        return {"keep": False, "status": "no effect", "metric": metric, "cw_pct": cw,
                "why": [f"cost-weighted {cw:+.3f}%, upper bound {oracle:+.3f}%"]}
    if cw >= 0.0:                                                     # review S3: must help on average
        why.append(f"cost-weighted {cw:+.2f}% >= 0" + (f" (> +{DROP_CW}%)" if cw > DROP_CW else ""))
    if losses > n / 2 and oracle > DROP_ORACLE:
        why.append(f"{losses}/{n} worse and the upper bound {oracle:+.2f}% > {DROP_ORACLE}%")
    if track == "paper" and level.get("paper_required"):   # e.g. O542 ventilation code: the paper version must have it
        return {"keep": True, "status": "keep (paper required)", "metric": metric, "cw_pct": cw, "why": why}
    if not why:
        return {"keep": True, "status": "keep", "metric": metric, "cw_pct": cw, "why": []}
    cond = []
    declared = level.get("conditions") or {}
    inside = {"city": declared.get("cities"), "season": declared.get("seasons")}
    for field in ("city", "season"):
        for g, row in r[field].items():
            if inside[field] and g not in inside[field]:              # review S5: inside the declared domain only
                continue
            gcw = row["billing_cw_pct"] if use_bill else row["cw_pct"]
            gw = row["billing_wins"] if use_bill else row["wins"]
            gl = row["billing_losses"] if use_bill else row["losses"]
            if row["n"] >= COND_MIN_N and gcw is not None and gcw <= COND_GAIN and gw > gl:
                cond.append({"field": field, "group": g, "cw_pct": gcw, "wins": gw, "losses": gl})
    if cond:
        return {"keep": False, "status": "conditional", "metric": metric, "cw_pct": cw, "why": why, "where": cond}
    return {"keep": False, "status": "drop", "metric": metric, "cw_pct": cw, "why": why}


CITY_ZH = {"atlanta": "亞特蘭大", "denver": "丹佛", "fairbanks": "費爾班克斯", "phoenix": "鳳凰城", "singapore": "新加坡"}
SEASON_ZH = {"winter": "冬", "spring": "春", "summer": "夏", "autumn": "秋"}


def direction(report, spec_levels, track, batch, batches):
    """Per-track direction after a screening batch (user 2026-11-04): which levels help, in which cities and seasons,
    and the estimated saving (best level per option added up: an additive estimate, interactions not measured yet;
    the per-scenario upper bounds added up as the ceiling). Paper: billing-corrected, physics only; competition:
    nominal, every level."""
    use_bill = track == "paper"
    helps, conditional, dropped = [], [], []
    for lid, row in report["levels"].items():
        r, v = row.get("result"), row.get(track) or {}
        if not r:
            continue
        m = (r.get("billing") or r) if use_bill else r
        entry = {"id": lid, "option": row["option"], "layer": row["layer"], "category": row["category"],
                 "cw_pct": m["cw_pct"], "oracle_pct": m["oracle_pct"], "wins": m["wins"], "losses": m["losses"],
                 "n": r["n"], "note": (spec_levels.get(lid) or {}).get("expected")}
        for field in ("city", "season"):
            entry[field] = {g: (x["billing_cw_pct"] if use_bill and "billing_cw_pct" in x else x["cw_pct"])
                            for g, x in r[field].items()}
        if v.get("keep") and entry["cw_pct"] < 0.0:
            helps.append(entry)
        elif v.get("keep"):
            dropped.append({"id": lid, "status": "kept, no gain", "why": [f"cost-weighted {entry['cw_pct']:+.3f}%"]})
        elif v.get("status") == "conditional":
            conditional.append({**entry, "where": v.get("where")})
        else:
            dropped.append({"id": lid, "status": v.get("status"), "why": v.get("why")})
    best = {}
    for e in helps:
        if e["option"] not in best or e["cw_pct"] < best[e["option"]]["cw_pct"]:
            best[e["option"]] = e
    cities = sorted({c for e in best.values() for c in e["city"]})
    seasons = sorted({c for e in best.values() for c in e["season"]})
    out = {"track": track, "batch": batch, "batches": batches, "made": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
           "metric": "billing-corrected" if use_bill else "nominal", "levels_screened": len(report["levels"]),
           "helps": sorted(helps, key=lambda e: e["cw_pct"]), "conditional": conditional, "dropped": dropped,
           "best_level_per_option": {o: e["id"] for o, e in best.items()},
           "estimated_saving_pct": sum(min(e["cw_pct"], 0.0) for e in best.values()),
           "estimated_saving_city_pct": {c: sum(min(e["city"].get(c) or 0.0, 0.0) for e in best.values())
                                         for c in cities},
           "estimated_saving_season_pct": {s: sum(min(e["season"].get(s) or 0.0, 0.0) for e in best.values())
                                           for s in seasons},
           "ceiling_pct": sum(min(e["oracle_pct"], 0.0) for e in best.values()),
           "caveat": "additive sum of single-level changes on the screen set (population-weighted); interactions come "
                     "from the rule-optimisation design"}
    zh = "論文版（只含物理上站得住的選項，計費修正後）" if use_bill else "比賽版（含只限比賽的選項，名目成本）"
    md = [f"# 方向摘要：{zh}，第 {batch}／{batches} 批", "",
          f"- 已篩選 {out['levels_screened']} 個選項等級；有幫助 {len(helps)} 個、只在部分氣候有幫助 "
          f"{len(conditional)} 個、沒有效果或剔除 {len(dropped)} 個。",
          f"- 估計省下：{out['estimated_saving_pct']:+.2f}%（各選項取最佳等級直接相加，交互作用還沒量；"
          f"單場都挑對的上限 {out['ceiling_pct']:+.2f}%）。",
          "- 各城市估計：" + "、".join(f"{CITY_ZH.get(c, c)} {v:+.2f}%" for c, v in
                                    out["estimated_saving_city_pct"].items()) + "。",
          "- 各季節估計：" + "、".join(f"{SEASON_ZH.get(s, s)} {v:+.2f}%" for s, v in
                                    out["estimated_saving_season_pct"].items()) + "。", "",
          "| 選項等級 | 層級 | 成本變化 | 好／差 | 幫助最多的城市 | 幫助最多的季節 |", "|---|---|---|---|---|---|"]
    for e in out["helps"]:
        c = min(e["city"].items(), key=lambda kv: kv[1] if kv[1] is not None else 99)
        s = min(e["season"].items(), key=lambda kv: kv[1] if kv[1] is not None else 99)
        md.append(f"| {e['id']} | {e['layer']} | {e['cw_pct']:+.2f}% | {e['wins']}／{e['losses']} | "
                  f"{CITY_ZH.get(c[0], c[0])} {c[1]:+.2f}% | {SEASON_ZH.get(s[0], s[0])} {s[1]:+.2f}% |")
    if conditional:
        md += ["", "只在部分氣候有幫助（留給樹決定何時開）："]
        for e in conditional:
            where = "、".join(f"{CITY_ZH.get(w['group'], SEASON_ZH.get(w['group'], w['group']))} {w['cw_pct']:+.2f}%"
                             for w in e["where"])
            md.append(f"- {e['id']}：整體 {e['cw_pct']:+.2f}%；{where}")
    return out, "\n".join(md) + "\n"


def table_md(report):
    """Traditional-Chinese table of the screen (one row per level)."""
    lines = ["| 選項等級 | 層級 | 類別 | n | 成本加權（母體） | 成本加權（30 場） | 等權 | 單場最佳上限 | 好／差 | "
             "計費修正後 | 論文版 | 比賽版 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lid, row in sorted(report["levels"].items(), key=lambda kv: kv[1]["result"]["cw_pct"]
                           if kv[1].get("result") else 99):
        r = row.get("result")
        if not r:
            lines.append(f"| {lid} | {row['layer']} | {row['category']} | 0 | — | — | — | — | — | — | — | — |")
            continue
        bill = f"{r['billing']['cw_pct']:+.2f}%" if "billing" in r else "—"
        lines.append(f"| {lid} | {row['layer']} | {row['category']} | {r['n']} | {r['cw_pct']:+.2f}% | "
                     f"{r['cw_sample_pct']:+.2f}% | {r['ew_pct']:+.2f}% | {r['oracle_pct']:+.2f}% | "
                     f"{r['wins']}／{r['losses']} | {bill} | {row['paper']['status']} | {row['competition']['status']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit("library module; used by pipeline/setup/geo_pipeline5.py")
