"""Single-option summary (2026-11-03, docs/plan/option-catalogue-20261103.md §5): every option level against the base
gt_u1b on the 66 design scenarios (real 40 + generated 26), from data/runs/b2rl/options/records and billing.

Per level: cost-weighted change sum(option - base) / sum(base) on all and on real; equal-weight mean of per-scenario
relative changes; cost-weighted change per city; comfort / energy / carbon and gas / cooling / fan / pump changes as a
share of the base total cost (they add up); the per-scenario oracle (on only where it helps); wins / losses; the
unbilled share of the boiler heat (billing_diag.py definition, summed over scenarios) and the heat delivered.
Only scenarios with both records count (n is reported). Writes data/runs/b2rl/options/summary.json and
summary.md (the table appended to the catalogue).

python3 -B pipeline/launch/launch.py pipeline/setup/options/opt_summary.py
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "pipeline/setup"))
sys.path.insert(0, str(HERE))
OUT = ROOT / "data/runs/b2rl/options"
FALLBACK = ROOT / "data/runs/b2rl/geo4/stage4/records"
DEVICES = ("gas", "ac_cool", "blower", "pump")
CITIES = ("atlanta", "denver", "fairbanks", "phoenix", "singapore")


def city(spec):
    return spec.split("/")[2].rsplit("_", 1)[0]


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def entry(name, spec, fallback=None):
    key = f"{name}__{spec.replace('/', '__')}.json"
    rec = read(OUT / "records" / key)
    if rec is None and fallback:
        rec = read(FALLBACK / f"{fallback}__{spec.replace('/', '__')}.json")
    if rec is None:
        return None
    s = rec["summary"]
    bill = read(OUT / "billing" / key)
    return {"cost": s["cost"], "comfort": s["comfort"], "energy": s["energy"], "carbon": s["carbon"],
            **{d: s["parts"][d]["energy"] + s["parts"][d]["carbon"] for d in DEVICES},
            "billing": bill}


def compare(base, opt, specs, real):
    both = [s for s in specs if base.get(s) and opt.get(s)]
    if not both:
        return None
    b = np.array([base[s]["cost"] for s in both])
    o = np.array([opt[s]["cost"] for s in both])
    total = b.sum()
    rel = o / b - 1.0
    real_both = [i for i, s in enumerate(both) if s in real]
    out = {"n": len(both), "n_real": len(real_both),
           "cw_all_pct": 100.0 * (o.sum() - total) / total,
           "cw_real_pct": 100.0 * (o[real_both].sum() - b[real_both].sum()) / b[real_both].sum() if real_both else None,
           "ew_all_pct": 100.0 * float(rel.mean()),
           "oracle_pct": 100.0 * (np.minimum(o, b).sum() - total) / total,
           "wins": int(np.sum(o < b - 1e-12)), "losses": int(np.sum(o > b + 1e-12)),
           "worst_pct": 100.0 * float(rel.max()), "best_pct": 100.0 * float(rel.min())}
    out["city_pct"], out["city_split"] = {}, {}
    for c in CITIES:
        idx = [i for i, s in enumerate(both) if city(s) == c]
        if idx:
            out["city_pct"][c] = 100.0 * (o[idx].sum() - b[idx].sum()) / b[idx].sum()
            out["city_split"][c] = {"n": len(idx), "wins": int(np.sum(o[idx] < b[idx] - 1e-12)),
                                    "losses": int(np.sum(o[idx] > b[idx] + 1e-12)),
                                    "oracle_pct": 100.0 * (np.minimum(o[idx], b[idx]).sum() - b[idx].sum())
                                    / b[idx].sum()}
    out["parts_pct"] = {k: 100.0 * sum(opt[s][k] - base[s][k] for s in both) / total
                        for k in ("comfort", "energy", "carbon", *DEVICES)}
    bills = [(base[s]["billing"], opt[s]["billing"]) for s in both]
    if all(x is not None and y is not None for x, y in bills):
        def share(rows):
            billed = sum(r["billed_j"] for r in rows)
            need = sum(r["delivered_j"] + r["loss_j"] + r["stored_j"] for r in rows)
            return 1.0 - billed / need if need > 0 else 0.0, need / 3.6e9, billed / 3.6e9
        sb, nb, bb = share([x for x, _ in bills])
        so, no, bo = share([y for _, y in bills])
        out["billing"] = {"unbilled_base_pct": 100.0 * sb, "unbilled_option_pct": 100.0 * so,
                          "heat_need_base_mwh": nb, "heat_need_option_mwh": no,
                          "billed_base_mwh": bb, "billed_option_mwh": bo,
                          "unbilled_change_mwh": (no - bo) - (nb - bb)}

        # billing-corrected ("honest") cost: the gas part (energy + carbon) scaled to the heat the boiler had to make,
        # gas x (delivered + loss + stored change) / billed, as a boiler billed for its burner heat would charge
        def honest(row, cost):
            bill, gas = row["billing"], row["gas"]
            billed = bill["billed_j"]
            need = bill["delivered_j"] + bill["loss_j"] + bill["stored_j"]
            if billed <= 0.0 or gas <= 0.0 or need <= 0.0:
                return cost
            return cost - gas + gas * need / billed

        hb = np.array([honest(base[s], base[s]["cost"]) for s in both])
        ho = np.array([honest(opt[s], opt[s]["cost"]) for s in both])
        out["billing"]["cw_honest_pct"] = 100.0 * (ho.sum() - hb.sum()) / hb.sum()
        out["billing"]["city_honest_pct"] = {
            c: 100.0 * (ho[idx].sum() - hb[idx].sum()) / hb[idx].sum()
            for c in CITIES for idx in [[i for i, s in enumerate(both) if city(s) == c]] if idx}
        out["billing"]["oracle_honest_pct"] = 100.0 * (np.minimum(ho, hb).sum() - hb.sum()) / hb.sum()
        for c in CITIES:
            idx = [i for i, s in enumerate(both) if city(s) == c]
            if idx:
                out["city_split"][c].update(
                    honest_wins=int(np.sum(ho[idx] < hb[idx] - 1e-12)),
                    honest_losses=int(np.sum(ho[idx] > hb[idx] + 1e-12)),
                    honest_oracle_pct=100.0 * (np.minimum(ho[idx], hb[idx]).sum() - hb[idx].sum()) / hb[idx].sum())
        out["billing"]["real_honest_pct"] = (100.0 * (ho[real_both].sum() - hb[real_both].sum()) / hb[real_both].sum()
                                             if real_both else None)
        out["honest_cost"] = {s: (float(x), float(y)) for s, x, y in zip(both, hb, ho)}
    out["per_scenario_pct"] = {s: 100.0 * float(r) for s, r in zip(both, rel)}
    return out


def main():
    import group_stage4 as g4
    from opt_models import LEVELS
    real, gen1, _ = g4.scenario_sets()
    specs = real + gen1
    base = {s: entry("gt_u1b", s, fallback="gt_u2a_pb11") for s in specs}
    identity = {}
    for s in specs:
        own = read(OUT / "records" / f"gt_u1b__{s.replace('/', '__')}.json")
        ref = read(FALLBACK / f"gt_u2a_pb11__{s.replace('/', '__')}.json")
        if own and ref:
            identity[s] = own["summary"]["cost"] - ref["summary"]["cost"]
    levels = {}
    for name, (code, switches) in LEVELS.items():
        opt = {s: entry(name, s) for s in specs}
        res = compare(base, opt, specs, set(real))
        if res is not None:
            levels[name] = {"code": code, "switches": switches, **res}
    # oracles across options: the best single level per scenario, and the sum of the per-level oracles
    # options only (not the pair tests P, room margin singles M or the baseline correction N4)
    done = [n for n, r in levels.items() if r["n"] == len(specs) and r["code"][0] not in "PMN"]
    oracle_best = None
    if done:
        b = np.array([base[s]["cost"] for s in specs])
        best = b.copy()
        for n in done:
            o = np.array([levels[n]["per_scenario_pct"][s] / 100.0 + 1.0 for s in specs]) * b
            best = np.minimum(best, o)
        oracle_best = 100.0 * (best.sum() - b.sum()) / b.sum()
    by_option = {}
    for n, r in levels.items():
        if r["code"][0] in "PMN":
            continue
        if r["code"] not in by_option or r["cw_all_pct"] < levels[by_option[r["code"]]]["cw_all_pct"]:
            by_option[r["code"]] = n
    oracle_sum = sum(min(levels[n]["oracle_pct"], 0.0) for n in by_option.values())
    # the same on the billing-corrected cost: best level per option by its corrected change, the sum of those
    # levels' corrected oracles, and the best corrected single level per scenario
    honest_by_option = {}
    for n, r in levels.items():
        if "billing" not in r or r["code"][0] in "PMN":
            continue
        key = r["code"]
        if key not in honest_by_option or r["billing"]["cw_honest_pct"] < \
                levels[honest_by_option[key]]["billing"]["cw_honest_pct"]:
            honest_by_option[key] = n
    oracle_sum_honest = sum(min(levels[n]["billing"]["oracle_honest_pct"], 0.0) for n in honest_by_option.values())
    oracle_best_honest = None
    honest_done = [n for n in done if "honest_cost" in levels[n]]
    if honest_done:
        hb = np.array([levels[honest_done[0]]["honest_cost"][s][0] for s in specs])
        best_h = hb.copy()
        for n in honest_done:
            best_h = np.minimum(best_h, np.array([levels[n]["honest_cost"][s][1] for s in specs]))
        oracle_best_honest = 100.0 * (best_h.sum() - hb.sum()) / hb.sum()
    # measured interactions of the pair tests: joint - (A + B), cost-weighted against gt_u1b, nominal and corrected
    from opt_models import PAIRS
    pairs = {}
    for label, (ab, a, b_name) in PAIRS.items():
        if not all(n in levels and "honest_cost" in levels[n] for n in (ab, a, b_name)):
            continue
        common = [s for s in specs if all(s in levels[n]["per_scenario_pct"] for n in (ab, a, b_name))]
        if not common:
            continue

        def delta(name, sel, honest):
            if honest:
                hb_ = sum(levels[name]["honest_cost"][s][0] for s in sel)
                return 100.0 * (sum(levels[name]["honest_cost"][s][1] for s in sel) - hb_) / hb_
            b_ = sum(base[s]["cost"] for s in sel)
            return 100.0 * sum(base[s]["cost"] * levels[name]["per_scenario_pct"][s] / 100.0 for s in sel) / b_

        row = {"n": len(common), "joint": ab, "a": a, "b": b_name}
        for honest in (False, True):
            key = "corrected" if honest else "nominal"
            row[key] = {"joint_pct": delta(ab, common, honest), "a_pct": delta(a, common, honest),
                        "b_pct": delta(b_name, common, honest)}
            row[key]["interaction_pct"] = row[key]["joint_pct"] - row[key]["a_pct"] - row[key]["b_pct"]
            row[key]["city_interaction_pct"] = {}
            for c in CITIES:
                sel = [s for s in common if city(s) == c]
                if sel:
                    row[key]["city_interaction_pct"][c] = (delta(ab, sel, honest) - delta(a, sel, honest)
                                                           - delta(b_name, sel, honest))
        pairs[label] = row
    summary = {"base": "gt_u1b (data/runs/b2rl/search/models/model-gt_u1b.json)", "simulator": "rom, 8640 steps",
               "pairs": pairs,
               "scenarios": {"real": len(real), "generated": len(gen1)},
               "base_identity_vs_gt_u2a_pb11": {"n": len(identity),
                                                "max_abs_cost_difference": max((abs(v) for v in identity.values()),
                                                                               default=None)},
               "base_cost_sum": float(sum(base[s]["cost"] for s in specs if base[s])),
               "levels": levels, "best_level_per_option": by_option,
               "oracle_sum_of_best_levels_pct": oracle_sum,
               "oracle_best_single_level_per_scenario_pct": oracle_best,
               "best_level_per_option_billing_corrected": honest_by_option,
               "oracle_sum_of_best_levels_billing_corrected_pct": oracle_sum_honest,
               "oracle_best_single_level_per_scenario_billing_corrected_pct": oracle_best_honest,
               "levels_complete": done}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    lines = ["| 代號 | 水準 | n | 成本加權（66） | 真實 40 | 等權 | 亞特蘭大 | 丹佛 | 費爾班克斯 | 鳳凰城 | 新加坡 | 舒適／能源／碳 | 瓦斯／冷氣電／風機 | 逐場開關上限 | 好／差 | 未計費熱量（底→選項） | 帳單修正後成本加權 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(levels.items(), key=lambda kv: kv[1]["cw_all_pct"]):
        p = r["parts_pct"]
        bill = r.get("billing")
        cities = [f"{r['city_pct'][c]:+.2f}" if c in r["city_pct"] else "—" for c in CITIES]
        lines.append(
            f"| {r['code']} | {n.replace('opt_', '')} | {r['n']} | {r['cw_all_pct']:+.2f}% | "
            f"{r['cw_real_pct']:+.2f}% | {r['ew_all_pct']:+.2f}% | " + " | ".join(cities) +
            f" | {p['comfort']:+.2f}／{p['energy']:+.2f}／{p['carbon']:+.2f} | "
            f"{p['gas']:+.2f}／{p['ac_cool']:+.2f}／{p['blower']:+.2f} | {r['oracle_pct']:+.2f}% | "
            f"{r['wins']}／{r['losses']} | "
            + (f"{bill['unbilled_base_pct']:.1f}→{bill['unbilled_option_pct']:.1f}% | "
               f"{bill['cw_honest_pct']:+.2f}%" if bill else "— | —") + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # per city: billing-corrected change, wins / losses and the per-scenario oracle (for the tree stage)
    city_lines = ["| 代號 | 水準 | " + " | ".join(["亞特蘭大", "丹佛", "費爾班克斯", "鳳凰城", "新加坡"]) + " |",
                  "|---|---|" + "---|" * len(CITIES)]
    for n, r in sorted(levels.items(), key=lambda kv: kv[1].get("billing", {}).get("cw_honest_pct", 99)):
        if "billing" not in r:
            continue
        cells = []
        for c in CITIES:
            sp = r["city_split"].get(c)
            if sp is None or "honest_wins" not in sp:
                cells.append("—")
                continue
            cells.append(f"{r['billing']['city_honest_pct'][c]:+.2f}%（{sp['honest_wins']}好／{sp['honest_losses']}差，"
                         f"上限 {sp['honest_oracle_pct']:+.2f}%）")
        city_lines.append(f"| {r['code']} | {n.replace('opt_', '')} | " + " | ".join(cells) + " |")
    (OUT / "summary_city.md").write_text("\n".join(city_lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"base identity: {summary['base_identity_vs_gt_u2a_pb11']}")
    print(f"oracle sum of best levels {oracle_sum:+.2f}%, best single level per scenario {oracle_best}")
    for label, row in pairs.items():
        for key in ("nominal", "corrected"):
            r = row[key]
            print(f"pair {label} ({key}, n {row['n']}): joint {r['joint_pct']:+.2f}% = A {r['a_pct']:+.2f}% + "
                  f"B {r['b_pct']:+.2f}% + interaction {r['interaction_pct']:+.2f}%; by city "
                  + " ".join(f"{c[:3]} {v:+.2f}" for c, v in r["city_interaction_pct"].items()))
    print(f"billing-corrected: oracle sum {oracle_sum_honest:+.2f}%, best single level per scenario "
          f"{oracle_best_honest}; best levels {honest_by_option}")
    for n in sorted(levels, key=lambda k: levels[k].get("billing", {}).get("cw_honest_pct", 99)):
        b = levels[n].get("billing")
        if b:
            print(f"  {n:22s} corrected {b['cw_honest_pct']:+.2f}% (real {b['real_honest_pct']:+.2f}%), oracle "
                  f"{b['oracle_honest_pct']:+.2f}%, cities "
                  + " ".join(f"{c[:3]} {v:+.2f}" for c, v in b["city_honest_pct"].items()))


if __name__ == "__main__":
    main()
