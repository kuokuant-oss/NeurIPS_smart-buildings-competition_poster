"""Single-option models on gt_u1b (2026-11-03, docs/plan/option-catalogue-20261103.md): each is gt_u1b's model with
one option's switches added to its rule_config (nothing else changes). Writes
data/runs/b2rl/options/models/model-opt_<name>_<level>.json (never search/models/); checks every model loads.

python3 -B pipeline/launch/launch.py pipeline/setup/options/opt_models.py
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages"))
BASE = ROOT / "data/runs/b2rl/search/models/model-gt_u1b.json"
OUT = ROOT / "data/runs/b2rl/options/models"
CYCLING = {"occupied_cycling": True, "cycle_charge_k": 0.0, "cycle_off_k": 0.5}
# in run order (expected effect first); option code -> catalogue §2.1
LEVELS = {
    "opt_budgetpre_070": ("O3", {"ahu_heat_budget": 0.7, "ahu_heat_budget_scope": "prestart"}),
    "opt_break_7200": ("O1", {"opt_start_break_max_s": 7200.0}),
    "opt_stagger_on": ("O12", {"opt_start_zone_stagger": True}),
    "opt_floor8_28115": ("O6", {"discharge_min_winter_k": 281.15}),
    "opt_budget_080": ("O3", {"ahu_heat_budget": 0.8}),
    "opt_arrival_60k03": ("O2", {"arrival_hold_min": 60, "arrival_hold_k": 0.3}),
    "opt_satmix_on": ("O11", {"sat_cost_model": "economizer"}),
    "opt_ledger_on": ("O7", {**CYCLING, "cycle_vent_ledger": True}),
    # 13:5x: break_7200 showed that a longer cap moves G36's single sample earlier (Singapore missed the late drift,
    # comfort x3 on Mondays); break_10800 dropped, replaced by the re-sampled (latest-feasible) start alone and with
    # the after-break cap
    "opt_cont_on": ("O1", {"opt_start_continuous": True}),
    "opt_break_7200c": ("O1", {"opt_start_continuous": True, "opt_start_break_max_s": 7200.0}),
    # 14:3x: the economizer model helped the hot cities (billing-corrected Phoenix -1.07%, Atlanta -0.69%) and hurt
    # Fairbanks (+0.85%); the same model only at 15 C outside air and above
    "opt_satmix_warm": ("O11", {"sat_cost_model": "economizer", "sat_cost_model_min_oat_k": 288.15}),
    "opt_budget_060": ("O3", {"ahu_heat_budget": 0.6}),
    "opt_floor9_28215": ("O6", {"discharge_min_winter_k": 282.15}),
    "opt_midday_10": ("O8", {"midday_charge_k": 1.0}),
    "opt_arrival_120k05": ("O2", {"arrival_hold_min": 120, "arrival_hold_k": 0.5}),
    "opt_satmix_lock": ("O11", {"sat_cost_model": "economizer", "sat_chiller_lockout_k": 288.15}),
    "opt_sp_combo": ("O10", {"sp_max_setback_pa": 200.0, "sp_rogue_hours": 24.0}),
    "opt_setback_frac07": ("O5", {"setback_min_zone_frac": 0.7, "mode_release_margin_k": 2.0}),
    "opt_lead_05": ("O13", {"zone_temp_lead": 0.5}),
    # 14:4x: precool_k is the cooling target ABOVE THE LOWER EDGE (A10), so 0.3 was a deep pre-cool, not the
    # shallow one the catalogue meant; dropped before it ran, replaced by 2.5 (about 0.4-0.5 K below the usual target)
    "opt_precool_25": ("O9", {"precool_until_hour": 12, "precool_k": 2.5}),
    # climate-gated options (coordinator 14:4x; docs/plan/lit/climate-coverage-20261103.md §6 C1-C4)
    "opt_c3_freeze": ("C3", {"freeze_min_airflow": True}),
    "opt_c4_tropic": ("C4", {"opt_start_continuous": True, "opt_start_break_max_s": 7200.0,
                             "opt_start_break_min_oat_k": 297.15}),
    "opt_c1_flush": ("C1", {"night_flush": True, "night_flush_min_range_k": 12.0, "night_flush_min_max_k": 298.15}),
    "opt_c2_precool": ("C2", {"precool_until_hour": 12, "precool_k": 2.5, "precool_min_range_k": 10.0,
                              "precool_min_max_k": 303.15}),
    # coordinator 16:1x (docs/plan/energy-audit-20261103.md §4.3, N4): the 62.1 outdoor-air total per AHU, a
    # paper-track baseline correction (expected to cost, not to save)
    "opt_n4_vot": ("N4", {"ahu_vot_check": True}),
    # coordinator 16:3x: the O13 fix (lead only in steady operation, clipped), then pair tests with the physical
    # partners of ems-options §6 (interactions 3, 4, 7, 8); P = pair, M = room margin single
    "opt_lead_05s": ("O13", {"zone_temp_lead": 0.5, "zone_temp_lead_steady": True, "zone_temp_lead_max_k": 0.3}),
    "opt_pair_break_budgetpre": ("P4", {"opt_start_break_max_s": 7200.0, "ahu_heat_budget": 0.7,
                                        "ahu_heat_budget_scope": "prestart"}),
    "opt_margin_03": ("M", {}),
    "opt_pair_ledger_margin": ("P8", {**CYCLING, "cycle_vent_ledger": True}),
    "opt_ledger_noheat": ("O7", {**CYCLING, "cycle_vent_ledger": True, "cycle_no_heating": True}),
    "opt_coolmargin_03": ("M", {}),
    "opt_pair_satmix_coolmargin": ("P7", {"sat_cost_model": "economizer"}),
    "opt_rogue_24": ("O10", {"sp_rogue_hours": 24.0}),
    "opt_rogue_24_cold": ("O10", {"sp_rogue_hours": 24.0, "sp_options_max_oat_k": 288.15}),
    "opt_spcap_200": ("O10", {"sp_max_setback_pa": 200.0}),
    "opt_spcap_200_cold": ("O10", {"sp_max_setback_pa": 200.0, "sp_options_max_oat_k": 288.15}),
    # coordinator 18:0x: minimal retune (one combination each) of the most coupled knobs a rule_config can reach;
    # the slow-layer parameters (optimal start cap, lead, heating maximum, guard width) are set hourly by gt_u1b's
    # schedule and cannot be retuned this way
    "opt_rt_ledger": ("O7", {**CYCLING, "cycle_off_k": 0.75, "cycle_vent_ledger": True, "cycle_no_heating": True}),
    "opt_rt_rogue": ("O10", {"sp_rogue_hours": 24.0, "sp_rogue_share": 0.9, "sp_options_max_oat_k": 288.15}),
    "opt_rt_satmix": ("O11", {"sat_cost_model": "economizer", "sat_cost_model_min_oat_k": 288.15,
                              "sat_cost_guard_ignore": 1}),
    "opt_rt_break": ("O1", {"opt_start_continuous": True, "opt_start_break_max_s": 7200.0,
                            "setback_warmup_deadband": "neutral"}),
}
# occupied room margins added to every option of gt_u1b's room tree (rooms.settings heat_margin / cool_margin)
ROOM_MARGINS = {
    "opt_margin_03": {"room_heat_margin_k": 0.3, "room_cool_margin_k": 0.3},
    "opt_pair_ledger_margin": {"room_heat_margin_k": 0.3, "room_cool_margin_k": 0.3},
    "opt_coolmargin_03": {"room_cool_margin_k": 0.3},
    "opt_pair_satmix_coolmargin": {"room_cool_margin_k": 0.3},
}
# measured interaction = joint - (A + B) (cost-weighted changes against gt_u1b)
PAIRS = {
    "P4 warm-up time x capacity": ("opt_pair_break_budgetpre", "opt_break_7200", "opt_budgetpre_070"),
    "P8 cycling x room margin": ("opt_pair_ledger_margin", "opt_ledger_on", "opt_margin_03"),
    "P7 economizer SAT x cooling margin": ("opt_pair_satmix_coolmargin", "opt_satmix_on", "opt_coolmargin_03"),
    "P3 static pressure cap x rogue exclusion": ("opt_sp_combo", "opt_spcap_200", "opt_rogue_24"),
}


def main():
    from smart_building.policies.b2rl import load_model
    base = json.loads(BASE.read_text(encoding="utf-8"))
    base_sha = hashlib.sha256(BASE.read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (code, switches) in LEVELS.items():
        model = copy.deepcopy(base)
        clash = set(switches) & set(model.get("rule_config") or {})
        if clash:
            raise ValueError(f"{name}: {sorted(clash)} already in gt_u1b's rule_config")
        model["rule_config"] = {**(model.get("rule_config") or {}), **switches}
        margins = ROOM_MARGINS.get(name)
        if margins:                                                  # every room option's occupied margins moved
            for option in model["room_policy"]["options"]:
                for key, add in margins.items():
                    option[key] = round(float(option.get(key, 0.0)) + add, 6)
        model.setdefault("source", {})["option"] = {"base": "gt_u1b", "base_sha256": base_sha, "code": code,
                                                    "switches": switches, "room_margins_added": margins or {},
                                                    "catalogue": "docs/plan/option-catalogue-20261103.md"}
        path = OUT / f"model-{name}.json"
        text = json.dumps(model) + "\n"
        if not path.is_file() or path.read_text(encoding="utf-8") != text:    # never rewrite a model in use
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        load_model(path)
        print(name, code, switches)


if __name__ == "__main__":
    main()
