"""Baseline v2 rule-screen candidates as fixed-name classes (reviewer 2026-10-27; docs/plan/baseline-v2.md §2.1.1-2.1.2).

The remote runner loads a policy by file and class name (remote/tasks.py load_policy_class), so every candidate of the
screen that runs on Kaggle is bound here once: A1 (laa_occrow_dampfloor_b2a1) plus one change, checked by the C3
whitelist on construction (c3_configured_class). The keys are the screen's preset names (.tmp/reviewer/rules/
diagnose.py PRESETS, autorun2.py CHANGES), so remote results map onto the local diag files one to one.
"""
from smart_building.policies.g36 import C3_COMBINATIONS, c3_configured_class

A1 = C3_COMBINATIONS["laa_occrow_dampfloor_b2a1"]
SCREEN_CHANGES = {
    "a1_tou": {"precool_until_hour": 12},
    "a1_costsat": {"sat_cost_optimal": True},
    "a1_spcap": {"sp_max_pa": 600.0},
    "a1_spmin": {"sp_max_pa": 200.0},
    "a1_cyc": {"occupied_cycling": True},
    "a1_cycl": {"occupied_cycling": True, "cycle_charge_k": 0.0, "cycle_off_k": 0.5},
    "a1_opt6": {"t_opt_max_s": 21600.0},
    # the cold rule (REQ-022) on A1, and the likely join with the two kept changes (2026-10-27 08:3x)
    "a1_cold3": {"setback_warmup_discharge": "supply_floor_setback", "setback_warmup_deadband": "supply",
                 "setback_release": "zone_hysteresis"},
}
SCREEN_CHANGES["a1_m25+a1_econ+a1_cold3"] = {"occupied_margin_k": 0.25, "economizer_mode": "supply_air",
                                            **SCREEN_CHANGES["a1_cold3"]}

G36PolicyB2STou = c3_configured_class({**A1, **SCREEN_CHANGES["a1_tou"]})
G36PolicyB2SCostSat = c3_configured_class({**A1, **SCREEN_CHANGES["a1_costsat"]})
G36PolicyB2SSpCap = c3_configured_class({**A1, **SCREEN_CHANGES["a1_spcap"]})
G36PolicyB2SSpMin = c3_configured_class({**A1, **SCREEN_CHANGES["a1_spmin"]})
G36PolicyB2SCyc = c3_configured_class({**A1, **SCREEN_CHANGES["a1_cyc"]})
G36PolicyB2SCycL = c3_configured_class({**A1, **SCREEN_CHANGES["a1_cycl"]})
G36PolicyB2SOpt6 = c3_configured_class({**A1, **SCREEN_CHANGES["a1_opt6"]})
G36PolicyB2SCold3 = c3_configured_class({**A1, **SCREEN_CHANGES["a1_cold3"]})
G36PolicyB2SM25EconCold3 = c3_configured_class({**A1, **SCREEN_CHANGES["a1_m25+a1_econ+a1_cold3"]})

SCREEN_CLASSES = {"a1_tou": "G36PolicyB2STou", "a1_costsat": "G36PolicyB2SCostSat", "a1_spcap": "G36PolicyB2SSpCap",
                  "a1_spmin": "G36PolicyB2SSpMin", "a1_cyc": "G36PolicyB2SCyc", "a1_cycl": "G36PolicyB2SCycL",
                  "a1_opt6": "G36PolicyB2SOpt6", "a1_cold3": "G36PolicyB2SCold3",
                  "a1_m25+a1_econ+a1_cold3": "G36PolicyB2SM25EconCold3"}


# Cold-rule base (2026-10-27 10:1x, user: the champion is the cold rule): A1 + the cold rule's REQ-022 switches + one
# change, for the candidates that run on Kaggle (job j12d-b2-c3screen); keys are the screen's names on that base.
COLD3_BASE = {**A1, **SCREEN_CHANGES["a1_cold3"]}
COLD3_SCREEN_CHANGES = {
    "a1_m50": {"occupied_margin_k": 0.5}, "a1_d14": {"discharge_offset_k": 14.0},
    "a1_h30": {"heating_max_ratio": 0.3, "t_opt_max_s": 21600.0}, "a1_sattr": {"sat_oat_low_k": 373.15, "sat_oat_high_k": 373.15},
    "a1_plant2": {"plant_ignore_requests": 2},
    "a1_stop": {"occupied_cycling": True, "cycle_charge_k": 0.0, "cycle_off_k": 0.75, "cycle_window_min": 90},
    "a1_eve": {"evening_charge_min": 120}, "a1_flush": {"night_flush": True}, "a1_sat20": {"sat_max_k": 293.15},
    "a1_opt6": SCREEN_CHANGES["a1_opt6"], "a1_tou": SCREEN_CHANGES["a1_tou"], "a1_spcap": SCREEN_CHANGES["a1_spcap"],
    "a1_cyc": SCREEN_CHANGES["a1_cyc"], "a1_cycl": SCREEN_CHANGES["a1_cycl"], "a1_spmin": SCREEN_CHANGES["a1_spmin"],
}
COLD3_SCREEN_CLASSES = {}
for _key, _change in COLD3_SCREEN_CHANGES.items():
    _class = "G36PolicyB2C3" + "".join(part.capitalize() for part in _key.split("_")[1:])
    globals()[_class] = c3_configured_class({**COLD3_BASE, **_change})
    COLD3_SCREEN_CLASSES[f"a1_cold3+{_key}"] = _class
