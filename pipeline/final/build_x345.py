"""Variants of gt_x1 suggested by the public notebooks' leaderboard ladder (scratch only).
  gt_x3: gt_x1 with the night-setback gate at 3 K instead of 2 K.
  gt_x4: gt_x1 with the empty-weekday confirmation delay 30 min instead of 60 min.
  gt_x5: both."""
import copy
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
x1 = json.loads((D / "model-gt_x1.json").read_text())


def sb3(m):
    m = copy.deepcopy(m)
    m["rule_config"]["setback_heat_drop_k"] = 3.0
    for g in m["rule_oat_gates"]:
        if g["id"] == "o508_2":
            g["on"] = {"setback_heat_drop_k": 3.0}
    return m


def ed30(m):
    m = copy.deepcopy(m)
    assert m["rule_config"]["empty_day_delay_min"] == 60
    m["rule_config"]["empty_day_delay_min"] = 30
    return m


for n, m in (("gt_x3", sb3(x1)), ("gt_x4", ed30(x1)), ("gt_x5", ed30(sb3(x1)))):
    m["source"]["reviewer_final_20261112"] = n
    (D / f"model-{n}.json").write_text(json.dumps(m, default=float))
    print(n, m["rule_config"]["setback_heat_drop_k"], m["rule_config"]["empty_day_delay_min"], [(g["id"], g["on"]) for g in m["rule_oat_gates"]])
