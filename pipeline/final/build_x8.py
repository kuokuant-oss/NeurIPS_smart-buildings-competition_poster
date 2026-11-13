"""gt_x8: gt_x6 plus the rule lever 'empty weekday = night' (= gt_x7 with the rule version's own room tree).
Completes the 2 x 2: {84-option room tree yes/no} x {empty weekday = night yes/no}, all with the night-setback gate at 3 K."""
import copy
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
x6 = json.loads((D / "model-gt_x6.json").read_text())
x7 = json.loads((D / "model-gt_x7.json").read_text())
x8 = copy.deepcopy(x6)
x8["rule_config"]["empty_day_setup"] = "night"
x8["source"]["reviewer_final_20261112"] = "gt_x8"
(D / "model-gt_x8.json").write_text(json.dumps(x8, default=float))
print("rule_config == gt_x7's:", x8["rule_config"] == x7["rule_config"], "| gates == gt_x7's:", x8["rule_oat_gates"] == x7["rule_oat_gates"],
      "| room tree == gt_x6's:", x8["room_policy"] == x6["room_policy"], "| options", len(x8["room_policy"]["options"]))
