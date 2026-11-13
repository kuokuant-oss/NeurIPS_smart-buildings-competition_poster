"""gt_x6: gt_x5 with the rule version's own 15-option room tree (isolates the 84-option tree).
gt_x7: gt_x5 plus the rule lever 'empty weekday = night' (empty_day_setup: night), i.e. the rule version's levers minus deep-cold keep."""
import copy
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
x5 = json.loads((D / "model-gt_x5.json").read_text())
b2 = json.loads((__import__("pathlib").Path(__file__).resolve().parents[2] / "data/runs/b2rl/search/models/model-gt_b2.json").read_text())
x6 = copy.deepcopy(x5)
x6["room_policy"] = copy.deepcopy(b2["room_policy"])
x7 = copy.deepcopy(x5)
x7["rule_config"]["empty_day_setup"] = "night"
for n, m in (("gt_x6", x6), ("gt_x7", x7)):
    m["source"]["reviewer_final_20261112"] = n
    (D / f"model-{n}.json").write_text(json.dumps(m, default=float))
    print(n, {k: v for k, v in m["rule_config"].items() if b2["rule_config"].get(k, "<none>") != v}, [k for k in b2["rule_config"] if k not in m["rule_config"]],
          len(m["room_policy"]["options"]), [(g["id"], g["on"]) for g in m["rule_oat_gates"]])
