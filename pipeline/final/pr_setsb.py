"""User decision 2026-11-12 14:3x: the tree-free rule carries night setback 3 K as a plain rule setting (all climates).
Adds it to the optimisation state so that the remaining tuning happens with it in place."""
import json, sys
from pathlib import Path
f = Path(sys.argv[1]) / "pr" / "state.json"
s = json.loads(f.read_text())
print("before:", s["last"], "| setback:", s["spec"]["rule_config"].get("setback_heat_drop_k"), "| no_reheat present:", "no_reheat_oat_k" in s["spec"]["rule_config"])
s["spec"]["rule_config"]["setback_heat_drop_k"] = 3.0
s["last"] = "add setback_3k (user decision, not by the acceptance test)"
f.write_text(json.dumps(s, indent=1))
(f.parent / "kept.jsonl").open("a").write(json.dumps({"label": "user", "move": "add setback_3k", "note": "user decision 2026-11-12; search -0.19% (t -1.0), not tested on holdout"}) + "\n")
print("after: setback", s["spec"]["rule_config"]["setback_heat_drop_k"], "| rule_config keys", len(s["spec"]["rule_config"]))
