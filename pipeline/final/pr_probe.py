"""Probe for the tree-free rule optimisation: (1) can the workspace's schedule builder (search.build_schedule, output
redirected to the scratch folder) reproduce gt_b2's parameter schedule from the autotune end state plus gt_b2's
rule_config? (2) what are the search / holdout scenario sets and do their paths exist?"""
import json
import sys
from pathlib import Path

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
SCR = Path(sys.argv[1]) / "pr"
SCR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "pipeline/setup"))
import search  # noqa: E402

search.OUT = SCR                                   # models go to <scratch>/pr/models, nothing into the workspace
b2 = json.loads((ROOT / "data/runs/b2rl/search/models/model-gt_b2.json").read_text())
end = json.loads((ROOT / "data/runs/b2rl/autotune/dcvnight1/summary.json").read_text())["spec"]
spec = {**end, "rule_config": b2["rule_config"]}
path, _ = search.build_schedule("probe", spec)
m = json.loads(Path(path).read_text())
same = {k: m.get(k) == b2.get(k) for k in ("layers", "features", "parameters", "layout", "rule_config", "ranges", "obs_mean", "obs_var", "log_std", "interval", "rule", "format")}
print("built at", path)
print("equal to gt_b2 per key:", same)
if not same["layers"]:
    import numpy as np
    for i, (a, c) in enumerate(zip(m["layers"], b2["layers"])):
        print(" layer", i, "max |weight diff|", float(np.abs(np.array(a["weight"]) - np.array(c["weight"])).max()), "max |bias diff|", float(np.abs(np.array(a["bias"]) - np.array(c["bias"])).max()))
(SCR / "spec-start.json").write_text(json.dumps(spec, indent=1))
sets = json.loads(search.EVAL_SETS.read_text())
print("eval-set names:", [k for k in sets if not k.startswith("_")])
for name in ("train_breakdown40", "train_ceiling_rest60"):
    s = sets.get(name)
    print(name, type(s).__name__, len(s) if s else None, s[:3] if s else None)
src = Path(search.__file__).read_text()
i = src.index("def run_candidate")
print(src[i:i + 1500])
i = src.index("def ensure_rest60")
print(src[i:i + 900])
