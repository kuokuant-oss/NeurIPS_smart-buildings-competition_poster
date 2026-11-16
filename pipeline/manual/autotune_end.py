"""End state of an autotune run as a round file: data/runs/b2rl/search/autotune-<tag>-end.json = {<name>: <spec>},
where <name> is the run whose holdout rows belong to the end state. pipeline/setup/overnight.py writes this file
itself after its INNER check; for the short run (short1, run by autotune.py alone) it was written by hand in the same
form, and the overnight run starts from it.

    python3 -B pipeline/manual/autotune_end.py short1"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    tag = sys.argv[1]
    summary = json.loads((ROOT / f"data/runs/b2rl/autotune/{tag}/summary.json").read_text(encoding="utf-8"))
    name = summary["rows_names"]["holdout"]
    out = ROOT / f"data/runs/b2rl/search/autotune-{tag}-end.json"
    out.write_text(json.dumps({name: summary["spec"]}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({name})")


if __name__ == "__main__":
    main()
