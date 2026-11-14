"""Compare the validation results of this run (data/final/runs_p2, runs_v2) with the recorded ones behind the paper
(data/final/recorded/runs_p2, runs_v2): per controller, scenario by scenario, the cost and its three parts.
python3 -B pipeline/final/compare_results.py [--tolerance 1e-9]"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW, REC = ROOT / "data/final", ROOT / "data/final/recorded"
PARTS = ("cost", "comfort_part", "energy_part", "carbon_part")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tolerance", type=float, default=1e-8,
                   help="relative difference counted as equal (floating-point noise between machines is ~1e-9)")
    args = p.parse_args()
    total = same = missing = 0
    worst = (0.0, "")
    for rec in sorted(REC.glob("runs_*/*.json")):
        new = NEW / rec.parent.name / rec.name
        rows = json.loads(rec.read_text())["rows"]
        got = json.loads(new.read_text())["rows"] if new.is_file() else {}
        n_same = n_missing = 0
        for key, row in rows.items():
            total += 1
            if key not in got:
                missing += 1
                n_missing += 1
                continue
            rel = max(abs(got[key][k] - row[k]) / max(abs(row[k]), 1e-12) for k in PARTS if k in row)
            if rel <= args.tolerance:
                same += 1
                n_same += 1
            if rel > worst[0]:
                worst = (rel, f"{rec.parent.name}/{rec.name} {key}")
        print(f"{rec.parent.name}/{rec.stem:<28} {n_same:>4} equal, {len(rows) - n_same - n_missing:>4} different, "
              f"{n_missing:>4} not run")
    print(f"TOTAL {total} recorded episodes: {same} equal within {args.tolerance:g}, {missing} not run; "
          f"largest relative difference {worst[0]:.3g} ({worst[1]})")


if __name__ == "__main__":
    main()
