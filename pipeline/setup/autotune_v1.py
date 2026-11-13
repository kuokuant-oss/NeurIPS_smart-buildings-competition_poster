"""[v1, kept for the record: failed the 2026-10-29 14:55 known-answer test pilot1 (one step at a time, holdout had to improve on every step); superseded by autotune.py v2] Coordinate search over the outside-air schedule's parameters (2026-10-29; side line "最佳化"; user 13:0x:
"先大規模自動化後才要一次驗證這麼多有點風險，可能中間就要先驗一遍看看自動重調參數的邏輯是不是正確的").

The state is a schedule spec of pipeline/setup/search.py (bands cold / mild / hot, breaks_c, layout, rule_config); every
tuned parameter must be written in all three bands. A move shifts one parameter by one step, in every band ("all",
the same shift in each band) or in one band, and is evaluated on the fast simulator over train_breakdown40 (the search
set). A move is kept only if (1) on the search set the mean cost falls by at least MIN_GAIN (relative) and the paired
difference is at least T_SEARCH standard errors, and (2) on train_ceiling_rest60 (the other 60 TRAIN scenarios, never
searched on) the mean cost also falls. A kept move is tried again in the same direction; a parameter is left when a
move in both directions has been rejected (or the bound is reached). Every evaluation and decision is a line of
data/runs/b2rl/autotune/<tag>/log.jsonl, the state after each kept move is state.json, and the end writes summary.json
and a "ROUND autotune <tag> done" line in data/runs/b2rl/search/run.log (the hourly watcher wakes on it). Costs are
mean cost over scenarios; "vs rule" is against the original rule G36PolicyB2R2 (data/runs/b2rl/ceiling/rule.json).

python3 -B pipeline/launch/launch.py pipeline/setup/autotune.py <plan.json> [--workers 16]
plan: {"tag": "...", "start": {"round": "data/runs/b2rl/search/round14.json", "name": "f2d_a12_g05",
                                "set": {"<band>": {"<parameter>": value}, ...}},
       "start_rows": "<search candidate whose train_ceiling100 rows are exactly the start's>" (optional),
       "moves": [{"parameter": "t_opt_max_s", "bands": "all", "step": 3600, "low": 3600, "high": 21600}, ...],
       "sweeps": 1, "max_accepted": null}"""
import argparse
import json
import math
from pathlib import Path
import statistics as st
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search  # noqa: E402  (build_schedule, run_candidate, log, EVAL_SETS)

OUT = ROOT / "data/runs/b2rl/autotune"
SEARCH_SET, HOLDOUT_SET = "train_breakdown40", "train_ceiling_rest60"
MIN_GAIN = 0.001          # relative fall of the search set's mean cost a kept move needs
T_SEARCH = 1.5            # and its paired difference in standard errors
BANDS = ("cold", "mild", "hot")


def mean_cost(rows, keys):
    return st.mean(rows[k]["cost"] for k in keys)


def paired(new, old, keys):
    d = [new[k]["cost"] - old[k]["cost"] for k in keys]
    se = st.stdev(d) / math.sqrt(len(d))
    return st.mean(d), se


def start_spec(plan):
    start = plan["start"]
    spec = json.loads(json.dumps(json.loads((ROOT / start["round"]).read_text(encoding="utf-8"))[start["name"]]))
    for band, values in start.get("set", {}).items():
        spec[band] = {**spec[band], **values}
    return spec


def shifted(spec, move, direction):
    """The spec with move's parameter shifted one step in direction, or None at the bound in every band."""
    out = json.loads(json.dumps(spec))
    bands = BANDS if move["bands"] == "all" else (move["bands"],)
    changed = False
    for band in bands:
        old = out[band][move["parameter"]]
        new = min(max(old + direction * move["step"], move["low"]), move["high"])
        new = int(round(new)) if move["parameter"] == "unoccupied_lead_min" else round(new, 6)
        changed |= new != old
        out[band][move["parameter"]] = new
    return out if changed else None


class Tuner:
    def __init__(self, plan, workers):
        self.plan, self.workers, self.tag = plan, workers, plan["tag"]
        self.dir = OUT / self.tag
        self.dir.mkdir(parents=True, exist_ok=True)
        sets = json.loads(search.EVAL_SETS.read_text(encoding="utf-8"))
        self.keys = {"search": sets[SEARCH_SET], "holdout": sets[HOLDOUT_SET]}
        rule = json.loads((search.CEILING / "rule.json").read_text(encoding="utf-8"))["rows"]
        self.rule = {part: mean_cost(rule, keys) for part, keys in self.keys.items()}
        self.n = 0
        self.rejected = set()                              # specs already rejected: not evaluated again
        for line in self.lines():                          # a restarted tuner keeps counting
            self.n = max(self.n, line.get("n", 0))

    def lines(self):
        path = self.dir / "log.jsonl"
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] \
            if path.exists() else []

    def record(self, entry):
        with (self.dir / "log.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"time": time.strftime("%m-%d %H:%M:%S"), **entry}) + "\n")

    def evaluate(self, spec, part):
        name = f"at_{self.tag}_{self.n:03d}"
        path, _ = search.build_schedule(name, spec)
        subset = SEARCH_SET if part == "search" else HOLDOUT_SET
        return name, search.run_candidate(name, path, subset, self.workers)

    def vs_rule(self, rows, part):
        return mean_cost(rows, self.keys[part]) / self.rule[part] - 1

    def run(self):
        spec = start_spec(self.plan)
        if self.plan.get("start_rows"):
            full = json.loads((search.OUT / "train_ceiling100" / f"{self.plan['start_rows']}.json")
                              .read_text(encoding="utf-8"))["rows"]
            rows = {part: {k: full[k] for k in keys} for part, keys in self.keys.items()}
            start_name = self.plan["start_rows"]
        else:
            self.n += 1
            start_name, search_rows = self.evaluate(spec, "search")
            _, holdout_rows = self.evaluate(spec, "holdout")
            rows = {"search": search_rows, "holdout": holdout_rows}
        start = {part: self.vs_rule(rows[part], part) for part in rows}
        self.record({"n": 0, "event": "start", "name": start_name, "spec": spec,
                     "vs_rule": start})
        search.log(f"AUTOTUNE {self.tag}: start {start_name}: {100 * start['search']:+.2f}% (search 40) "
                   f"{100 * start['holdout']:+.2f}% (holdout 60) vs rule")
        accepted = 0
        for sweep in range(int(self.plan.get("sweeps", 1))):
            kept_this_sweep = 0
            for move in self.plan["moves"]:
                for direction in (1, -1):
                    moved = False
                    while True:
                        candidate = shifted(spec, move, direction)
                        if candidate is None or json.dumps(candidate, sort_keys=True) in self.rejected:
                            break
                        self.n += 1
                        decision, new_rows, name, stats = self.try_move(candidate, rows)
                        values = {b: candidate[b][move["parameter"]] for b in BANDS}
                        self.record({"n": self.n, "sweep": sweep, "name": name, "parameter": move["parameter"],
                                     "bands": move["bands"], "direction": direction, "values": values,
                                     "decision": decision, **stats})
                        search.log(f"AUTOTUNE {self.tag} {self.n:03d}: {move['parameter']} {move['bands']} -> "
                                   f"{values}: {decision} ({json.dumps(stats)})")
                        if decision != "kept":
                            self.rejected.add(json.dumps(candidate, sort_keys=True))
                            break
                        spec, rows, moved = candidate, new_rows, True
                        accepted += 1
                        kept_this_sweep += 1
                        (self.dir / "state.json").write_text(json.dumps({"n": self.n, "spec": spec}, indent=1)
                                                             + "\n", encoding="utf-8")
                        if self.plan.get("max_accepted") and accepted >= self.plan["max_accepted"]:
                            return self.finish(spec, rows, start, "max_accepted")
                    if moved:
                        break
            if not kept_this_sweep:
                break
        return self.finish(spec, rows, start, "converged" if not kept_this_sweep else "sweeps done")

    def try_move(self, candidate, rows):
        name, search_rows = self.evaluate(candidate, "search")
        keys = self.keys["search"]
        diff, se = paired(search_rows, rows["search"], keys)
        gain = 1 - mean_cost(search_rows, keys) / mean_cost(rows["search"], keys)
        stats = {"search_vs_rule": round(self.vs_rule(search_rows, "search"), 6), "search_gain": round(gain, 6),
                 "search_t": round(-diff / se, 2) if se > 0 else None}
        if not (gain >= MIN_GAIN and se > 0 and -diff / se >= T_SEARCH):
            return "rejected on the search set", None, name, stats
        _, holdout_rows = self.evaluate(candidate, "holdout")
        keys = self.keys["holdout"]
        diff, se = paired(holdout_rows, rows["holdout"], keys)
        stats.update(holdout_vs_rule=round(self.vs_rule(holdout_rows, "holdout"), 6),
                     holdout_gain=round(1 - mean_cost(holdout_rows, keys) / mean_cost(rows["holdout"], keys), 6),
                     holdout_t=round(-diff / se, 2) if se > 0 else None)
        if diff >= 0:
            return "rejected on the holdout set", None, name, stats
        return "kept", {"search": search_rows, "holdout": holdout_rows}, name, stats

    def finish(self, spec, rows, start, why):
        end = {part: self.vs_rule(rows[part], part) for part in rows}
        full = {**rows["search"], **rows["holdout"]}
        keys = self.keys["search"] + self.keys["holdout"]
        rule100 = (self.rule["search"] * len(self.keys["search"]) + self.rule["holdout"] * len(self.keys["holdout"])) \
            / len(keys)
        summary = {"tag": self.tag, "stopped": why, "evaluations": self.n, "start_vs_rule": start, "end_vs_rule": end,
                   "end_vs_rule_train100": mean_cost(full, keys) / rule100 - 1, "spec": spec}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        search.log(f"ROUND autotune {self.tag} done ({why}, {self.n} evaluations): search 40 "
                   f"{100 * start['search']:+.2f}% -> {100 * end['search']:+.2f}%, holdout 60 "
                   f"{100 * start['holdout']:+.2f}% -> {100 * end['holdout']:+.2f}%, TRAIN 100 "
                   f"{100 * summary['end_vs_rule_train100']:+.2f}% vs rule")
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    Tuner(json.loads((ROOT / args.plan).read_text(encoding="utf-8")), args.workers).run()


if __name__ == "__main__":
    main()
