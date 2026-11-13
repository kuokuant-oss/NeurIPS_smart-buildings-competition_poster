"""Coordinate search over the outside-air schedule's parameters, v2 (2026-10-29; side line "最佳化"; user 13:0x:
"先大規模自動化後才要一次驗證這麼多有點風險，可能中間就要先驗一遍看看自動重調參數的邏輯是不是正確的").

v1 (pipeline/setup/autotune_v1.py) failed the known-answer test pilot1 at 14:55: from f2d_a12_g05 with the optimal start
cap at 3 h and the lead at 120 min it stopped where it started (TRAIN 100 -13.57% vs the rule, against -15.14% for the
known answer 1 h / 180 min), because it moved one step at a time and every step had to improve the holdout set: 3 h ->
2 h is flat on the holdout (-13.33% -> -13.15%) and the gain only comes at 1 h, where the lead starts to matter.

The state is a schedule spec of pipeline/setup/search.py (bands cold / mild / hot, breaks_c, layout, rule_config); every
tuned parameter must be written in all three bands. For one parameter at a time, a line search evaluates the values
1..reach steps on each side of the current one (the same shift in every band for "all", or in one band) on the fast
simulator over train_breakdown40 (the search set) and takes the best. It is kept only if (1) on the search set the mean
cost falls by at least MIN_GAIN (relative) and the paired difference is at least T_SEARCH standard errors, and (2) on
train_ceiling_rest60 (the other 60 TRAIN scenarios, never searched on) the mean cost does not rise by more than
HOLDOUT_TOL (relative): the holdout set guards against moves that only fit the search set. A kept value starts a new line
search around it; a parameter is left when its line search keeps nothing. Sweeps repeat over the parameters until one
keeps nothing. Specs already evaluated (or turned down by the holdout set) are not evaluated again. A coordinate search
cannot find a gain that needs two parameters to move together while neither helps alone (a degenerate fake landscape
of the 2026-10-29 mock tests); the known-answer pilots check that the real landscape is not like that. Every evaluation and decision is a line of
data/runs/b2rl/autotune/<tag>/log.jsonl, the state after each kept move is state.json, and the end writes summary.json
(with the end state against the start on the holdout set) and a "ROUND autotune <tag> done" line in
data/runs/b2rl/search/run.log (the hourly watcher wakes on it). Costs are mean cost over scenarios; "vs rule" is against
the original rule G36PolicyB2R2 (data/runs/b2rl/ceiling/rule.json).

python3 -B pipeline/launch/launch.py pipeline/setup/autotune.py <plan.json> [--workers 16]
plan: {"tag": "...", "start": {"round": "data/runs/b2rl/search/round14.json", "name": "f2d_a12_g05",
                                "set": {"<band>": {"<parameter>": value}, ...}},
       "start_rows": "<search candidate whose train_ceiling100 rows are exactly the start's>" (optional),
       "moves": [{"parameter": "t_opt_max_s", "bands": "all", "step": 3600, "low": 3600, "high": 21600,
                  "reach": 2}, ...],
       "sweeps": 2}"""
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
MIN_GAIN = 0.001          # relative fall of the search set's mean cost a kept value needs
T_SEARCH = 1.5            # and its paired difference in standard errors
HOLDOUT_TOL = 0.001       # the holdout set's mean cost may not rise by more than this (relative)
BANDS = ("cold", "mild", "hot")


def bands_of(spec):
    """The spec's bands: its levels for a knots schedule (search.build_knots, 2026-10-30), else cold / mild / hot."""
    return tuple(dict.fromkeys(spec["levels"])) if "knots_c" in spec else BANDS


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


def shifted(spec, move, steps):
    """The spec with move's parameter shifted `steps` steps (negative = down), or None if no band changes."""
    out = json.loads(json.dumps(spec))
    bands = bands_of(spec) if move["bands"] == "all" else \
        (tuple(move["bands"]) if isinstance(move["bands"], list) else (move["bands"],))
    changed = False
    for band in bands:
        old = out[band][move["parameter"]]
        new = min(max(old + steps * move["step"], move["low"]), move["high"])
        new = int(round(new)) if move["parameter"] == "unoccupied_lead_min" else round(new, 6)
        changed |= new != old
        out[band][move["parameter"]] = new
    return out if changed else None


def key(spec):
    return json.dumps(spec, sort_keys=True)


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
        self.seen = {}                                     # spec key -> search rows (no spec is evaluated twice)
        self.holdout_rejected = set()                      # spec keys the holdout set already turned down
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
        given = self.plan.get("start_rows")
        if isinstance(given, dict):                        # {"search": name, "holdout": name}: the subsets' own rows
            rows = {part: json.loads((search.OUT / (SEARCH_SET if part == "search" else HOLDOUT_SET)
                                      / f"{name}.json").read_text(encoding="utf-8"))["rows"]
                    for part, name in given.items()}
            start_name = given["search"]
        elif given:
            full = json.loads((search.OUT / "train_ceiling100" / f"{given}.json").read_text(encoding="utf-8"))["rows"]
            rows = {part: {k: full[k] for k in keys} for part, keys in self.keys.items()}
            start_name = given
        else:
            self.n += 1
            start_name, search_rows = self.evaluate(spec, "search")
            _, holdout_rows = self.evaluate(spec, "holdout")
            rows = {"search": search_rows, "holdout": holdout_rows}
        self.seen[key(spec)] = (rows["search"], start_name)
        self.names = {"search": start_name, "holdout": start_name if not isinstance(given, dict) else given["holdout"]}
        start_rows, start = rows, {part: self.vs_rule(rows[part], part) for part in rows}
        self.record({"n": 0, "event": "start", "name": start_name, "spec": spec, "vs_rule": start})
        search.log(f"AUTOTUNE {self.tag}: start {start_name}: {100 * start['search']:+.2f}% (search 40) "
                   f"{100 * start['holdout']:+.2f}% (holdout 60) vs rule")
        kept = 0
        for sweep in range(int(self.plan.get("sweeps", 2))):
            kept_this_sweep = 0
            for move in self.plan["moves"]:
                while True:                                # line searches until one keeps nothing
                    result = self.line_search(spec, rows, move, sweep)
                    if result is None:
                        break
                    spec, rows, self.names = result
                    kept += 1
                    kept_this_sweep += 1
                    (self.dir / "state.json").write_text(json.dumps({"n": self.n, "spec": spec}, indent=1) + "\n",
                                                         encoding="utf-8")
            if not kept_this_sweep:
                break
        return self.finish(spec, rows, start_rows, start, "converged" if not kept_this_sweep else "sweeps done", kept)

    def line_search(self, spec, rows, move, sweep):
        """The best value 1..reach steps either side on the search set; (spec, rows) if it is kept, else None."""
        reach = int(move.get("reach", 2))
        tried = []
        for steps in [s for k in range(1, reach + 1) for s in (k, -k)]:
            candidate = shifted(spec, move, steps)
            if candidate is None or key(candidate) in self.seen:
                if candidate is not None:
                    tried.append((candidate, *self.seen[key(candidate)]))
                continue
            self.n += 1
            name, search_rows = self.evaluate(candidate, "search")
            self.seen[key(candidate)] = (search_rows, name)
            tried.append((candidate, search_rows, name))
            self.record({"n": self.n, "sweep": sweep, "name": name, "parameter": move["parameter"],
                         "bands": move["bands"],
                         "values": {b: candidate[b][move["parameter"]] for b in bands_of(candidate)},
                         "search_vs_rule": round(self.vs_rule(search_rows, "search"), 6)})
        if not tried:
            return None
        keys = self.keys["search"]
        candidate, search_rows, name = min(tried, key=lambda t: mean_cost(t[1], keys))
        diff, se = paired(search_rows, rows["search"], keys)
        gain = 1 - mean_cost(search_rows, keys) / mean_cost(rows["search"], keys)
        values = {b: candidate[b][move["parameter"]] for b in bands_of(candidate)}
        stats = {"best": name, "values": values, "search_vs_rule": round(self.vs_rule(search_rows, "search"), 6),
                 "search_gain": round(gain, 6), "search_t": round(-diff / se, 2) if se > 0 else None}
        decision = "kept"
        if not (gain >= MIN_GAIN and se > 0 and -diff / se >= T_SEARCH):
            decision = "nothing better on the search set"
        elif key(candidate) in self.holdout_rejected:
            decision = "rejected on the holdout set (earlier)"
        else:
            self.n += 1
            hname, holdout_rows = self.evaluate(candidate, "holdout")
            hkeys = self.keys["holdout"]
            hdiff, hse = paired(holdout_rows, rows["holdout"], hkeys)
            hgain = 1 - mean_cost(holdout_rows, hkeys) / mean_cost(rows["holdout"], hkeys)
            stats.update(holdout_vs_rule=round(self.vs_rule(holdout_rows, "holdout"), 6),
                         holdout_gain=round(hgain, 6), holdout_t=round(-hdiff / hse, 2) if hse > 0 else None)
            if hgain < -HOLDOUT_TOL:
                decision = "rejected on the holdout set"
                self.holdout_rejected.add(key(candidate))
        self.record({"n": self.n, "sweep": sweep, "event": "line search", "parameter": move["parameter"],
                     "bands": move["bands"], "decision": decision, **stats})
        search.log(f"AUTOTUNE {self.tag} {self.n:03d}: {move['parameter']} {move['bands']} best {values}: {decision} "
                   f"({json.dumps({k: v for k, v in stats.items() if k not in ('values', 'best')})})")
        if decision != "kept":
            return None
        return candidate, {"search": search_rows, "holdout": holdout_rows}, {"search": name, "holdout": hname}

    def finish(self, spec, rows, start_rows, start, why, kept):
        end = {part: self.vs_rule(rows[part], part) for part in rows}
        hdiff, hse = paired(rows["holdout"], start_rows["holdout"], self.keys["holdout"])
        full = {**rows["search"], **rows["holdout"]}
        keys = self.keys["search"] + self.keys["holdout"]
        rule100 = (self.rule["search"] * len(self.keys["search"]) + self.rule["holdout"] * len(self.keys["holdout"])) \
            / len(keys)
        summary = {"tag": self.tag, "stopped": why, "kept": kept, "evaluations": self.n, "start_vs_rule": start,
                   "end_vs_rule": end, "end_vs_rule_train100": mean_cost(full, keys) / rule100 - 1,
                   "holdout_end_vs_start_t": round(-hdiff / hse, 2) if hse > 0 else None,
                   "rows_names": self.names, "spec": spec}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        search.log(f"ROUND autotune {self.tag} done ({why}, {kept} kept, {self.n} evaluations): search 40 "
                   f"{100 * start['search']:+.2f}% -> {100 * end['search']:+.2f}%, holdout 60 "
                   f"{100 * start['holdout']:+.2f}% -> {100 * end['holdout']:+.2f}% "
                   f"({summary['holdout_end_vs_start_t']} SE), TRAIN 100 "
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
