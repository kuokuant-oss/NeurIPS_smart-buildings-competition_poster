"""Tree-free rule optimisation (reviewer, 2026-11-12). Same method and acceptance rule as the workspace's autotune
(pipeline/setup/autotune.py, 2026-10-29): one change at a time on the fast simulator; a change is kept only if on the search
set (train_breakdown40) the mean cost falls by >= 0.1 % and the paired difference is >= 1.5 standard errors, and on the
holdout set (train_ceiling_rest60) the mean cost does not rise by more than 0.1 %.
Start: gt_b2's rule (parameter schedule + rule_config) with NO room policy (no tree). Candidates:
  phase A  rule options: drop one group added after the tree was introduced; add a building-wide room setting; setback depth
  phase B  the outside-air parameter schedule (the moves of autotune-night1.json, one step each side)
Everything is written under <scratch>/pr; the workspace is only read.
python3 -B pipeline/launch/launch.py pr_opt.py <scratch> <workers> [A|B|AB]"""
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/kaggle"
SCEN = ROOT / "data/processed/scenarios/split-v1"
SCR = Path(sys.argv[1]) / "pr"
MIN_GAIN, T_SEARCH, HOLDOUT_TOL = 0.001, 1.5, 0.001
SEARCH, HOLDOUT = "train_breakdown40", "train_ceiling_rest60"

GROUPS = {   # rule options added after 2026-10-30 (the last tree-free tuning); ahu_vot_check (ventilation code) is not touched
    "freeze_min_airflow": ["freeze_min_airflow", "freeze_coldest_only"],
    "pump_dp_reset": ["pump_dp_reset"],
    "sp_smooth_alpha": ["sp_smooth_alpha"],
    "return_air_area_weighted": ["return_air_area_weighted"],
    "inversion_supply_air": ["inversion_supply_air"],
    "low_load_max_loop": ["low_load_max_loop"],
    "sat_rate_limit": ["sat_rate_limit_k", "sat_guard_hysteresis_k"],
    "sat_cost_fan_observed": ["sat_cost_fan_observed"],
    "night_flush": ["night_flush", "night_flush_min_range_k", "night_flush_min_max_k"],
    "opt_start_continuous": ["opt_start_continuous"],
    "no_reheat": ["no_reheat_oat_k", "no_reheat_min_oat24_k"],
    "daily_layer": ["daily_layer", "daily_min_oat24_k"],
    "economic_target": ["economic_target_k", "economic_keep_cold_k"],
    "economic_keep_cold": ["economic_keep_cold_k"],
    "cold_low_flow": ["cold_low_flow_oat_k", "cold_low_flow_sp_pa", "cold_low_flow_deadband_only"],
    "big_room": ["big_room_heat_max_ratio", "big_room_need"],
    "empty_day_setup": ["empty_day_setup"],
}
ADDS = {     # building-wide (every room the same) versions of what the room tree chooses per room, and two rule doses
    "all_rooms_62_1_minimum": {"zone_min_ra_lps_m2": 0.3, "zone_min_rp_lps_m2": 0.125, "zone_min_oa_divide": True},
    "all_rooms_wall_heat_max": {"zone_heat_wall_ref_m": 20.0, "zone_heat_wall_power": 1.0},
    "setback_1k": {"setback_heat_drop_k": 1.0},
    "setback_2k": {"setback_heat_drop_k": 2.0},
    "setback_3k": {"setback_heat_drop_k": 3.0},
    "empty_day_delay_30": {"empty_day_delay_min": 30},
    "empty_day_delay_120": {"empty_day_delay_min": 120},
}


def log(text):
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with (SCR / "log.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _episode(args):
    name, spec_key = args
    os.environ["SB_FASTSIM"] = "rom"
    os.environ["SB_FASTSIM_NO_BUILD"] = "1"
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    for p in (str(ROOT / "packages"), str(DATA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from smart_building.learned.b2rl_env import setup_simulator
    setup_simulator()                      # the project's own setup for the generated scenes (cwd reference/starter-kit)
    from smart_building.evaluation.episode import run_episode
    from smart_building.policies.b2rl import b2rl_policy_class
    t0 = time.perf_counter()
    try:
        pool_name, building, weather = spec_key.split("/")
        base = ROOT / "data/processed/scenarios/split-v1-r1/train" if pool_name == "train" else SCEN / pool_name
        r = run_episode(base / building / weather, b2rl_policy_class(SCR / "models" / f"model-{name}.json"), 8640, policy_key=name, label=spec_key)
    except Exception as error:  # noqa: BLE001
        return name, spec_key, {"error": repr(error)[:300]}
    return name, spec_key, {"cost": -float(r["mean_reward"]), "comfort_part": float(r["comfort_part"]) / int(r["steps"]),
                            "elapsed": time.perf_counter() - t0}


class Opt:
    def __init__(self, workers):
        sys.path.insert(0, str(ROOT / "packages"))
        sys.path.insert(0, str(ROOT / "pipeline/setup"))
        import search
        search.OUT = SCR
        self.search = search
        self.workers = workers
        self.sets = {k: [s for s in v if "minneap" not in s.lower()] for k, v in json.loads(search.EVAL_SETS.read_text()).items()
                     if isinstance(v, list)}
        log(f"sets: search {len(self.sets[SEARCH])} scenes, holdout {len(self.sets[HOLDOUT])} scenes (Minneapolis excluded)")
        self.rows_file = SCR / "rows.json"
        self.rows = json.loads(self.rows_file.read_text()) if self.rows_file.is_file() else {}

    def key(self, spec):
        return "s" + hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]

    def evaluate(self, specs, subset):
        """rows[key][scene] for every spec on the subset; missing episodes run together on one pool."""
        todo = []
        for spec in specs:
            k = self.key(spec)
            self.rows.setdefault(k, {})
            if not (SCR / "models" / f"model-{k}.json").is_file():
                path, _ = self.search.build_schedule(k, spec)
                m = json.loads(Path(path).read_text())
                assert not m.get("room_policy") and not m.get("rule_oat_gates"), "tree-free rule only"
            todo += [(k, s) for s in self.sets[subset] if s not in self.rows[k]]
        if todo:
            log(f"  {len(todo)} episodes ({len(specs)} candidates on {subset})")
            with mp.get_context("spawn").Pool(self.workers, maxtasksperchild=1) as pool:
                for n, (name, s, row) in enumerate(pool.imap_unordered(_episode, todo), 1):
                    if "error" in row:
                        log(f"  ERROR {name} {s}: {row['error']}")
                        raise SystemExit(1)
                    self.rows[name][s] = row
                    if n % 40 == 0 or n == len(todo):
                        self.rows_file.write_text(json.dumps(self.rows))
        return [self.rows[self.key(spec)] for spec in specs]

    def compare(self, new, old, subset):
        keys = self.sets[subset]
        d = [new[k]["cost"] - old[k]["cost"] for k in keys]
        base = st.mean(old[k]["cost"] for k in keys)
        se = st.stdev(d) / math.sqrt(len(d))
        return st.mean(d) / base, (st.mean(d) / se if se > 0 else 0.0)

    def greedy(self, state, make_candidates, label):
        """Repeat: evaluate the candidates around the state on the search set, take the best that passes the search
        test and the holdout test, move there. Candidates that did not help are dropped for the rest of the phase."""
        dead = set()
        while True:
            cands = {n: s for n, s in make_candidates(state).items() if n not in dead}
            if not cands:
                break
            base = self.evaluate([state], SEARCH)[0]
            rows = self.evaluate(list(cands.values()), SEARCH)
            scored = []
            for (n, s), r in zip(cands.items(), rows):
                rel, t = self.compare(r, base, SEARCH)
                scored.append((rel, t, n, s, r))
                log(f"  [{label}] {n}: search {100 * rel:+.3f}% (t {t:+.1f})")
                if rel > -MIN_GAIN / 2:
                    dead.add(n)
            kept = False
            for rel, t, n, s, r in sorted(scored, key=lambda x: x[0]):
                if not (rel <= -MIN_GAIN and t <= -T_SEARCH):
                    break
                hb, hn = self.evaluate([state, s], HOLDOUT)
                hrel, ht = self.compare(hn, hb, HOLDOUT)
                ok = hrel <= HOLDOUT_TOL
                log(f"  [{label}] {n}: holdout {100 * hrel:+.3f}% (t {ht:+.1f}) -> {'KEPT' if ok else 'turned down'}")
                if not ok:
                    dead.add(n)          # a kept move keeps its name alive: the next step in the same direction is tried
                if ok:
                    state, kept = s, True
                    (SCR / "state.json").write_text(json.dumps({"spec": state, "last": n, "label": label}, indent=1))
                    (SCR / "kept.jsonl").open("a").write(json.dumps({"label": label, "move": n, "search": rel, "search_t": t, "holdout": hrel, "holdout_t": ht}) + "\n")
                    break
            if not kept:
                break
        return state


def option_candidates(spec):
    out, rc = {}, spec["rule_config"]
    for name, keys in GROUPS.items():
        if any(k in rc for k in keys):
            out[f"drop {name}"] = {**spec, "rule_config": {k: v for k, v in rc.items() if k not in keys}}
    for name, add in ADDS.items():
        if any(rc.get(k) != v for k, v in add.items()):
            out[f"add {name}"] = {**spec, "rule_config": {**rc, **add}}
    return out


def schedule_candidates(spec, moves):
    out = {}
    for mv in moves:
        bands = ("cold", "mild", "hot") if mv["bands"] == "all" else (mv["bands"],)
        for sign in (-1, 1):
            new = json.loads(json.dumps(spec))
            ok = True
            for b in bands:
                v = new[b][mv["parameter"]] + sign * mv["step"]
                if not mv["low"] - 1e-9 <= v <= mv["high"] + 1e-9:
                    ok = False
                new[b][mv["parameter"]] = v
            if ok:
                out[f"{mv['parameter']} {mv['bands']} {'+' if sign > 0 else '-'}{mv['step']}"] = new
    return out


if __name__ == "__main__":
    workers, phases = int(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "AB")
    SCR.mkdir(parents=True, exist_ok=True)
    opt = Opt(workers)
    sf = SCR / "state.json"
    state = json.loads(sf.read_text())["spec"] if sf.is_file() else json.loads((SCR / "spec-start.json").read_text())
    start = json.loads((SCR / "spec-start.json").read_text())
    log(f"start: phases {phases}, {workers} workers, state {opt.key(state)} (start {opt.key(start)})")
    if "A" in phases:
        state = opt.greedy(state, option_candidates, "options")
    if "B" in phases:
        moves = json.loads((ROOT / "pipeline/setup/autotune-night1.json").read_text())["moves"]
        state = opt.greedy(state, lambda s: schedule_candidates(s, moves), "schedule")
    a, b = opt.evaluate([start, state], SEARCH)
    ha, hb = opt.evaluate([start, state], HOLDOUT)
    log(f"END {opt.key(state)}: vs start search {100 * opt.compare(b, a, SEARCH)[0]:+.3f}% holdout {100 * opt.compare(hb, ha, HOLDOUT)[0]:+.3f}% (t {opt.compare(hb, ha, HOLDOUT)[1]:+.1f})")
    (SCR / "end.json").write_text(json.dumps({"spec": state, "key": opt.key(state)}, indent=1))
    log("DONE")
