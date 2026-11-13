"""Training-pool evaluations with the corrected simulator (SB_TANK=balance, SB_RETURN_AIR=flow), fast simulator as before.
Final tuned rule (pr/state.json) with night setback 0/1/2/3 K, the chain of kept changes, and two all-room settings,
on the search set (40) and the held-out set (60). Everything under <scratch>/pr (rows_p2.json); workspace only read.
python3 -B pipeline/launch/launch.py pr_p2.py <scratch> <workers> <smoke|setback|chain|rooms>"""
import json, os, sys, time, statistics as st
import multiprocessing as mp
from pathlib import Path
os.environ["SB_RETURN_AIR"] = "flow"; os.environ["SB_TANK"] = "balance"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_opt  # noqa: E402
ROOT, DATA, SCEN, SCR = pr_opt.ROOT, pr_opt.DATA, pr_opt.SCEN, pr_opt.SCR


def _episode(args):
    name, spec_key, mode = args
    os.environ["SB_FASTSIM"] = "rom"; os.environ["SB_FASTSIM_NO_BUILD"] = "1"
    if mode in ("both", "flow"): os.environ["SB_RETURN_AIR"] = "flow"
    else: os.environ.pop("SB_RETURN_AIR", None)
    if mode in ("both", "tank"): os.environ["SB_TANK"] = "balance"
    else: os.environ.pop("SB_TANK", None)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    for p in (str(ROOT / "packages"), str(DATA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from smart_building.learned.b2rl_env import setup_simulator
    setup_simulator()
    if mode in ("both", "tank"):
        cwd = os.getcwd()
        import run_c
        run_c._tank_balance()
        os.chdir(cwd)
    from smart_building.evaluation.episode import run_episode
    from smart_building.policies.b2rl import b2rl_policy_class
    t0 = time.perf_counter()
    try:
        pool_name, building, weather = spec_key.split("/")
        base = ROOT / "data/processed/scenarios/split-v1-r1/train" if pool_name == "train" else SCEN / pool_name
        r = run_episode(base / building / weather, b2rl_policy_class(SCR / "models" / f"model-{name}.json"), 8640, policy_key=name, label=spec_key)
    except Exception as error:  # noqa: BLE001
        return name, spec_key, mode, {"error": repr(error)[:300]}
    return name, spec_key, mode, {"cost": -float(r["mean_reward"]), "elapsed": time.perf_counter() - t0}


def main():
    workers, what = int(sys.argv[2]), sys.argv[3]
    o = pr_opt.Opt(workers)
    final = json.loads((SCR / "state.json").read_text())["spec"]; start = json.loads((SCR / "spec-start.json").read_text())
    cp = lambda s: json.loads(json.dumps(s))
    def rc(s, drop=(), **add):
        n = cp(s); [n["rule_config"].pop(k, None) for k in drop]; n["rule_config"].update(add); return n
    def hot(s, k, v):
        n = cp(s); n["hot"][k] = v; return n
    c1 = rc(start, drop=("no_reheat_oat_k", "no_reheat_min_oat24_k")); c2 = rc(c1, setback_heat_drop_k=3.0)
    c3 = hot(c2, "occupied_margin_k", 0.15); c4 = hot(c3, "unoccupied_lead_min", 240); c5 = hot(c4, "occupied_margin_k", 0.2)
    c6 = hot(c5, "occupied_margin_k", 0.25)
    for a, b in ((c3["hot"]["occupied_margin_k"], 0.15), (c5["hot"]["occupied_margin_k"], 0.2)): assert abs(a - b) < 1e-9
    assert o.key(c6) == o.key(final), "chain does not end at the final tuned rule"
    sets = {"setback": {f"sb{d}": (rc(final, drop=("setback_heat_drop_k",)) if d == 0 else rc(final, setback_heat_drop_k=float(d))) for d in (3, 2, 0, 1)},
            "chain": dict(zip(("c0", "c1", "c2", "c3", "c4", "c5", "c6"), (start, c1, c2, c3, c4, c5, c6))),
            "wait": {"final": final, "d30": rc(final, empty_day_delay_min=30), "d120": rc(final, empty_day_delay_min=120)},
            "rooms": {"final": final, "min62": rc(final, **pr_opt.ADDS["all_rooms_62_1_minimum"]), "wall": rc(final, **pr_opt.ADDS["all_rooms_wall_heat_max"])}}
    rows_file = SCR / "rows_p2.json"
    rows = json.loads(rows_file.read_text()) if rows_file.is_file() else {}
    if what == "smoke":
        k = o.key(final); sc = o.sets[pr_opt.SEARCH][:3]
        todo = [(k, s, m) for s in sc for m in ("both", "flow", "tank", "none")]
        with mp.get_context("spawn").Pool(12, maxtasksperchild=1) as pool:
            for name, s, m, row in pool.imap_unordered(_episode, todo):
                print("SMOKE", s, m, row.get("cost", row), "| stored released:", o.rows[k].get(s, {}).get("cost"), flush=True)
        return
    specs = sets[what]
    subsets = (pr_opt.SEARCH,) if what == "rooms" else (pr_opt.SEARCH, pr_opt.HOLDOUT)
    todo = []
    for n, s in specs.items():
        k = o.key(s); rows.setdefault(k, {})
        if not (SCR / "models" / f"model-{k}.json").is_file():
            o.search.build_schedule(k, s)
        todo += [(k, sc, "both") for sub in subsets for sc in o.sets[sub] if sc not in rows[k]]
    print(len(todo), "episodes", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for i, (name, s, m, row) in enumerate(pool.imap_unordered(_episode, todo), 1):
            if "error" in row:
                print("ERROR", name, s, row["error"], flush=True); continue
            rows[name][s] = row
            if i % 20 == 0 or i == len(todo): rows_file.write_text(json.dumps(rows))
    names = list(specs)
    for sub in subsets:
        keys = o.sets[sub]
        for i, n in enumerate(names):
            r = rows[o.key(specs[n])]; line = f"RESULT {what} {sub} n={len(keys)} {n}: mean {st.mean(r[k]['cost'] for k in keys):.6f}"
            refs = ([names[i - 1], "c0"] if i > 0 else []) if what == "chain" else ([x for x in names if x != n] if what == "setback" else (["final"] if n != "final" else []))
            for ref in refs:
                rel, t = o.compare(r, rows[o.key(specs[ref])], sub)
                line += f" | vs {ref} {100*rel:+.3f}% (t {t:+.1f}, lower in {sum(r[k]['cost'] < rows[o.key(specs[ref])][k]['cost'] for k in keys)})"
            print(line, flush=True)
    print("DONE", what, flush=True)


if __name__ == "__main__":
    main()
