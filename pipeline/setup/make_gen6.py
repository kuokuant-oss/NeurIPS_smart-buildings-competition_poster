"""Write pipeline/setup/geo5_gen6.py from geo5_gen_buildings.py (戊之三, 2026-11-09): pool gen6, cold weathers only, seeds
outside TRAIN / TRAIN r1 / gen5, outputs under data/runs/b2rl/geo5/r3. python3 -I pipeline/setup/make_gen6.py"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rep(src, old, new):
    assert src.count(old) == 1, old[:70]
    return src.replace(old, new)


def main():
    src = (HERE / "geo5_gen_buildings.py").read_text(encoding="utf-8")
    src = rep(src, 'POOL, LABEL = "gen5", "split-v1/gen5-tree"',
              'POOL, LABEL = "gen6", "split-v1/gen6-r3"\nCOLD_COMBOS = ["fairbanks_1", "fairbanks_3", "denver_1"]')
    src = rep(src, 'OUT = ROOT / "data/runs/b2rl/geo5/tree"',
              'OUT = ROOT / "data/runs/b2rl/geo5/r3"\nGEN5 = ROOT / "data/runs/b2rl/geo5/tree/gen5_split.json"')
    old = ('        | set(old["pools"]["train"]["seeds"]) | {x["seed"] for x in old["pools"]["train"].get("rejected", [])}\n')
    src = rep(src, old, old + '    g5 = json.loads(GEN5.read_text(encoding="utf-8"))\n'
                             '    used |= set(g5["seeds"]) | {x["seed"] for x in g5["rejected"]}   # 戊之三: not gen5 either\n')
    src = rep(src, 'scratch = ROOT / ".tmp/gen5-tree"', 'scratch = ROOT / ".tmp/gen6-r3"')
    src = rep(src, '    combos = [f"{s}_{k}" for s in STATIONS for k in PRESETS]\n    random.Random(LABEL).shuffle(combos)\n',
              '    combos = list(COLD_COMBOS)                                # 戊之三: cold weathers only (point 3)\n')
    src = rep(src, '"script": "pipeline/setup/geo5_gen_buildings.py"', '"script": "pipeline/setup/geo5_gen6.py"')
    src = rep(src, '(OUT / "gen5_split.json")', '(OUT / "gen6_split.json")')
    src = rep(src, '(OUT / "gen5_specs.json")', '(OUT / "gen6_specs.json")')
    src = rep(src, '    tree = json.loads((OUT / "pipeline.json").read_text(encoding="utf-8"))',
              '    tree = json.loads((ROOT / "data/runs/b2rl/geo5/tree/pipeline.json").read_text(encoding="utf-8"))')
    head = ('"""戊之三 (2026-11-09, user points 3 and 6): new generated buildings made exactly like geo5_gen_buildings.py\n'
            '(official generator, same checks), new seeds outside TRAIN, TRAIN r1 and gen5, cold weathers only (Fairbanks\n'
            'January / March, Denver January); pool gen6 (spec "gen6/<building>/<weather>"), record\n'
            'data/runs/b2rl/geo5/r3/gen6_split.json, specs gen6_specs.json.\n'
            'python3 -B pipeline/launch/launch.py pipeline/setup/geo5_gen6.py --n 30\n"""')
    src = head + src.split('"""', 2)[2]
    (HERE / "geo5_gen6.py").write_text(src, encoding="utf-8")
    print("written geo5_gen6.py")


if __name__ == "__main__":
    main()
