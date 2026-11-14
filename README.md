# A learned configuration layer for guideline-based HVAC control of unseen buildings

Code and records of the paper submitted to the paper track of the NeurIPS 2026 Smart Buildings Challenge
(Kuan-Ting Kuo, Chun Fu; National Taiwan University; Kaggle team Anton).

Reproduction comes in two stages.

| Stage | Paper | What | Command | Status |
|---|---|---|---|---|
| 1 | Table 8, rows 14 and 15 | the two submitted Kaggle notebooks (public 0.01056 and 0.01071), rebuilt and compared byte for byte | `run.py submissions` | verified |
| 1 | Tables 3, 4, 9 | the final models, rebuilt from their recorded predecessors | `run.py models` | verified |
| 2 | — | scenarios: training pool, 12 new validation buildings, unseen-station weather and scenarios | `run.py setup` | verified |
| 2 | Tables 2, 5, 7 (costs of 270 scenarios) | validation cost of the reference controllers, the default and the tuned guideline rule, the rule with the learned layer, the per-room selection alone and the two learning targets | `run.py validate` | verified with the paper's reduced thermal models (see below) |
| 2 | Tables 6, 13; Figures 2-5 | the breakdowns, per-step traces and figures built from these runs | `run.py validate --traces`, `run.py tables` | verified with the paper's reduced thermal models |
| 2 | Tables 3, 10, 11, 14 | the full training chain, stage by stage, from the PPO run to both final models | `run.py chain`, `run.py retrain <stage>` | in progress |

Stage 1 needs only `python3`. Both stages were run from a fresh clone of this repository.

Names in the code: tuned rule = `pr_opt2` (public score 0.01071), rule with the learned layer = `gt_x8`
(0.01056), tuned rule with the per-room selection only = `pr_room`, guideline rule with default settings =
`g36_default` (`G36PolicyB2R2`, 0.01441), learned from the cost of individual rooms = `gm_tree_m1`, from the cost of the
whole building = `gt_t2b`.

## Stage 1: the two submissions (public scores 0.01056 and 0.01071) — verified

Needs only `python3` (standard library):

```bash
python3 run.py submissions   # rebuild both Kaggle notebooks from the model files and the controller sources
python3 run.py models        # rebuild the final models from their recorded predecessors
```

`run.py submissions` rebuilds `sb-gt-x8.ipynb` (rule with the learned layer, submission 58255120, public 0.01056) and
`sb-pr-opt2.ipynb` (tuned rule, submission 58270357, public 0.01071) with the command used for the submission. Each of
the 15 embedded source files has the SHA-256 recorded at submission (`data/submissions/<name>/embedded-files.json`),
and, with its one build-date line set to the submission's date, the rebuilt notebook is byte-identical to
`data/submissions/<name>`. `run.py models` rebuilds `gt_x1` to `gt_x8`, `pr_opt2` and `pr_room` and replays the three
hand-made edits; every output is byte-identical to the shipped model file.

The model files and notebooks here differ from the submitted ones in one string: the absolute path of the training
machine's working directory, kept as metadata in each model's `source` field (the controller never reads it), is
written as `/home/user/`. With that path restored, both rebuilt notebooks have the SHA-256 of the submitted notebooks
(`notebook_sha256` in `embedded-files.json`).

To run a notebook, upload it to the competition as a notebook (internet off, competition data attached).

## Stage 2: simulation experiments

What has been rerun from a fresh clone:

* `run.py setup`: the downloaded NOAA files, the five weather files and the 60 unseen-station scenarios, the training
  pool (1,468 files) and the 180 scenarios of the 12 new validation buildings are identical to the ones used for the
  paper (absolute paths inside `config.gin` aside).
* `run.py validate`: all 2,880 recorded validation episodes (270 scenarios; four reference controllers, the default
  and the tuned rule, the rule with the learned layer, the per-room selection alone, the two learning targets and the
  two alternatives of Table 14) are reproduced to the last digit, giving the paper's mean costs (Tables 2, 5, 7) and
  the comparison of the rule with the learned layer and the tuned rule (-1.07 %, cheaper in 177 of 270 scenarios),
  when the fast simulator uses the reduced thermal models of the paper's runs.
* Per-step traces (`--traces`, 540 episodes): every array identical to the paper's. From them and the runs,
  `pipeline/final/an12.py` and `mknum.py` rebuild all 43 number macros of the paper (`numbers.tex`) identically
  (per-scenario means read from the float32 traces differ by at most 2e-7 relative, below any printed digit), and
  `figs10.py`, `figs9.py`, `figs7.py` render Figures 2-5 pixel-identical to the paper's.
* The reduced thermal models themselves are not bit-reproducible: rebuilding them (`run.py setup`) gives the same
  model inputs and cache keys but numerically different files, even with the same script and thread count. With
  rebuilt models, controllers that switch equipment often (the starter kit's fixed-setpoint and PID controllers) give
  different 30-day costs in some scenarios (up to 19.5 % in one episode); bang-bang and the guideline rule agree to
  floating-point noise.

Still in progress: the stages of the training chain. The recorded results behind the paper are in `data/final/recorded` (validation, tuned-rule search) and
`data/runs/b2rl` (stages of the training chain).

Linux (tested on Ubuntu under WSL 2), `git`, `python3` and [uv](https://docs.astral.sh/uv/). The environment is
locked to Python 3.13 and the package versions of the Kaggle notebook image (`pyproject.toml`, `uv.lock`; PyTorch with
CUDA libraries, about 6 GB). The training chain took about two weeks on one workstation.

```bash
python3 run.py env                    # Python 3.13 + .venv from uv.lock
python3 run.py setup --workers 8      # simulator, weather, scenarios, reduced thermal models
python3 run.py validate --workers 8   # 270 validation scenarios, every controller of Tables 2 and 5-7
python3 run.py tables                 # paper numbers + comparison with the recorded results
```

A few scenarios only: `python3 run.py validate --models gt_x8 pr_opt2 --scenes "building_train_1/(atlanta|riyadh)" --limit 2`,
then `python3 -B pipeline/final/compare_results.py`.

The public starter kit at commit `23a4560` passes the fast simulator's check of the official sources it mirrors.

## Data

Nothing of the competition's data is in this repository. `run.py setup`

* clones the challenge's public starter kit (https://github.com/smart-buildings-2026/starter-kit, Apache 2.0) at
  commit `23a4560` into `reference/starter-kit`. This commit has the same simulator as the copy on the Kaggle data
  page that was used for the paper; the two differ only in comments, a debug-image helper and a path constant;
* downloads the 2023 NOAA ISD global-hourly files of the five unseen stations (Riyadh, Dubai, Seattle, London,
  Beijing; public domain) and converts them to the starter kit's weather format;
* generates every scenario with the organisers' generator from recorded seeds: the training pool (80 buildings,
  367 scenarios; the generator refuses to write a pool that differs from `configs/splits/v1-train-r1.json`), the 12
  new validation buildings (`data/final/recorded/val2-manifest.json`) and the 60 unseen-station scenarios of the six
  starter-kit buildings.

The starter kit also ships weather of a sixth station (Minneapolis). It is not used anywhere: no scenario of this
repository uses it, and the scripts filter it out.

## Simulator

All tuning, learning and local validation ran on a fast version of the simulator (`SB_FASTSIM=rom`,
`packages/smart_building/fastsim`): the thermal field of each building is replaced by a reduced-order (Galerkin)
model; the HVAC devices, occupancy, weather, reward, observations and actions are the official code. Against the
official simulator on 87 whole episodes (29 scenarios, 3 controllers) the cost differed by at most 0.455 % (median
0.003 %), and the paired comparisons of the controllers had the same sign in all 29 scenarios. The Kaggle leaderboard
uses the official simulator, and the submitted notebooks never use the fast one.

Local validation adds two corrections proposed in the starter kit's repository (pull requests #2 and #3): the
hot-water tank follows an energy balance (`SB_TANK=balance`, in memory, `pipeline/final/run_c.py`), and the return air
of an air handler is mixed in proportion to the airflow of its rooms (`SB_RETURN_AIR=flow`,
`packages/smart_building/fastsim/patch.py`). The training chain ran without them.

## Layout

```
run.py                     entry point
packages/smart_building/   controllers (g36/, policies/g36.py, policies/b2rl.py), fast simulator, episode loop,
                           scenario generation, PPO
scripts/                   PPO command line, scenario pools, Kaggle notebook builder
pipeline/setup/            stages of the training chain (screening designs, trees, climate condition, searches)
pipeline/final/            final models, tuned-rule search, validation runs, paper numbers and figures
pipeline/manual/           the hand-made steps of the chain, replayed from recorded edits
pipeline/stages.json       the chain, stage by stage (run.py chain)
configs/                   scenario pools and evaluation sets
data/runs/b2rl/            recorded models and stage records (state, summaries, logs)
data/submissions/          the two notebooks as submitted to Kaggle
data/final/recorded/       the validation results behind the paper and the tuned rule's search log
```

## The training chain

`python3 run.py chain` lists the stages; `python3 run.py retrain <stage> --workers N` runs one. Every stage reads
the recorded output of the stage before it, so any stage can be rerun alone. The stages ran as separate jobs over two
weeks, and the final models were assembled from their outputs with a few hand-made steps; the chain is therefore a
sequence of stages, not one script. In `run.py chain`:

* `recorded`: the command is the one recorded in the stage's own launcher or log;
* `reconstructed`: the command follows the script's documented usage; the exact arguments of the original run were
  not recorded;
* `manual`: a decision made by hand. Three files were edited by hand (the autotune end state with
  `dcv_area_share` 0.7; the merges `gt_m1` and `gt_b2`); `pipeline/manual/edits/` records each edit, and
  `recorded_edit.py apply` replays it. The last edits to the rule with the learned layer (`build_x*.py`: four
  building-wide gates removed, night setback 3 K, 30 min waiting time, the 15-set tree, empty working day as night)
  and the night setback of the tuned rule (`pr_setsb.py`) were chosen by hand and are applied by these scripts.
  The model files record every such step in their `source` field.

The stages that compare candidates against a Kaggle gate or submit to Kaggle are run with submission switched off
(`--submit dry` / `none`). Raw per-episode outputs of the chain (about 50 GB) are not included; the summaries, state
files and logs of every stage are.

## License

Apache License 2.0 (`LICENSE`, `NOTICE`).
