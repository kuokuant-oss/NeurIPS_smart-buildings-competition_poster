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
| 2 | Tables 3, 10, 11, 14 | the full training chain, stage by stage, from the PPO run to both final models | `run.py chain`, `run.py retrain <stage>` | verified stage by stage (see [The training chain](#the-training-chain)); PPO is not deterministic |

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
`sb-pr-opt2.ipynb` (tuned rule, submission 58270357, public 0.01071) with the command used for the submission; with its
one build-date line set to the shipped one's, each rebuilt notebook is byte-identical to `data/submissions/<name>`, and
each of its 15 embedded source files has the SHA-256 listed in `embedded-files.json`. `run.py models` rebuilds `gt_x1`
to `gt_x8`, `pr_opt2` and `pr_room` and replays the four hand-made edits; every output is byte-identical to the
shipped model file.

The sources and models in this repository are the submitted ones without comments, docstrings and descriptive
metadata (the models' `source` field, which the controller never reads). The notebooks submitted to Kaggle (SHA-256
`947d55118c1a4090…` for gt-x8, `cce6239c1be97019…` for pr-opt2) were built from the same code: for every embedded
module the syntax tree without docstrings is the same (one log message in `policies/g36.py` is shorter), and every
model file is the same apart from its `source` field.

To run a notebook, upload it to the competition as a notebook (internet off, competition data attached).

## Stage 2: simulation experiments

What has been rerun from a fresh clone:

* `run.py setup`: the downloaded NOAA files, the five weather files and the 60 unseen-station scenarios, the training
  pool (1,468 files) and the 180 scenarios of the 12 new validation buildings are identical to the ones used for the
  paper (absolute paths inside `config.gin` aside); so are the two pools of the training chain's checks (`setup
  --chain`: 480 and 116 files). The split configs in `configs/splits` were written by this repository's generator.
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
  rebuilt models, 2,570 of the 2,880 validation episodes agree to 1e-8 relative. In the others (all in the 90
  starter-kit scenarios), the guideline-rule controllers' costs differ by less than 0.005 %, bang-bang not at all, and
  the starter kit's fixed-setpoint and PID controllers, which switch equipment often, by up to 19.5 % in one episode
  (their mean costs by at most 0.3 %). The paper's comparison is unchanged: the rule with the learned layer costs
  0.009109 on average against 0.009207 for the tuned rule (-1.07 %, cheaper in 177 of 270 scenarios).

The training chain is described and its verification reported in [The training chain](#the-training-chain). The
recorded results behind the paper are in `data/final/recorded` (validation, tuned-rule search) and `data/runs/b2rl`
(stages of the training chain).

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

`python3 run.py chain` lists the 18 stages in order; `python3 run.py retrain <stage> --workers N` runs one. Run
`python3 run.py setup --chain` first (the stages need the VAL and INNER pools and the reduced thermal models of the
training pool). The stages ran as separate jobs over two weeks and the final models were assembled from their outputs
with a few hand-made steps, so the chain is a sequence of stages, not one script.

Every stage can be rerun alone: what it reads from earlier stages is shipped as a record (models, end states, the
summaries and per-scenario rows it compares against), and `retrain` first sets the stage's own recorded state aside
(renamed `<name>.recorded`, the `fresh` entries of `pipeline/stages.json`) so the stage starts from the beginning.
Raw per-episode outputs of the chain (about 50 GB) are not included; a stage that reuses earlier episodes when they
are on disk simulates them instead.

The original run was not a plain sequence of commands; `pipeline/stages.json` reproduces it with these steps:

* hand-made files, replayed from recorded edits (`pipeline/manual/edits/`, `recorded_edit.py apply`): the autotune end
  state with `dcv_area_share` 0.7 (`night1_dcv07`), `gt_wub` (the learned `gt_wua` with one more climate gate) and the
  merges `gt_m1` and `gt_b2`; the end state of the short autotune run as a round file (`pipeline/manual/autotune_end.py`);
* hand-made choices applied by scripts: the reviewed option lists of the building-wide stage (`geo5/review-design.json`
  for the design, `geo5/review.json` after the first closed loop), the night setback of the tuned rule
  (`pr_setsb.py`) and the last edits to the rule with the learned layer (`build_x*.py`);
* runs that were stopped by hand: the overnight autotune after its first stage (`OVERNIGHT_STAGES=1`) and pipeline 2
  after stage n2 (`GEO2_STOP_AFTER=n2`); the first phase-B run of the tuned-rule search used the optimiser before a
  kept move could be retried in the same direction (`PR_OPT_RETRY_KEPT=0`);
* one generated building whose reduced model cannot be built is skipped (`geo5/r3/rom_fail.json`), and the composed
  room tree of `gt_r3d1` is compacted without changing a decision (`tree_compact.py`, 9,504 -> 729 leaves);
* two stages reused records of runs outside the chain: the rows of `gt_r5pa2` on the design scenarios of the
  climate-gate stage (from the tree stage of the building-wide stage) and the rows of `gt_r3a1` on the selection set
  (the bar of the cold-climate stage, from its first attempt); when they are not on disk, the stage records them (same
  model, same scenarios);
* stages that compared candidates against a Kaggle gate or submitted to Kaggle stop before that step (`--until`,
  `--submit none`); the Kaggle steps of the original pipeline (gates, submissions, score polling) are stubs that stop
  with a message.

The building-wide stages stop with exit code 3 and a `wake.json` explaining why when a check fails, including a
memory guard (18 GB for the process tree); `python3 run.py retrain <stage>` again continues from the episodes already
recorded. The reduced thermal models of the chain's generated buildings (pools gen5, gen6) are built by the stage
that generates them, with 2 workers (1-5 GB each; several built at once by the collection would trip the guard).

`recorded` in `run.py chain` means the command is the one recorded in the stage's own launcher, state or log;
`reconstructed`, that it follows the script's usage and was checked against the stage's log; `manual`, a hand-made step.

### Verification

Each stage was rerun from this repository and its products compared with the recorded ones. Stages that reuse
evaluations already on disk were replayed with the original per-episode records (from the generating machine, not
included); stages that simulate every episode were rerun on the rebuilt reduced thermal models.

| Stage | Product | Result |
|---|---|---|
| ppo | checkpoint at update 100 | not deterministic (see below); the recorded checkpoint is shipped and every later stage starts from it |
| offset | constant-action model, INNER rows | rerun: costs within 8e-6 relative of the recorded rows |
| ceiling | parameter sets, rows of 100 TRAIN episodes | rerun: parameter sets identical; costs within 1e-6 relative |
| breakdown | 12 one-parameter variants | rerun: mean effects identical to the printed digits (per-scenario differences of about 1e-7 flip the better / worse counts of the two near-zero `sp_max_pa` variants) |
| search | rounds 1-26, confirm1, schedule1 | replayed: all 121 result lines identical |
| autotune, dcv07 | end states, `night1_dcv07` | replayed: identical |
| geo1 | `gm_tree_m1` | replayed: identical |
| geo2 | the 80 new scenarios, stages n1-n2 | replayed: identical |
| geo3 | `gt_t2b` | replayed: identical |
| geo4 | `gt_u1b` | replayed: identical |
| geo5 | `gt_r5pa2` (design: 8 factors, -0.935 % predicted) | replayed: identical |
| wu | `gt_wua`, then `gt_wub` by the recorded edit | replayed: identical |
| r3 | `gt_r3d1` (after compaction) | replayed: identical |
| merge, final | `gt_m1`, `gt_b2`, `gt_x1`-`gt_x8` | identical (also `run.py models`) |
| tuned_rule | search log (79 decisions), `pr_opt2`, `pr_room` | replayed: identical |
| table10 | `rows_p2.json`: 14 candidates, 1,280 episodes | rerun: costs within 3.3e-4 relative |
| (generated buildings) | gen5 (60) and gen6 (30) draws | rerun: seeds and every building file identical |

"Identical" means equal content apart from descriptive metadata (the models' `source` field, the compaction tool's
path, the date a file was made); "replayed", that every evaluation came from the recorded per-episode records, so the
stage's decisions are checked exactly; "rerun", that every episode was simulated again.

The replays were also run as one chain: a single copy of this repository, `python3 run.py retrain <stage>` for every
stage from search to tuned_rule in the order of `run.py chain` (ppo, offset, ceiling, breakdown and table10, which
simulate every episode, were rerun on their own), with the original per-episode records as the only cache and every
file read or written traced. Every product was written and came out as listed above, and every file a stage read was
in this repository, made by an earlier stage of the run, or a record the stage makes itself when it is missing; the
two records that came from runs outside the chain (`gt_r5pa2` on the climate-gate scenarios, `gt_r3a1` on the
selection set) were removed and recorded anew by the stages, with the same products.

PPO is not deterministic: the trainer collects decisions from its environments asynchronously (a small building does
not wait for a large one), so the batches depend on timing, and two runs with the same seed differ from the first
update on. A rerun to update 100 ends with a 12-step reward of -0.459 against -0.476 recorded (sigma 0.405 against
0.419).

## License

Apache License 2.0 (`LICENSE`, `NOTICE`).
