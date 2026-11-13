"""The B2-RL rerun with the adjustments the user chose on 2026-10-28 10:3x ("照建議的前幾項調整（折扣換算成每小時、訓練
中定期驗證與提前停止、修正參數對應、加入靜壓上限），在快速模擬器上重跑一次"; docs/plan/reviews/WIN-b2rl-rl-alignment-20261028.md):

- layout v2: + the static pressure cap sp_max_pa, each side of a default scaled on its own (policies/b2rl.py LAYOUTS);
- gamma 0.96, GAE lambda 0.81 per hourly decision (H3C's 0.99 / 0.95 per 15 minutes: the same 25 h / 4.2 h time
  constants);
- H3C's schedule and early stopping: learning rate 3e-4 to 3e-5 over 300 updates (one week per environment each, as
  H3C's epoch is one 7-day episode per environment), then at the floor; `b2rl.py validate` every 25 updates on
  train_val10 (first 7 days, mean action, paired with the rule), stop after 3 validations past update 100 without a
  1% improvement, cap 700 updates; the best checkpoint is kept (best.json);
- 4 environments (H3C's 4; this host also runs the lookahead pilot's 8 official processes), seed 42, fast simulator rom;
  everything else as the full run (runbook §4.2).

Starts the trainer and the validator in their own sessions; logs data/runs/b2rl/v2-rom-s42.log and
data/runs/b2rl/v2-rom-s42-validate.log. Refuses if the run folder exists. Host python3, standard library.

python3 -B pipeline/setup/v2_run.py"""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
NAME = "v2-rom-s42"
RUN = ROOT / "data/runs/b2rl" / NAME
ENVS = 4
UPDATES_DECAY = 300
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
ENV.pop("SB_FASTSIM", None)        # b2rl.py --fast sets it for its own workers


def start(args, log):
    stream = open(ROOT / "data/runs/b2rl" / log, "a", encoding="utf-8")
    return subprocess.Popen(args, cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True).pid


def main():
    if RUN.exists():
        raise SystemExit(f"{RUN.relative_to(ROOT)} exists; nothing started")
    trainer = start(["nice", "-n", "5", "python3", "-B", "scripts/b2rl.py", "train", "--seed", "42", "--envs", str(ENVS),
                     "--out", f"data/runs/b2rl/{NAME}", "--total-decisions", str(UPDATES_DECAY * ENVS * 168),
                     "--max-updates", "700", "--layout", "v2", "--gamma", "0.96", "--gae-lambda", "0.81",
                     "--fast", "rom"], f"{NAME}.log")
    validator = start(["nice", "-n", "5", "python3", "-B", "scripts/b2rl.py", "validate", "--run",
                       f"data/runs/b2rl/{NAME}", "--set", "train_val10", "--max-steps", "2016", "--every", "25",
                       "--min-update", "100", "--patience", "3", "--min-delta", "0.01", "--cap", "700",
                       "--workers", "1", "--fast", "rom"], f"{NAME}-validate.log")
    print(f"trainer {trainer}, validator {validator}")


if __name__ == "__main__":
    main()
