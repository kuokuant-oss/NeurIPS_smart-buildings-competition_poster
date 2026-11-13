"""Wait for one submission's public score (2026-11-01; memory kaggle-scoring-wait: about 8 hours, check hourly), like
pipeline/setup/pipeline.py stage_score: every hour read `kaggle competitions submissions --csv`; when the row is COMPLETE
or ERROR, fill the leaderboard-log row's 「（待出分）」 with the public score and write an EVENT line to
data/runs/b2rl/geo/pipeline.log (the hourly watcher wakes on it). Gives up after 14 hours with an ERROR line.
Host python3, standard library; the Kaggle CLI from tools/bin with the repo's config folder (never printed).

python3 -B pipeline/setup/score_wait.py <ref>"""
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REF = sys.argv[1]
ENV = dict(os.environ, PATH=f"{ROOT}/tools/bin:" + os.environ.get("PATH", ""),
           KAGGLE_CONFIG_DIR=str(ROOT / ".secrets/kaggle"))
LOG = ROOT / "data/runs/b2rl/geo/pipeline.log"


def log(kind, text):
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind} score {REF}: {text}\n")


def main():
    for _ in range(15):
        out = subprocess.run(["kaggle", "competitions", "submissions", "neurips-2026-smart-buildings", "--csv"],
                             cwd=ROOT, env=ENV, capture_output=True, text=True).stdout
        line = next((r for r in out.splitlines() if r.startswith(f"{REF},")), "")
        fields = next(csv.reader([line])) if line else []
        status = fields[4] if len(fields) > 4 else ""
        if "COMPLETE" in status.upper() or "ERROR" in status.upper():
            public = fields[5] if len(fields) > 5 else ""
            path = ROOT / "docs/experiments/leaderboard-log.md"
            text = path.read_text(encoding="utf-8")
            start = text.find(f"| {REF} |")
            if start >= 0 and public:
                end = text.find("\n", start)
                text = text[:start] + text[start:end].replace("（待出分）", f"**{public}**（待補對照）", 1) + text[end:]
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(text)
            log("EVENT", f"scored: {status}, public {public or '?'}")
            return
        time.sleep(3600)
    log("ERROR", "still not scored after 14 hours")


if __name__ == "__main__":
    main()
