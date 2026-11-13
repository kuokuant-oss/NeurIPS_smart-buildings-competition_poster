"""Pipeline 5 submission rule (2026-11-04, docs/plan/geo-pipeline5.md §5).

- Estimated public score of a gated candidate, anchored on the last scored submission:
    estimate = last public x (1 + candidate S4-first-30 change vs the original rule) / (1 + last one's change)
  (Kaggle gate rows of pipeline/setup/gates.jsonl; the INNER-based estimate is reported too). backtest() replays the
  rule on the scored history of the leaderboard log.
- Submit only when estimate <= reference - 0.0001, reference = the last scored public score, or a lower estimate of a
  submission still waiting for its score ("累積的改善每到 0.0001 公開分數才合併提交").
- Except the result of a whole optimisation run (final=True, what the pipeline's submit stage passes): it is submitted
  without the score threshold, as the new pure-rule baseline, waiting for the next Kaggle day when today's automatic
  slots are used (user 2026-11-05: "最佳化所有選項都跑完可以送出最好的那個，建立新純規則 baseline"; the threshold only
  keeps intermediate results from using the quota). The threshold is still computed and reported.
- Standing quota (memory follow-auto-submit-rules, leaderboard log): 4 automatic a day + 1 spare that is never used
  automatically, Kaggle's day (UTC); authorised until 2026-11-13 00:00 Taipei. Today's count: `kaggle competitions
  submissions <competition> --csv`, rows dated today (UTC), the pattern of submit_candidate.py / score_wait.py.
- The submission itself is pipeline/setup/submit_candidate.py (build, local run, private kernel, parity, quota, submit,
  leaderboard row); then score_wait.py polls hourly. Dry run: everything but those two.
Host python3, standard library; the Kaggle CLI from tools/bin with the repository's config folder (never printed)."""
import csv
import datetime
import io
import json
import os
import re
import subprocess

from geo5_lib import ROOT, SETUP, read_json

COMPETITION = "neurips-2026-smart-buildings"
AUTHORIZED_UNTIL = datetime.datetime(2026, 11, 13, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
AUTO_PER_DAY = 4
MARGIN = 0.0001
LEADERBOARD = ROOT / "docs/experiments/leaderboard-log.md"
ENV = dict(os.environ, PATH=f"{ROOT}/tools/bin:" + os.environ.get("PATH", ""),
           KAGGLE_CONFIG_DIR=str(ROOT / ".secrets/kaggle"))
ENV.pop("SB_FASTSIM", None)


def gate_rows():
    path = SETUP / "gates.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []
    return rows


def gate_of(slug):
    return ([r for r in gate_rows() if r["job"] == f"jg-b2rl-{slug}"] or [None])[-1]


def leaderboard():
    """Rows of the leaderboard log: utc, ref, slug, public (None while pending), inner / s4 relative change (the
    gates.jsonl row when there is one, else the one-decimal text of the row)."""
    out = []
    for line in LEADERBOARD.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| 2026"):
            continue
        c = [x.strip() for x in line.split("|")]
        if len(c) < 8:
            continue
        slug = c[3].split()[0].strip("`") if c[3] else ""
        m = re.match(r"\**([0-9]+\.[0-9]+)", c[7])
        public = float(m.group(1)) if m else None
        rels = [float(x) / 100.0 for x in re.findall(r"（([+-][0-9.]+)%", c[5])]
        gate = gate_of(slug)
        inner = gate["inner"]["rel"] if gate else (rels[0] if len(rels) > 0 else None)
        s4 = gate["s4_first"]["rel"] if gate else (rels[1] if len(rels) > 1 else None)
        out.append({"utc": c[1], "ref": c[2], "slug": slug, "public": public, "pending": "待出分" in c[7],
                    "inner_rel": inner, "s4_rel": s4, "gate_row": bool(gate)})
    return out


def _est(public, rel_ref, rel_new):
    return public * (1.0 + rel_new) / (1.0 + rel_ref)


def reference(rows=None, exclude=()):
    """Anchor = last scored row (and the pending ones after it); exclude: refs of pipeline-5 paper-track submissions,
    so the competition line is anchored on competition submissions only (user 2026-11-04: two tracks)."""
    rows = rows if rows is not None else leaderboard()
    rows = [r for r in rows if r["ref"] not in set(exclude)]
    scored = [r for r in rows if r["public"] is not None and r["s4_rel"] is not None]
    if not scored:
        raise RuntimeError("no scored submission with gate numbers in the leaderboard log")
    last = scored[-1]
    after = rows[rows.index(last) + 1:]
    pending = []
    for r in after:
        if r["public"] is None and r["s4_rel"] is not None:
            pending.append({**r, "estimate": _est(last["public"], last["s4_rel"], r["s4_rel"])})
    return last, pending


def estimate(gate, rows=None, exclude=()):
    last, pending = reference(rows, exclude)
    est = _est(last["public"], last["s4_rel"], gate["s4_first"]["rel"])
    est_inner = _est(last["public"], last["inner_rel"], gate["inner"]["rel"]) if last["inner_rel"] is not None else None
    ref = min([last["public"]] + [p["estimate"] for p in pending])
    return {"estimate": est, "estimate_inner": est_inner, "anchor": {k: last[k] for k in ("ref", "slug", "public",
                                                                                           "s4_rel", "inner_rel")},
            "pending": [{k: p[k] for k in ("ref", "slug", "estimate")} for p in pending],
            "reference": ref, "threshold": ref - MARGIN}


def backtest(rows=None):
    """The anchored estimate replayed on consecutive scored rows with gate numbers: |error| of each step."""
    rows = rows if rows is not None else leaderboard()
    scored = [r for r in rows if r["public"] is not None and r["s4_rel"] is not None]
    out = []
    for a, b in zip(scored, scored[1:]):
        e = _est(a["public"], a["s4_rel"], b["s4_rel"])
        out.append({"from": a["slug"], "to": b["slug"], "estimate": round(e, 6), "public": b["public"],
                    "error": round(e - b["public"], 6)})
    return out


def count_today(fake_csv=None, timeout=180):
    """Submissions on today's Kaggle day (UTC) and the CSV's first data row; fake_csv (test) replaces the CLI."""
    if fake_csv is not None:
        text = fake_csv
    else:
        result = subprocess.run(["kaggle", "competitions", "submissions", COMPETITION, "--csv"], cwd=ROOT, env=ENV,
                                capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError(f"kaggle submissions listing failed (code {result.returncode}): "
                               f"{(result.stdout + result.stderr).strip()[-200:]}")
        text = result.stdout
    lines = text.splitlines()
    header = next(i for i, x in enumerate(lines) if x.startswith("ref,"))
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    rows = list(csv.reader(io.StringIO("\n".join(lines[header:]))))
    data = rows[1:]
    n = sum(1 for r in data if len(r) > 2 and r[2].startswith(today))
    return n, today, (data[0][:3] if data else None)


def _next_day_authorised(when):
    """Whether the next Kaggle day (UTC, 00:05, when submit_candidate.py resumes) is still inside the authorisation."""
    nxt = (when + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    return nxt < AUTHORIZED_UNTIL


def decide_paper(gate, count, duplicate, paper_scored=None, rows=None, when=None, exclude=(), final=True):
    """Paper track (user 2026-11-04): gated and recorded like the competition version, submitted after the
    competition submission within the 4 automatic a day (never the spare), not when it behaves like the competition
    version. final (the result of a whole optimisation run, user 2026-11-05): no score threshold, and when today's
    automatic slots are used it waits for the next Kaggle day. Not final: only when an automatic slot of today is
    left, and - once a paper submission has a public score - only when its estimate is 0.0001 below that score. The
    estimate is anchored like the competition one."""
    when = when or datetime.datetime.now(datetime.timezone.utc)
    est = estimate(gate, rows, exclude)
    reasons = []
    if not (gate["inner"]["gate"] and gate["s4_first"]["gate"]):
        reasons.append("the Kaggle gates did not both pass")
    if when >= AUTHORIZED_UNTIL:
        reasons.append("the automatic-submission authorisation (until 2026-11-13 00:00 Taipei) has ended")
    wait_day = count is not None and count >= AUTO_PER_DAY
    if count is None:
        reasons.append("today's submission count is unknown")
    elif wait_day and not final:
        reasons.append(f"no automatic slot left today ({count} of {AUTO_PER_DAY} used; the spare is never automatic)")
    elif wait_day and not _next_day_authorised(when):
        reasons.append("today's automatic slots are used and the next Kaggle day is after the authorisation")
    if duplicate:
        reasons.append("behaves exactly like the competition version")
    paper_threshold = None
    if paper_scored:
        last = paper_scored[-1]
        paper_est = last["public"] * (1.0 + gate["s4_first"]["rel"]) / (1.0 + last["s4_rel"])
        paper_threshold = last["public"] - MARGIN
        est["estimate_vs_paper"] = paper_est
        if paper_est > paper_threshold and not final:
            reasons.append(f"estimated public {paper_est:.5f} > last paper submission {last['public']:.5f} - {MARGIN}")
    return {"submit": not reasons, "reasons": reasons, "count_today": count, "auto_per_day": AUTO_PER_DAY,
            "quota_wait": bool(wait_day and final and not reasons), "paper_threshold": paper_threshold,
            "final": final, **est}


def decide(gate, count, rows=None, when=None, exclude=(), final=True):
    """Whether to submit now: gates passed, authorised date, quota (4 automatic, waiting for the next Kaggle day when
    they are used), and - unless final (the result of a whole optimisation run, user 2026-11-05) - the estimate below
    the threshold."""
    when = when or datetime.datetime.now(datetime.timezone.utc)
    est = estimate(gate, rows, exclude)
    reasons = []
    if not (gate["inner"]["gate"] and gate["s4_first"]["gate"]):
        reasons.append("the Kaggle gates did not both pass")
    if est["estimate"] > est["threshold"] and not final:
        reasons.append(f"estimated public {est['estimate']:.5f} > threshold {est['threshold']:.5f} (reference "
                       f"{est['reference']:.5f} - {MARGIN})")
    if when >= AUTHORIZED_UNTIL:
        reasons.append("the automatic-submission authorisation (until 2026-11-13 00:00 Taipei) has ended")
    wait_day = count is not None and count >= AUTO_PER_DAY
    if count is None:
        reasons.append("today's submission count is unknown")
    elif wait_day and not _next_day_authorised(when):
        reasons.append("today's automatic slots are used and the next Kaggle day is after the authorisation")
    return {"submit": not reasons, "reasons": reasons, "count_today": count, "auto_per_day": AUTO_PER_DAY,
            "quota_wait": bool(wait_day and not reasons), "final": final, **est}


def commands(name, model_rel, gates_file_rel, description):
    """What the real path runs (submit_candidate.py waits for a day with < 4 submissions itself)."""
    slug = name.replace("_", "-")
    return {"submit": ["python3", "-B", "pipeline/setup/submit_candidate.py", slug, "G36PolicyB2RL", model_rel,
                       gates_file_rel, description],
            "state": f"pipeline/setup/submit-{slug}.json",
            "score_wait": ["python3", "-B", "pipeline/setup/score_wait.py", "<submission ref from the state file>"]}


def submitted_ref(name):
    st = read_json(SETUP / f"submit-{name.replace('_', '-')}.json", {})
    return (st.get("submitted") or {}).get("ref")
