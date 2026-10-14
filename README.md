# Reproduction package — Stage-Wise Evaluation, Failure Localization, and Verified Repair for Text-to-SQL Agents

**Author:** Prabhu Rangaraju

Reproduces the experimental results of *"Closing the Loop: Stage-Wise
Evaluation, Failure Localization, and Verified Repair for Text-to-SQL
Agents"* (Prabhu Rangaraju and Ramesh Kumar Nanjundappan, equal first
authors; manuscript submitted to SCSN-AI 2027 (IEEE ICSC 2027 workshop)).

The package re-runs the full experiment pipeline: a rule-based
Text-to-SQL agent is evaluated question-by-question, each failure is
localized to a pipeline stage, a targeted repair is applied, and the
repaired agent is re-evaluated. All evaluated agents are rule-based —
there are no LLM API calls, no paid services, and no API keys to
configure.

## Prerequisites

- Python 3.10+
- Internet access for the one-time dataset download (~700 MB; skipped on
  reruns once the data is present)
- ~1 GB free disk space

## One command

From this directory:

```bash
bash reproduce.sh
```

That is the entire reproduction. It runs nine steps in order:

1. **Dependencies** — creates a local virtualenv (`.venv/`, ignored by
   git) and installs `requirements.txt` (pandas, pyarrow).
2. **Data fetch** — downloads the public benchmark datasets via
   `fetch_data.sh` (skipped for any dataset already present). Nothing is
   bundled in this package, per the datasets' licenses (see below).
3. **Synthetic arm** (`synth/`) — builds the deterministic 12-question
   retail database, then runs baseline evaluation → stage-wise diagnosis
   → verified repair → re-evaluation.
4. **Spider arm** (`spider/`) — 88-question slice over the
   `concert_singer` and `car_1` databases.
5. **BIRD arm** (`bird/`) — 100-question Mini-Dev slice over
   `california_schools`, `financial`, and `thrombosis_prediction`.
6. **Strengthening experiments** (`strengthening/`) — blind-repair
   ablation, mutant-DB test-suite accuracy, efficiency gate,
   diagnoser-ordering and guard ablations, Agent-B comparison, bootstrap
   confidence intervals, contingency inventory, fix-layer analysis,
   diagnoser ablations.
7. **Held-out generalization** (`strengthening/heldout/`) — dev/held-out
   splits → baselines → fix evaluation on held-out questions.
8. **Validation tests** — `strengthening/tests.py` (76 internal checks).
9. **Verification** — `scripts/verify_reproduction.py` checks the
   headline numbers below against the freshly produced outputs and
   prints PASS/FAIL per check.

All stochastic steps use fixed seeds, so reruns are deterministic. The
only run-to-run variation is wall-clock timing, which is excluded from
the serialized artifacts.

Expected runtime: ~30–40 minutes on a modest 2-CPU machine for the
first run (measured 33 min end-to-end, including the one-time ~700 MB
dataset download); ~25–30 minutes on reruns with data cached. The bulk
of the compute is the BIRD arm and held-out fix evaluation, which run
many agent-generated SQL queries against the BIRD databases (all local
SQLite, no APIs, no model calls — the evaluated agents are rule-based).

## What is verified

| Arm | Baseline EX | After repair | Net gain |
|---|---|---|---|
| Synthetic (n=12) | 5/12 | 12/12 | rp=7, rmm=0 |
| Spider (n=88) | 10/88 | 15/88 | rp=5, rmm=0 |
| BIRD (n=100) | 0/100 | 6/100 | rp=6, rmm=0 |

- Blind-repair ablation: localized repair +5 vs blind repair −5 (Spider,
  EX 5/88 under blind repair).
- Held-out generalization: rp=4, rmm=0 on both the Spider and BIRD
  held-out slices, each with Wilson 95% CIs.

(rp = questions repaired, rmm = regressions introduced; EX = execution
accuracy.)

## Data (not bundled — fetched at reproduce time)

| Dataset | Source | License |
|---|---|---|
| Spider questions | HuggingFace `xlangai/spider` (sha256-pinned) | CC BY-SA 4.0 |
| Spider databases | Official `spider.zip` via yale-lily.github.io/spider (sha256-pinned; only `concert_singer` + `car_1` extracted) | CC BY-SA 4.0 |
| BIRD Mini-Dev | Official BIRD Mini-Dev package (`bird-bench` OSS bucket; sha256-pinned) | CC BY-NC 4.0 (non-commercial use only) |
| Synthetic retail DB | Built deterministically by `synth/db.py` (no download) | — |

Run `bash fetch_data.sh` alone to download just the data without running
the experiments.

## Troubleshooting

- If a download fails (e.g. a Google Drive hiccup), rerun
  `bash fetch_data.sh` — completed files are kept and skipped.
- If `python3 -m venv` is unavailable, install your distribution's
  `python3-venv` package and rerun.
- To verify without rerunning everything: `bash fetch_data.sh`, run the
  arms you need, then `.venv/bin/python scripts/verify_reproduction.py`.

## Citation

See `CITATION.md`.
