#!/usr/bin/env bash
# Author: Prabhu Rangaraju
# One-command reproduction of the text-to-SQL evaluation-framework results.
#
# Usage:  bash reproduce.sh
# Run from this directory. Requires Python 3.10+.
#
# What it does:
#   1. creates a local virtualenv (.venv) and installs dependencies
#      (requirements.txt: pandas, pyarrow)
#   2. fetches the public benchmark datasets (skipped if already present;
#      nothing is bundled, per the datasets' licenses)
#   3. synthetic arm: build retail.db -> baseline -> diagnose -> fix ->
#      re-evaluate (synth/run.py)
#   4. Spider arm: 88-question slice (spider/run.py)
#   5. BIRD arm: 100-question Mini-Dev slice (bird: python src/run_bird.py)
#   6. strengthening experiments: blind-repair ablation, mutant-DB
#      test-suite accuracy, efficiency gate, ordering/guard ablations,
#      Agent-B comparison, bootstrap CIs, contingency inventory,
#      fix-layer analysis, diagnoser ablations
#   7. held-out generalization: dev/held-out splits, baselines, fix evals
#   8. validation tests (strengthening/tests.py, 72 checks)
#   9. verifies headline numbers against the paper (scripts/verify_reproduction.py)
#
# All stochastic steps use fixed seeds; reruns are deterministic
# (wall-clock timings are the only run-to-run variation and are excluded
# from the serialized artifacts).
# Expected runtime: minutes of compute on a modern laptop (all local
# SQLite, no APIs, no model calls — the evaluated agents are rule-based),
# plus one-time dataset downloads (~700 MB).
set -euo pipefail
cd "$(dirname "$0")"

echo "=================================================================="
echo " STEP 1/9: virtualenv + dependencies"
echo "=================================================================="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -r requirements.txt
VPY="$(pwd)/.venv/bin/python"

echo "=================================================================="
echo " STEP 2/9: fetch public benchmark datasets (skipped if present)"
echo "=================================================================="
bash fetch_data.sh

echo "=================================================================="
echo " STEP 3/9: synthetic arm (n=12 retail suite)"
echo "=================================================================="
(cd synth && "$VPY" run.py)

echo "=================================================================="
echo " STEP 4/9: Spider arm (n=88)"
echo "=================================================================="
(cd spider && "$VPY" run.py)

echo "=================================================================="
echo " STEP 5/9: BIRD arm (n=100 Mini-Dev slice)"
echo "=================================================================="
(cd bird && "$VPY" src/run_bird.py)

echo "=================================================================="
echo " STEP 6/9: strengthening experiments"
echo "=================================================================="
(cd strengthening && "$VPY" exp_a1_blind.py)
(cd strengthening && "$VPY" exp_a2_spider.py)
(cd strengthening && "$VPY" exp_a2_synth.py)
(cd strengthening && "$VPY" exp_a3_spider.py)
(cd strengthening && "$VPY" exp_a3_synth.py)
(cd strengthening && "$VPY" exp_a4_ordering.py)
(cd strengthening && "$VPY" exp_a5_agentb.py)
(cd strengthening && "$VPY" exp_a7_synth_layers.py)
(cd strengthening && "$VPY" exp_a6_bootstrap.py)
(cd strengthening && "$VPY" exp_a7_contingency.py)
(cd strengthening && "$VPY" diag_b_ablation.py)
(cd strengthening && "$VPY" diag_b_s5.py)

echo "=================================================================="
echo " STEP 7/9: held-out generalization"
echo "=================================================================="
(cd strengthening/heldout && "$VPY" split.py)
(cd strengthening/heldout && "$VPY" spider_baseline.py)
(cd strengthening/heldout && "$VPY" spider_fixeval.py)
(cd strengthening/heldout && "$VPY" bird_baseline.py)
(cd strengthening/heldout && "$VPY" bird_fixeval.py)

echo "=================================================================="
echo " STEP 8/9: validation tests"
echo "=================================================================="
(cd strengthening && "$VPY" tests.py)

echo "=================================================================="
echo " STEP 9/9: verify headline numbers"
echo "=================================================================="
"$VPY" scripts/verify_reproduction.py
