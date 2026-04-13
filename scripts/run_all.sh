#!/usr/bin/env bash
set -e
python main.py
python scripts/generate_figures.py
python scripts/run_crows_pairs.py
python scripts/run_bbq.py
python scripts/run_fairness_judge.py
python scripts/build_dashboard.py
