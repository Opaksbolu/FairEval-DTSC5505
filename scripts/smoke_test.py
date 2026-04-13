from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
required=[ROOT/'main.py',ROOT/'README.md',ROOT/'requirements.txt',ROOT/'src'/'data_loader.py',ROOT/'src'/'preprocessing.py',ROOT/'src'/'models.py',ROOT/'src'/'fairness_metrics.py',ROOT/'src'/'experiments.py',ROOT/'src'/'agreement.py',ROOT/'src'/'llm_eval.py',ROOT/'src'/'bbq_eval.py',ROOT/'src'/'fairness_judge.py',ROOT/'scripts'/'run_crows_pairs.py',ROOT/'scripts'/'run_bbq.py',ROOT/'scripts'/'run_fairness_judge.py',ROOT/'scripts'/'build_dashboard.py',ROOT/'datasets'/'adult.csv',ROOT/'datasets'/'compas.csv',ROOT/'datasets'/'german_credit.csv',ROOT/'datasets'/'crows_pairs.csv',ROOT/'datasets'/'bbq.csv']
missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
 print('SMOKE TEST FAILED'); print('Missing files:'); [print(' -',i) for i in missing]; sys.exit(1)
print('SMOKE TEST PASSED'); print('Core package files and upgrade scripts are present.')
