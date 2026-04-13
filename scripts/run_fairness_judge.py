from pathlib import Path
import sys, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import load_crows_pairs
from src.fairness_judge import run_fairness_judge_experiment
(ROOT/'outputs').mkdir(exist_ok=True)
crows=load_crows_pairs(); benchmark=pd.read_csv(ROOT/'outputs'/'crows_pairs_llm_eval.csv')
cases_df,summary_df=run_fairness_judge_experiment(crows,benchmark,cases_out=str(ROOT/'outputs'/'fairness_judge_cases.csv'),summary_out=str(ROOT/'outputs'/'fairness_judge_summary.csv'))
print('Saved outputs/fairness_judge_cases.csv'); print('Saved outputs/fairness_judge_summary.csv'); print(summary_df.to_string(index=False))
