from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import load_bbq_optional
from src.bbq_eval import run_bbq_llm_eval, summarize_bbq_results
(ROOT/'outputs').mkdir(exist_ok=True)
df=load_bbq_optional()
if df is None: raise SystemExit('datasets/bbq.csv not found')
out=run_bbq_llm_eval(df,out_path=str(ROOT/'outputs'/'bbq_llm_eval.csv'),max_items=200)
print('Saved outputs/bbq_llm_eval.csv'); print('Summary:', summarize_bbq_results(out))
