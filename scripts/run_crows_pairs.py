from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import load_crows_pairs
from src.llm_eval import run_crows_pairs_llm_eval
(ROOT/'outputs').mkdir(exist_ok=True)
df=load_crows_pairs(); out=run_crows_pairs_llm_eval(df,out_path=str(ROOT/'outputs'/'crows_pairs_llm_eval.csv'),max_items=50)
acc=out['correct'].mean() if 'correct' in out.columns and len(out) else None
print('Saved outputs/crows_pairs_llm_eval.csv'); print('Rows:',len(out)); print('Mode(s):',sorted(out['mode'].dropna().unique().tolist()) if 'mode' in out.columns else []); print('Accuracy vs expected less-biased choice:',round(float(acc),4) if acc is not None else 'n/a')
