from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_eval import run_crows_pairs_eval

DATA = ROOT / "datasets" / "crows_pairs.csv"
OUT = ROOT / "outputs" / "crows_pairs_llm_eval.csv"

use_api = os.getenv("FAIREVAL_USE_OPENAI", "0") == "1"
model = os.getenv("FAIREVAL_OPENAI_MODEL", "gpt-5")

df = run_crows_pairs_eval(
    csv_path=str(DATA),
    output_path=str(OUT),
    model=model,
    use_api=use_api,
    max_items=50,
)

print(f"Saved {OUT}")
print(f"Rows: {len(df)}")
print(f"Mode(s): {sorted(df['mode'].unique().tolist())}")
print(f"Accuracy vs expected less-biased choice: {df['correct'].mean():.3f}")