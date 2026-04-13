
from pathlib import Path
import pandas as pd, html, base64, mimetypes

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'dashboard'
OUT_DIR.mkdir(exist_ok=True)

results = pd.read_csv(ROOT / 'outputs' / 'results.csv')
tau = pd.read_csv(ROOT / 'outputs' / 'agreement_kendall_tau.csv', index_col=0)
alpha = (ROOT / 'outputs' / 'agreement_krippendorff_alpha.txt').read_text().strip()

crows = pd.read_csv(ROOT / 'outputs' / 'crows_pairs_llm_eval.csv') if (ROOT / 'outputs' / 'crows_pairs_llm_eval.csv').exists() else pd.DataFrame()
bbq = pd.read_csv(ROOT / 'outputs' / 'bbq_llm_eval.csv') if (ROOT / 'outputs' / 'bbq_llm_eval.csv').exists() else pd.DataFrame()
judge_summary = pd.read_csv(ROOT / 'outputs' / 'fairness_judge_summary.csv') if (ROOT / 'outputs' / 'fairness_judge_summary.csv').exists() else pd.DataFrame()

fig_paths = [
    ROOT / 'figures' / 'fig1_accuracy_by_model_dataset.png',
    ROOT / 'figures' / 'fig2_demographic_parity.png',
    ROOT / 'figures' / 'fig3_eo_vs_pp.png',
    ROOT / 'figures' / 'fig4_kendall_tau_heatmap.png',
]

def figure_src(path: Path) -> str:
    if not path.exists():
        return ''
    mime = mimetypes.guess_type(path.name)[0] or 'image/png'
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f"data:{mime};base64,{encoded}"

crows_acc = 'n/a'
bbq_acc = 'n/a'
if not crows.empty and 'correct' in crows.columns:
    crows_acc = f"{float(crows['correct'].mean()):.3f}"

if not bbq.empty and 'correct' in bbq.columns:
    valid = bbq[bbq['correct'].isin([True, False])]
    if len(valid):
        bbq_acc = f"{float(valid['correct'].mean()):.3f}"

judge_html = judge_summary.to_html(index=False) if not judge_summary.empty else '<p>No fairness-judge summary found.</p>'
fig_cards = ''.join(
    f"<div class='card'><img src='{figure_src(path)}' alt='{html.escape(path.name)}'></div>"
    for path in fig_paths
)

html_text = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>FairEval Dashboard</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#222}}
h1,h2{{margin-bottom:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}
.card{{border:1px solid #ddd;border-radius:10px;padding:16px;background:#fafafa}}
img{{max-width:100%;border:1px solid #ddd;border-radius:6px;background:white}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ccc;padding:6px 8px;text-align:left}}
th{{background:#f0f0f0}}
.small{{color:#555;font-size:13px}}
</style>
</head>
<body>
<h1>FairEval Dashboard</h1>
<p class='small'>Static dashboard generated from packaged outputs. This view is designed for inspection during Milestone 6 grading.</p>
<div class='grid'>
  <div class='card'>
    <h2>Core benchmark summary</h2>
    <p><strong>Krippendorff's alpha:</strong> {html.escape(alpha)}</p>
    <p><strong>CrowS-Pairs proxy accuracy:</strong> {html.escape(crows_acc)}</p>
    <p><strong>BBQ proxy accuracy:</strong> {html.escape(bbq_acc)}</p>
    <h3>Tabular results</h3>
    {results.to_html(index=False)}
  </div>
  <div class='card'>
    <h2>Agreement matrix</h2>
    {tau.to_html()}
    <h3>Fairness-judge experiment</h3>
    {judge_html}
  </div>
</div>
<h2>Generated figures</h2>
<div class='grid'>{fig_cards}</div>
</body>
</html>"""

(OUT_DIR / 'index.html').write_text(html_text, encoding='utf-8')
print('Saved dashboard/index.html')
