# FairEval-DTSC5505

FairEval is an integrated fairness evaluation framework developed for Milestone 6. The project benchmarks multiple machine learning models on multiple tabular datasets, compares fairness metrics, measures agreement between those metrics, and extends the analysis with optional LLM-based text bias evaluation and a fairness-judge experiment. The project also includes generated figures, a Streamlit dashboard, and a reproducible local package structure.

---

## Project Goals

The main goal of FairEval is to study whether fairness conclusions remain stable across:
- multiple datasets
- multiple machine learning models
- multiple fairness metrics
- optional LLM-based bias benchmarks

The project is designed to answer a central question: **do different fairness metrics agree on which models are fairer, or do they produce conflicting conclusions?**

---

## Milestone 6 Requirement Coverage

This project covers the Milestone 6 roadmap requirements as follows:

- **Multiple datasets**: Adult, COMPAS, German Credit
- **Multiple models**: Logistic Regression, Decision Tree, Random Forest, GaussianNB
- **Multiple fairness metrics**: Demographic Parity, Equalized Odds, Predictive Parity
- **Metric agreement analysis**: Kendall tau agreement matrix and Krippendorff’s alpha
- **LLM benchmark branch**: CrowS-Pairs and BBQ
- **Fairness-judge experiment**: included
- **Generated visualizations**: included
- **Interactive Streamlit dashboard**: included
- **Reproducible local package**: included

---

## Execution Modes

The project supports two execution styles:

### 1. Core mode
This is the fully reproducible local workflow:
- tabular fairness benchmarking
- fairness metric comparison
- metric agreement analysis
- generated figures
- Streamlit dashboard

### 2. Extended mode
This adds optional API-backed LLM evaluation:
- CrowS-Pairs
- BBQ
- fairness-judge style evaluation

The core mode is the required and fully reproducible package. The LLM branch is an optional extension.

---

## Project Structure

```text
FairEval-DTSC5505/
├── datasets/
├── outputs/
├── figures/
├── dashboard/
├── scripts/
├── src/
├── streamlit_app.py
├── main.py
├── requirements.txt
└── README.md

Setup Instructions

1. Clone or open the project folder

Open the FairEval-DTSC5505 folder in VS Code or Terminal.

2. Create a virtual environment

On macOS or Linux:

python3 -m venv .venv

On Windows:

python -m venv .venv

3. Activate the virtual environment

On macOS or Linux:

source .venv/bin/activate

On Windows PowerShell:

.venv\Scripts\Activate.ps1

On Windows Command Prompt:

.venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

If needed, install these manually:

pip install pandas numpy scikit-learn matplotlib streamlit openai python-dotenv

5. Confirm the dataset files exist

Make sure these files are inside the datasets/ folder:
	•	adult.csv
	•	compas.csv
	•	german_credit.csv
	•	crows_pairs.csv
	•	optional: bbq.csv

6. Run the smoke test

python scripts/smoke_test.py

This checks that the expected project files and outputs are present.

⸻

Main Components

1. Tabular fairness benchmarking

The project evaluates multiple classical machine learning models on:
	•	Adult
	•	COMPAS
	•	German Credit

For each dataset/model pair, the project reports:
	•	Accuracy
	•	Demographic Parity Difference
	•	Equalized Odds Difference
	•	Predictive Parity Difference

2. Agreement analysis

The project measures agreement between metrics using:
	•	Kendall tau agreement matrix
	•	Krippendorff’s alpha

This component is important because it shows whether fairness metrics rank models similarly or differently.

3. LLM benchmark branch

The framework includes optional LLM-based evaluation on:
	•	CrowS-Pairs
	•	BBQ

These experiments extend the evaluation beyond tabular fairness into text-based bias assessment.

4. Fairness-judge experiment

A fairness-judge style evaluation is included to compare benchmark expectations with judge-based reasoning.

5. Dashboard and figures

The project includes:
	•	saved outputs
	•	generated figures
	•	an interactive Streamlit dashboard for presentation and inspection

⸻

How to Run

Core workflow

Run the full local benchmarking pipeline:

python main.py
python scripts/generate_figures.py
python scripts/run_bbq.py
python scripts/run_fairness_judge.py
python scripts/build_dashboard.py

Optional CrowS-Pairs API workflow

If you want to run the API-backed CrowS-Pairs evaluation:

On macOS/Linux:

export OPENAI_API_KEY="your_key_here"
export FAIREVAL_USE_OPENAI=1
export FAIREVAL_OPENAI_MODEL="gpt-5"
PYTHONPATH=. python scripts/run_crows_pairs.py

On Windows PowerShell:

$env:OPENAI_API_KEY="your_key_here"
$env:FAIREVAL_USE_OPENAI="1"
$env:FAIREVAL_OPENAI_MODEL="gpt-5"
$env:PYTHONPATH="."
python scripts/run_crows_pairs.py

This step is optional and is used for the extended LLM evaluation branch.

Launch the Streamlit dashboard

streamlit run streamlit_app.py


Outputs

Important generated files include:
	•	outputs/results.csv
	•	outputs/agreement_kendall_tau.csv
	•	outputs/agreement_krippendorff_alpha.txt
	•	outputs/crows_pairs_llm_eval.csv
	•	outputs/bbq_llm_eval.csv
	•	outputs/fairness_judge_summary.csv
	•	figures/
	•	dashboard/index.html

⸻

Key Findings

The project demonstrates that:
	•	model fairness can vary substantially across datasets
	•	fairness metrics do not always agree
	•	accuracy and fairness may not move together
	•	LLM bias benchmarks can be integrated into a broader fairness evaluation workflow
	•	dashboard packaging improves interpretability and presentation quality

⸻

Reproducibility Notes
	•	Core tabular benchmarking is fully reproducible locally.
	•	LLM/API-backed evaluation is optional.
	•	API keys should never be committed to the repository.
	•	.env, .venv, and .git should not be included in the final submission package.

⸻

Final Submission Notes

The final submission should include:
	•	source code
	•	scripts
	•	dashboard
	•	outputs
	•	figures
	•	README

The final submission should not include:
	•	.env
	•	.venv
	•	.git
	•	exposed API keys