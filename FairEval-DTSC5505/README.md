# FairEval-DTSC5505

FairEval is a benchmarking and evaluation framework for studying whether fairness conclusions remain stable across multiple datasets, models, and fairness metrics. The project benchmarks seven classical machine learning models on three tabular fairness datasets, measures agreement among fairness metrics, and extends the evaluation with optional large language model bias benchmarking branches. It also includes generated figures, saved output artifacts, and a Streamlit dashboard for presentation and inspection.

## Project Goal

The central question of this project is:

**Do fairness benchmarks agree, or do different metrics produce different judgments about which models are fairer?**

To answer that question, FairEval evaluates:
- multiple tabular datasets
- multiple classical models
- multiple fairness metrics
- metric agreement statistics
- optional language-bias benchmark branches

The project is designed as a reproducible local package so that the report, dashboard, code, and saved outputs all tell the same story.

---

## Milestone 6 Coverage

This submission addresses the Milestone 6 requirements through the following components:

- **Multiple datasets:** Adult, COMPAS, German Credit
- **Multiple models:** Logistic Regression, Random Forest, Decision Tree, Support Vector Machine, K-Nearest Neighbors, Gradient Boosting, Gaussian Naive Bayes
- **Multiple fairness metrics:** Demographic Parity Difference, Equalized Odds Difference, Predictive Parity Difference
- **Agreement analysis:** Kendall tau agreement matrix and Krippendorff’s alpha
- **Cross-modal branch:** CrowS-Pairs, BBQ, and fairness-judge outputs
- **Generated visualizations:** included
- **Interactive Streamlit dashboard:** included
- **Reproducible local package:** included

---

## Execution Modes

FairEval supports two main execution modes.

### 1. Core Mode
This is the fully local and reproducible workflow. It includes:
- tabular fairness benchmarking
- fairness metric comparison
- agreement analysis
- saved figures
- Streamlit dashboard

### 2. Extended Mode
This adds optional API-backed language-bias evaluation. It includes:
- CrowS-Pairs
- BBQ
- fairness-judge style evaluation

The core mode is the required local workflow. The extended mode is an optional enhancement.

---

## Operating System and Environment Assumptions

### Supported Environment
This project was developed and tested primarily on:
- **Operating System:** macOS
- **Python Version:** 3.12 (recommended)

It may also run on Windows or Linux if the same dependency versions are installed and folder paths are preserved.

### Recommended IDE
- **Visual Studio Code** is recommended for:
  - viewing the project structure
  - running the integrated terminal
  - launching Streamlit
  - recording the code walkthrough

---

## Required Libraries

### Core requirements
Install the core dependencies with:

```bash
pip install -r requirements.txt
```

Core packages include:
- numpy==2.1.3
- pandas==2.2.3
- scikit-learn==1.5.2
- scipy==1.14.1
- matplotlib==3.9.2
- krippendorff==0.8.1
- python-dotenv==1.0.1
- streamlit>=1.44

### Optional language-benchmark requirements
Install optional packages with:

```bash
pip install -r requirements-optional.txt
```

Optional packages include:
- openai==2.30.0
- datasets==3.1.0
- pyarrow==18.1.0

### Conda environment option
You may also create the environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate faireval-m6
```

---

## Project Structure

```text
FairEval-DTSC5505/
├── dashboard/
├── datasets/
├── docs/
├── figures/
├── outputs/
├── scripts/
├── src/
├── main.py
├── streamlit_app.py
├── requirements.txt
├── requirements-optional.txt
├── environment.yml
├── README.md
├── Project-Final Report and Presentation-DTSC5505-Abdul-Azeem Opakunle.docx
└── Do Fairness Benchmarks Agree Project Presentation.pptx
```

### Folder Overview
- `datasets/` – input datasets used by the benchmark
- `src/` – source modules for loading data, preprocessing, model execution, fairness metrics, and agreement analysis
- `scripts/` – helper scripts for figures, optional benchmark branches, dashboard generation, and reproducibility checks
- `outputs/` – saved result artifacts such as CSVs and agreement files
- `figures/` – generated figures used in the report and presentation
- `dashboard/` – dashboard artifacts
- `docs/` – project notes, submission helpers, or recording references
- `main.py` – main entry point for the local benchmark workflow
- `streamlit_app.py` – Streamlit dashboard application

---

## Datasets Used

This project uses the following datasets:

### Tabular Datasets
- **Adult Census Income** – income classification benchmark
- **COMPAS** – two-year recidivism classification benchmark
- **German Credit** – credit risk classification benchmark

### Language-Bias Benchmarks
- **CrowS-Pairs**
- **BBQ**

### Expected Local Dataset Files
Place the following files inside the `datasets/` folder:
- `adult.csv`
- `compas.csv`
- `german_credit.csv`
- `crows_pairs.csv`
- `bbq.csv` (if using the BBQ branch)

### If a dataset is not included directly
If a dataset is too large, restricted, or redistributed separately, the submission should still include:
- the exact dataset name
- the official source
- download instructions
- any preprocessing scripts
- expected file locations and names

---

## Setup Instructions

### 1. Open the project folder
Open the project in VS Code or Terminal.

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
```

On Windows:

```bash
python -m venv .venv
```

### 3. Activate the environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to run the optional language-model branch:

```bash
pip install -r requirements-optional.txt
```

---

## How to Run the Project

All commands below should be run from the **project root directory**.

### Step 1: Run the main local benchmark pipeline

```bash
python main.py
```

This step performs the main tabular workflow:
- dataset loading
- preprocessing
- model training
- prediction
- fairness metric computation
- agreement analysis
- output generation

### Step 2: Generate figures

```bash
python scripts/generate_figures.py
```

This regenerates the project figures used in the report and presentation.

### Step 3: Run optional BBQ branch

```bash
python scripts/run_bbq.py
```

### Step 4: Run optional fairness-judge branch

```bash
python scripts/run_fairness_judge.py
```

### Step 5: Build dashboard artifacts

```bash
python scripts/build_dashboard.py
```

### Step 6: Launch the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

This opens a local dashboard view for:
- executive summary metrics
- tabular benchmark results
- agreement analysis
- language-branch outputs
- roadmap coverage view

---

## Optional CrowS-Pairs API Workflow

If you want to run the API-backed CrowS-Pairs branch, set the environment variables first.

### macOS/Linux

```bash
export OPENAI_API_KEY="your_key_here"
export FAIREVAL_USE_OPENAI=1
export FAIREVAL_OPENAI_MODEL="gpt-5"
PYTHONPATH=. python scripts/run_crows_pairs.py
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:FAIREVAL_USE_OPENAI="1"
$env:FAIREVAL_OPENAI_MODEL="gpt-5"
$env:PYTHONPATH="."
python scripts/run_crows_pairs.py
```

This branch is optional and is not required for the core local package.

---

## Reproducibility Check

If available, run the smoke test:

```bash
python scripts/smoke_test.py
```

This checks whether the expected project files and output structure are present.

---

## Main Output Files

Important generated files include:

- `outputs/results.csv`
- `outputs/agreement_kendall_tau.csv`
- `outputs/agreement_krippendorff_alpha.txt`
- `outputs/crows_pairs_llm_eval.csv`
- `outputs/bbq_llm_eval.csv`
- `outputs/fairness_judge_summary.csv`
- `outputs/fairness_judge_cases.csv`
- `figures/`
- `dashboard/index.html`

These files support the report, slides, dashboard, and code walkthrough.

---

## Approximate Runtime Expectations

Runtime depends on machine speed and whether optional branches are enabled.

Approximate expectations:
- `python main.py` – a few minutes
- `python scripts/generate_figures.py` – under a minute to a few minutes
- `python scripts/run_bbq.py` – depends on configuration and benchmark size
- `python scripts/run_fairness_judge.py` – depends on configuration and benchmark size
- `streamlit run streamlit_app.py` – launches quickly once dependencies are installed

If some outputs are already included in `outputs/`, they do not need to be regenerated just to inspect the dashboard or report artifacts.

---

## Key Findings

The project demonstrates that:
- model fairness can vary substantially across datasets
- fairness metrics do not always agree
- accuracy and fairness do not necessarily move together
- agreement analysis helps reveal whether fairness conclusions are stable or metric-dependent
- language-bias benchmark outputs can be integrated into a broader evaluation workflow
- dashboard packaging improves interpretability and presentation quality

A key final revision in the project corrected the Adult sensitive-attribute parsing issue that had previously forced all Adult disparity values to zero. The corrected rerun restored non-zero Adult disparities across all seven models and materially improved the scientific validity of the benchmark.

---

## Report, Slides, and Recordings

This project submission includes:
- final report
- presentation slides
- source code package
- saved outputs
- datasets or data instructions
- README

The following recordings are expected as part of the milestone submission and may be submitted separately if not stored directly in the repository:
- presentation recording
- code walkthrough / execution recording

---

## Notes for the User

The intended evaluation flow is:
1. read the README
2. inspect the project structure
3. run the benchmark or inspect saved outputs
4. review the report and presentation
5. launch the Streamlit dashboard
6. compare the report narrative with the generated artifacts

The package is designed so that the report, slides, outputs, dashboard, and code walkthrough are all based on the same saved evidence.
