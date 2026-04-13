# Code Walkthrough and Execution Recording Script

## Opening
Hello, this is the code walkthrough and execution demo for my DTSC 5505 Milestone 6 project, FairEval.
In this recording I will show the folder structure, explain the purpose of the major files, and demonstrate how the project is run and how the outputs used in the report are produced.

## 1. Show the top-level project structure
Start by opening the project directory and briefly showing:
- `main.py`
- `src/`
- `scripts/`
- `datasets/`
- `outputs/`
- `figures/`
- `README.md`
- `requirements.txt`
- `environment.yml`
- `docs/`

Say:
"This package is organized so that the grader can inspect the code, the datasets, the generated outputs, and the reproducibility instructions in one place."

## 2. Explain README and environment files
Open `README.md`.
Scroll through:
- project overview
- environment setup
- exact dependency versions
- project structure
- how to run
- runtime expectations

Say:
"The README is intended to make the project reproducible without guesswork. It explains both a quick smoke test and the full pipeline."

## 3. Explain the datasets folder
Open `datasets/` and show:
- adult.csv
- compas.csv
- german_credit.csv
- crows_pairs.csv
- bbq.csv
- README_DATASET.md

Say:
"The validated benchmark uses Adult, COMPAS, and German Credit. CrowS-Pairs and BBQ are also included to support the framework's cross-modal extension."

## 4. Walk through the src package
Open the `src` folder and explain each file briefly.

### data_loader.py
"This file loads the packaged CSV datasets."

### preprocessing.py
"This file handles dataset-specific preprocessing, train/test split, and feature transformation with scikit-learn."

### models.py
"This file defines the two baseline models used in the final validated experiments: logistic regression and random forest."

### fairness_metrics.py
"This file computes the fairness disparity measures."

### experiments.py
"This file coordinates model training and evaluation across datasets."

### agreement.py
"This file computes Kendall's tau and Krippendorff's alpha so that I can measure agreement among fairness metrics."

### llm_eval.py and bbq_eval.py
"These files provide optional adapters for CrowS-Pairs and BBQ. They preserve the framework's cross-modal design, although the final validated report focuses on the tabular pipeline."

## 5. Show main.py
Open `main.py`.
Explain the order:
- ensure output directory
- load datasets
- preprocess tabular datasets
- run experiments
- save results
- compute agreement
- save agreement artifacts
- optional LLM calls

Say:
"This is the single main entry point for the benchmark."

## 6. Run a quick integrity check
In the terminal, run:
```bash
python scripts/smoke_test.py
```

Explain:
"This fast check confirms that the packaged files are present before running the full experiment."

## 7. Run the full benchmark
In the terminal, run:
```bash
python main.py
python scripts/generate_figures.py
```

While it runs, narrate:
- the datasets are being loaded
- preprocessing is being applied
- models are being trained
- fairness metrics are being computed
- agreement artifacts are being written
- figures are being regenerated from the output CSV files

## 8. Show outputs
After execution, open:
- `outputs/results.csv`
- `outputs/agreement_kendall_tau.csv`
- `outputs/agreement_krippendorff_alpha.txt`

Say:
"These are the exact files used as the basis for the report's quantitative claims."

## 9. Show figures
Open the `figures/` folder and show:
- `fig1_accuracy_by_model_dataset.png`
- `fig2_demographic_parity.png`
- `fig3_eo_vs_pp.png`
- `fig4_kendall_tau_heatmap.png`

Say:
"These are the report and presentation figures regenerated directly from the saved outputs."

## 10. Close
Conclude with:
"This walkthrough demonstrates that the package is organized, reproducible, and aligned with the written report. The included README, scripts, datasets, outputs, and figures make it possible for the instructor to inspect or rerun the project end to end."