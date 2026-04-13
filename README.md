# FairEval - Milestone 6 Submission Package

**Project Title:** Do Fairness Benchmarks Agree? A Cross-Metric Analysis Framework for ML and LLM Bias Auditing  
**Project Category:** Benchmarking / Evaluation  
**Student Name:** Abdul Azeem Opakunle  
**Course:** DTSC 5505  
**Milestone:** Milestone 6 - Final Report & Presentation

## 1. Project Overview
FairEval is a reproducible benchmarking framework for studying whether commonly used fairness metrics agree when auditing machine-learning systems. The final validated pipeline evaluates two classical models (logistic regression and random forest) on three tabular fairness benchmarks (Adult, COMPAS, and German Credit), computes predictive and fairness metrics, and then quantifies agreement among those fairness metrics using Kendall's tau and Krippendorff's alpha.

The package also includes optional scaffolding for lightweight language-model benchmark execution on CrowS-Pairs and BBQ. Those adapters are included to preserve the project's cross-modal design direction, but the **validated empirical results used in the final report are the tabular results** in `outputs/results.csv` and the agreement artifacts in `outputs/agreement_kendall_tau.csv` and `outputs/agreement_krippendorff_alpha.txt`.

## 2. What is included
- `main.py` - main entry point for the benchmark
- `src/` - modular implementation (data loading, preprocessing, models, metrics, agreement analysis, optional LLM adapters)
- `scripts/run_all.sh` - one-command pipeline runner
- `scripts/generate_figures.py` - regenerate report figures from saved outputs
- `scripts/smoke_test.py` - quick integrity check for package contents
- `scripts/run_crows_pairs.py` - runs the packaged CrowS-Pairs proxy benchmark
- `scripts/run_bbq.py` - runs the packaged BBQ proxy benchmark
- `scripts/run_fairness_judge.py` - runs the fairness-judge experiment on CrowS-Pairs outputs
- `scripts/build_dashboard.py` - builds a static dashboard in `dashboard/index.html`
- `datasets/` - packaged CSVs used by the project
- `outputs/` - saved benchmark outputs used by the report
- `figures/` - regenerated PNG figures referenced in the report and slides
- `docs/presentation_script.md` - narration script for the slide presentation
- `docs/code_walkthrough_script.md` - separate script for the code demo / execution recording
- `docs/recording_links.txt` - placeholder file where you can paste the final two video links
- `datasets/README_DATASET.md` - dataset provenance and redistribution notes
- `requirements.txt` - core environment for the validated tabular pipeline
- `requirements-optional.txt` - optional packages for API-backed LLM evaluation / dataset refresh
- `environment.yml` - conda environment for the core tabular pipeline

## 3. Recommended environment
### Operating system assumptions
Validated primarily for:
- macOS or Linux
- Python 3.12.x recommended

### Recommended IDE
- VS Code, PyCharm, or any terminal + editor workflow

## 4. Core installation (recommended for grading)
The package already includes the datasets, outputs, and figures, so the grader does **not** need to download external data to verify the final artifact.

### Option A - pip + venv
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B - conda
```bash
conda env create -f environment.yml
conda activate faireval-m6
```

## 5. Exact core library versions
Core validated stack:
- numpy==2.1.3
- pandas==2.2.3
- scikit-learn==1.5.2
- scipy==1.14.1
- matplotlib==3.9.2
- krippendorff==0.8.1
- python-dotenv==1.0.1

Optional extras (only needed for API-backed / refresh utilities):
- openai==2.30.0
- datasets==3.1.0
- pyarrow==18.1.0

Install the optional extras only if needed:
```bash
pip install -r requirements-optional.txt
```

## 6. Project structure
```text
Project-Final Report and Presentation-DTSC5505-Abdul Azeem Opakunle/
  main.py
  README.md
  requirements.txt
  requirements-optional.txt
  environment.yml
  docs/
    presentation_script.md
    code_walkthrough_script.md
    recording_links.txt
  src/
    data_loader.py
    preprocessing.py
    models.py
    fairness_metrics.py
    experiments.py
    agreement.py
    llm_eval.py
    bbq_eval.py
  scripts/
    run_all.sh
    generate_figures.py
    smoke_test.py
    download_datasets.sh
  datasets/
    adult.csv
    compas.csv
    german_credit.csv
    crows_pairs.csv
    bbq.csv
    README_DATASET.md
  outputs/
    results.csv
    agreement_kendall_tau.csv
    agreement_krippendorff_alpha.txt
    crows_pairs_llm_eval.csv
    bbq_llm_eval.csv
  figures/
    fig1_accuracy_by_model_dataset.png
    fig2_demographic_parity.png
    fig3_eo_vs_pp.png
    fig4_kendall_tau_heatmap.png
```

## 7. How to run
### Quick validation (fastest)
```bash
python scripts/smoke_test.py
```
This confirms that the packaged datasets, outputs, and figures exist and that the core files are present.

### Full tabular pipeline
```bash
python main.py
python scripts/generate_figures.py
```

### One-command pipeline
```bash
bash scripts/run_all.sh
```

### Lightweight text benchmark extensions
```bash
python scripts/run_crows_pairs.py
python scripts/run_bbq.py
python scripts/run_fairness_judge.py
python scripts/build_dashboard.py
```

## 8. What each command does
1. `python main.py`
   - loads Adult, COMPAS, and German Credit
   - preprocesses features and sensitive attributes
   - trains logistic regression and random forest
   - computes accuracy, demographic parity, equalized odds, and predictive parity
   - writes results and agreement outputs to `outputs/`

2. `python scripts/generate_figures.py`
   - reads `outputs/results.csv` and `outputs/agreement_kendall_tau.csv`
   - regenerates the four PNG figures used in the final report and slides

## 9. Approximate runtime expectations
Approximate runtime on a normal laptop:
- smoke test: < 10 seconds
- full tabular pipeline: typically a few minutes, depending on CPU
- figure generation: < 30 seconds

The offline proxy extensions typically run in seconds. API-backed evaluation remains optional and requires network/API access.

## 10. Dataset handling and compliance
All datasets used by the packaged benchmark are included in `datasets/`.  
See `datasets/README_DATASET.md` for:
- exact dataset names
- source notes
- redistribution guidance
- expected file structure

## 11. Important note on optional LLM evaluation
The codebase includes two execution modes for the text branch: offline proxy mode for packaged reproducibility and API-backed mode when `OPENAI_API_KEY` is available. The package also includes a separate fairness-judge experiment and a static dashboard.

## 12. How the package supports the Milestone 6 deliverables
This zip satisfies the code-package side of the assignment by providing:
- fully organized Python source code
- supporting scripts
- datasets used
- reproducibility README
- environment files
- saved outputs
- regenerated figures
- a presentation narration script
- a separate code walkthrough / execution script

You still need to record and submit:
1. the narrated presentation video
2. the code walkthrough / execution video

Paste those two shareable links into `docs/recording_links.txt` before final Canvas upload if your course accepts links.