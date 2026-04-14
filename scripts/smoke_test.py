from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

required_files = [
    ROOT / "main.py",
    ROOT / "streamlit_app.py",
    ROOT / "src" / "models.py",
    ROOT / "src" / "experiments.py",
    ROOT / "src" / "fairness_metrics.py",
    ROOT / "scripts" / "generate_figures.py",
    ROOT / "scripts" / "run_crows_pairs.py",
    ROOT / "scripts" / "run_bbq.py",
    ROOT / "scripts" / "run_fairness_judge.py",
    ROOT / "scripts" / "build_dashboard.py",
    ROOT / "outputs" / "results.csv",
    ROOT / "outputs" / "agreement_kendall_tau.csv",
    ROOT / "outputs" / "agreement_krippendorff_alpha.txt",
    ROOT / "outputs" / "fairness_judge_summary.csv",
    ROOT / "outputs" / "tool_comparison.csv",
]

missing = [str(p) for p in required_files if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

results = pd.read_csv(ROOT / "outputs" / "results.csv")

required_columns = {
    "Dataset",
    "Model",
    "Accuracy",
    "Demographic Parity",
    "Equalized Odds",
    "Predictive Parity",
    "Accuracy_CI_Low",
    "Accuracy_CI_High",
    "DP_CI_Low",
    "DP_CI_High",
    "EO_CI_Low",
    "EO_CI_High",
    "PP_CI_Low",
    "PP_CI_High",
}

missing_cols = required_columns - set(results.columns)
if missing_cols:
    raise ValueError(f"results.csv is missing columns: {sorted(missing_cols)}")

models = set(results["Model"].unique())
expected_models = {
    "LogisticRegression",
    "RandomForest",
    "DecisionTree",
    "SVM",
    "KNN",
    "GradientBoosting",
    "GaussianNB",
}
missing_models = expected_models - models
if missing_models:
    raise ValueError(f"results.csv is missing benchmark models: {sorted(missing_models)}")

datasets = set(results["Dataset"].unique())
expected_datasets = {"adult", "compas", "german"}
missing_datasets = expected_datasets - datasets
if missing_datasets:
    raise ValueError(f"results.csv is missing datasets: {sorted(missing_datasets)}")

print("SMOKE TEST PASSED")
print("Core package files, outputs, confidence intervals, and expanded benchmark models are present.")