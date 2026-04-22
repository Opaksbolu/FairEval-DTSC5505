# Presentation Script - FairEval Milestone 6

## Slide 1 - Title
Hello everyone. My name is Abdul Azeem Opakunle, and this is my DTSC 5505 Milestone 6 final project presentation.
My project is titled **"Do Fairness Benchmarks Agree? A Cross-Metric Analysis Framework for ML and LLM Bias Auditing."**
This project is in the **Benchmarking / Evaluation** category and focuses on a simple but important question: when we evaluate fairness in machine learning, do the metrics actually tell the same story?

## Slide 2 - Motivation
Fairness auditing is increasingly expected in high-stakes machine-learning systems, but fairness is not a single number.
Different metrics emphasize different notions of harm.
That means a model can appear acceptable under one metric and problematic under another.
So the motivation for FairEval is to build a reproducible framework that makes those differences visible instead of hiding them behind one favorable score.

## Slide 3 - Research Question and Contribution
The central research question is:
**Do fairness benchmarks agree strongly enough to support simple fairness conclusions?**
The main contribution of FairEval is not another new metric.
Instead, it provides a reproducible evaluation workflow that computes multiple fairness metrics and then measures **agreement among the metrics themselves** using Kendall's tau and Krippendorff's alpha.

## Slide 4 - Datasets
The validated tabular benchmark uses three datasets:
Adult, COMPAS, and German Credit.
These were chosen because they are standard fairness benchmarks and represent different decision settings: income prediction, recidivism prediction, and credit risk.
The project package also includes CrowS-Pairs and BBQ so the framework can extend toward LLM bias benchmarking, although the final validated empirical results in this report come from the tabular branch.

## Slide 5 - Pipeline and Models
The pipeline follows a modular structure:
data loading, preprocessing, model training, fairness evaluation, agreement analysis, and figure generation.
For the final validated experiments, I used two baseline models:
logistic regression and random forest.
The goal was not to maximize leaderboard performance, but to create a clean and reproducible benchmark that makes comparison easy.

## Slide 6 - Metrics
The benchmark reports:
accuracy,
demographic parity difference,
equalized odds difference,
and predictive parity difference.
Then it computes a Kendall's tau agreement matrix and Krippendorff's alpha across the fairness metrics.
This is the key idea of the project:
the disagreement itself becomes an empirical result.

## Slide 7 - Accuracy Results
Looking first at predictive performance, both models perform strongly on Adult and COMPAS, while German Credit is the hardest dataset.
Logistic regression and random forest are close on Adult.
On COMPAS, both are extremely accurate.
On German Credit, random forest performs somewhat better than logistic regression.
So the dataset matters more than the choice between these two models.

## Slide 8 - Fairness Results
The fairness results are more interesting.
On Adult, the implemented metrics show zero disparity for both models under the current grouping and split.
On COMPAS, disparities are clearly non-zero, and random forest is worse than logistic regression on all three fairness metrics.
On German Credit, the picture is mixed:
demographic parity and predictive parity are relatively small, but equalized odds is much larger, especially for logistic regression.
That tells us that fairness conclusions depend heavily on which metric we choose.

## Slide 9 - Agreement Analysis
This slide shows the most important finding in the project.
Demographic parity and equalized odds have perfect positive agreement in the aggregated model rankings.
But predictive parity moves in the opposite direction, producing negative agreement with the other two metrics.
The overall Krippendorff's alpha is -0.111111, which is below zero.
That means the fairness metrics do **not** provide strong overall consensus.

## Slide 10 - Interpretation and Limitations
The key interpretation is that fairness evaluation should be treated as a multi-metric diagnostic process, not a single-score checkbox.
The project also has clear limitations.
The validated benchmark is still small: two models and three tabular datasets.
The LLM evaluation branch is included architecturally but not fully validated in the final empirical results.
And the study does not yet include mitigation experiments or confidence intervals.
Still, even in this smaller benchmark, disagreement among fairness metrics is already substantial.

## Slide 11 - Reproducibility and Deliverables
The final Milestone 6 package includes the cleaned source code, datasets, outputs, figures, README, environment files, and scripts for both required recordings.
The pipeline can be rerun from the packaged CSV files, and the report figures can be regenerated directly from the saved outputs.
This was an important part of the final milestone because reproducibility is central to trustworthy auditing.

## Slide 12 - Conclusion
To conclude, FairEval shows that fairness metrics should not be treated as interchangeable.
In the final experiments, the metrics do not agree strongly enough to support a simple one-number fairness conclusion.
The project therefore contributes both a usable benchmarking artifact and a concrete empirical finding.
Future work will expand the model set, add stronger LLM evaluation, and implement the originally planned LLM-as-a-Fairness-Judge module.
Thank you.