# Dataset Documentation and Redistribution Notes

## Included datasets
The package includes the following CSV files:

1. `adult.csv`
   - benchmark family: Adult income prediction
   - role in project: validated tabular fairness benchmark
   - label used in code: `income`
   - sensitive attribute used in code: `sex`

2. `compas.csv`
   - benchmark family: COMPAS recidivism prediction
   - role in project: validated tabular fairness benchmark
   - label used in code: `two_year_recid`
   - sensitive attribute used in code: `race`

3. `german_credit.csv`
   - benchmark family: German Credit
   - role in project: validated tabular fairness benchmark
   - label used in code: `target` or `class`
   - sensitive attribute logic in code: primarily `personal_status`, with fallbacks

4. `crows_pairs.csv`
   - benchmark family: CrowS-Pairs
   - role in project: optional LLM bias benchmarking extension

5. `bbq.csv`
   - benchmark family: BBQ (Bias Benchmark for QA)
   - role in project: optional LLM bias benchmarking extension

## Why the datasets are included
The course instructions require that datasets be included when possible or that the setup be fully reproducible.
To reduce grading friction, the final package includes the benchmark CSVs directly.

## File-structure expectation
The project expects the following paths:
- `datasets/adult.csv`
- `datasets/compas.csv`
- `datasets/german_credit.csv`
- `datasets/crows_pairs.csv`
- `datasets/bbq.csv`

## Alternative setup / refresh
If a user wants to refresh the packaged data, the project also includes `scripts/download_datasets.sh`.
That script is optional because the final zip already includes the needed files.

## Note on LLM benchmark data
CrowS-Pairs and BBQ are included because the project roadmap targeted a cross-modal framework.
The final validated empirical analysis in the report remains the tabular branch.
