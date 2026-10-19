# Retail Sales Forecasting and Inventory Replenishment Planning

BAN6800 Data Analytics Capstone, Abdul Basit Sheikh, Nexford University.

## Module submission versions — instructor navigation

- **Module 2 grading version:** [fixed repository snapshot](https://github.com/basitbrn-NXU/retail-inventory-analytics/tree/2936b2a34d0c2fa8769b7feb97e20c2eee2bd6cb) · [named `module-2-submission` branch](https://github.com/basitbrn-NXU/retail-inventory-analytics/tree/module-2-submission).
- **Module 3 work:** the current `main` branch includes later pipeline implementation. These additions should be reviewed for Module 3, separately from the Module 2 planning submission.
- **All Module 3-related files:** [complete labeled file index](docs/module3/README.md), including shared files updated after Module 2.
- **Module 4 modeling and reported results:** [training and evaluation source folder](https://github.com/basitbrn-NXU/retail-inventory-analytics/tree/module-4-development/module4) · [Module 4 run guide](https://github.com/basitbrn-NXU/retail-inventory-analytics/blob/module-4-development/docs/module4/README.md). The modeling code is on `module-4-development`, not `main`.


The Module 2 snapshot preserves the repository immediately before the Module 3 implementation commit. The fixed commit link remains tied to that exact version as development continues. The Module 2 report was submitted separately through the course portal.

## Module 4 modeling: instructor entry point

The Module 4 submission uses the `module-4-development` branch. Opening only the default `main` branch shows the earlier data pipeline.

| Presented component | Code that produces it |
| --- | --- |
| Random Forest, histogram gradient boosting, chronological CV and validation comparison | [module4/train_models.py](https://github.com/basitbrn-NXU/retail-inventory-analytics/blob/module-4-development/module4/train_models.py) |
| Frozen 28-day test predictions, baseline comparison and group metrics | [module4/evaluate_test.py](https://github.com/basitbrn-NXU/retail-inventory-analytics/blob/module-4-development/module4/evaluate_test.py) |
| Global/local SHAP explanations | [module4/explain_model.py](https://github.com/basitbrn-NXU/retail-inventory-analytics/blob/module-4-development/module4/explain_model.py) |
| Fixed-model sensitivity scenarios | [module4/sensitivity.py](https://github.com/basitbrn-NXU/retail-inventory-analytics/blob/module-4-development/module4/sensitivity.py) |
| DiCE counterfactual search, including a no-solution outcome | [module4/counterfactual.py](https://github.com/basitbrn-NXU/retail-inventory-analytics/blob/module-4-development/module4/counterfactual.py) |

The submitted report's final-test results are MAE **2.3337**, RMSE **3.4014**, R² **0.4367** and WAPE **54.56%** for gradient boosting. The weekly baseline has MAE **2.8657**, RMSE **4.1591** and WAPE **67.00%**. These are the archived Module 4 evaluation results, not newly rerun results.

Use the [Module 4 run guide](https://github.com/basitbrn-NXU/retail-inventory-analytics/blob/module-4-development/docs/module4/README.md) and [Module 4 dependency file](https://github.com/basitbrn-NXU/retail-inventory-analytics/blob/module-4-development/requirements-module4.txt) from that branch. Training requires the verified Module 3 `data/processed/prepared.parquet`; source data, the local MLflow database and run artifacts are excluded from Git. The report's local development run ID is `2380b2054b204fff8f19fec80656d2cf`. A fresh training run creates its own ID; the evaluation command must use that new ID. This navigation update does not retrain a model or change the submitted results.

## Purpose and scope
Forecast retail sales and compare inventory replenishment scenarios using the
Walmart M5 dataset. The fixed cohort contains 100 regularly selling household
product-store histories across ten stores. Source data remains local and is
excluded from public GitHub. Zero sales retain an unknown cause. Inventory
levels, lead times and costs are scenario inputs, so actual savings are not claimed.

## Module 3 implementation
The Prefect workflow in flows/m5_pipeline.py verifies source hashes, validates
raw inputs, joins daily sales/calendar/weekly prices, creates chronological
features, screens for PII, executes a Great Expectations suite, assesses
representation using Fairlearn and publishes local Parquet with lineage.

The executed output contains 78,600 records: 73,000 training, 2,800 validation
and 2,800 test observations. Twelve GX expectations and twelve Pytest tests
passed. Two complete runs produced matching Parquet hashes. The initial results came
from the preparation workspace. On October 12, 2026, the learner also ran the
pipeline in Windows/PyCharm and Docker Desktop. Docker image construction succeeded,
12 container tests passed, and the full container run passed all 12 GX checks.
Windows and Docker output Parquet SHA256 hashes match exactly. See
[learner verification evidence](docs/module3_evidence/learner_verification_2026-10-12.json).
Forecasting code is available on `module-4-development`; the dashboard is on `module-5-dashboard`. Inventory simulation remains outside the demonstrated results.

## Run and inspect
[Step-by-step Anaconda and Docker guide](docs/run_pipeline.md)

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m flows.m5_pipeline --source /absolute/path/m5-forecasting-accuracy.zip
```

requirements.txt pins the tested Module 3 pipeline packages. Module 4 and Module 5 have separate dependency files on their respective branches. Prefect runs locally without a
cloud account. The pipeline outputs live under data/processed/ and stay outside Git.

## Data and leakage safeguards
Source CSVs: sales_train_evaluation.csv, calendar.csv, sell_prices.csv.
The cohort manifest and audit use training data only. Training covers March 29,
2014 to March 27, 2016, validation March 28 to April 24 and test April 25 to May 22.
All post-training actual sales are masked before computing historical features;
held-out actual selling prices are excluded. Some holdout features are therefore
intentionally null. Module 4 must build direct or recursive horizon inputs without
future actual sales. Labels in the prepared table are for evaluation, not predictors.

## Documentation and evidence
- [Module 2 Technical RAID and input dictionary](docs/Technical_RAID_and_Data_Dictionary.xlsx)
- [Module 3 output dictionary](docs/data_dictionary_module3.md)
- [Governance and anonymization plan](docs/data_governance.md)
- [Executed aggregate evidence](docs/module3_evidence/)
- [Learner notebook walkthrough](notebooks/module3_walkthrough.ipynb)
- [Module 2 architecture diagram](docs/architecture.png)
- [GitHub project board](https://github.com/users/basitbrn-NXU/projects/1)

## Repository structure
- flows/: Prefect orchestration
- src/: preparation, Great Expectations validation and bias checks
- tests/: Pytest processing and rejection tests
- data/: source documentation; raw/prepared records remain local
- manifests/: cohort identifiers, selection rules and source hashes
- notebooks/: learner inspection walkthrough
- app/: planned dashboard
- docs/: architecture, RAID, dictionaries, governance and aggregate evidence

The course presentation is submitted through the course portal. Code and aggregate evidence are accessible in this public repository.
