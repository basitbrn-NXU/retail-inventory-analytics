# Retail Sales Forecasting and Inventory Replenishment Planning

BAN6800 Data Analytics Capstone, Abdul Basit Sheikh, Nexford University.

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
passed. Two complete runs produced matching Parquet hashes. These are results
from the preparation workspace; learners should reproduce them locally.
The Dockerfile is provided but its container build/run has not been executed.
Forecasting models, inventory simulation and the dashboard remain later work.

## Run and inspect
[Step-by-step Anaconda and Docker guide](docs/run_pipeline.md)

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m flows.m5_pipeline --source /absolute/path/m5-forecasting-accuracy.zip
```

requirements.txt pins the tested pipeline packages. Future dashboard dependencies
will be added and tested during the later module. Prefect runs locally without a
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
