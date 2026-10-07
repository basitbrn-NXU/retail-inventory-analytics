# Retail Sales Forecasting and Inventory Replenishment Planning

## Purpose
This BAN6800 project uses Walmart M5 sales data to forecast sales
and evaluate inventory replenishment decisions.

## Business objectives
- Improve sales forecasts compared with a seasonal baseline.
- Compare ordering policies to balance availability and holding costs.
- Present forecasts and inventory scenarios in an interactive dashboard.

## Initial scope
Focus on 100 regularly selling household product-store histories.
The selected cohort is recorded in manifests/cohort_manifest.csv.
Selection uses training data only, with price coverage checked over the
latest 730 training days. Selection rules, source hashes, and split dates
are recorded in manifests/cohort_audit.json.

## Data sources
- sales_train_evaluation.csv
- calendar.csv
- sell_prices.csv

Raw data is stored locally and excluded from GitHub.

## Important assumptions
Zero sales are retained without assuming their cause.
Stock levels, supplier lead times, and inventory costs are simulation
inputs. Cost savings will be estimated, not claimed as actual results.

## Tools
Python, JupyterLab, pandas, NumPy, scikit-learn, Streamlit, and Plotly.

## Repository folders
- docs: project documentation
- data: data-source documentation
- notebooks: exploratory analysis and modeling
- src: reusable Python code
- app: interactive dashboard
- tests: validation checks
- manifests: data and model version records

## Current status
The repository structure, data dictionary, Technical RAID log, architecture,
and cohort selection records are available. Data preparation code,
forecasting models, inventory simulations, and the dashboard are planned
for the following course modules.

## Project planning
[GitHub project board](https://github.com/users/basitbrn-NXU/projects/1)

## Documentation
- [Technical RAID log and data dictionary](docs/Technical_RAID_and_Data_Dictionary.xlsx)
- [Planned architecture](docs/architecture.png)
- [Editable architecture diagram](docs/architecture.drawio)

The architecture shows the planned workflow; referenced Python modules
will be implemented in later assignments.
