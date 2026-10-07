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
The final selection will use training data and check price coverage.

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
Repository setup is in progress. Forecasting models and inventory
simulations have not yet been implemented.
