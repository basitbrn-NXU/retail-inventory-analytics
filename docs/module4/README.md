# Module 4: experiment setup

## Scope and instructor clarification

The instructor approved continuing with the M5 dataset provided explainability is demonstrated. This approval supports retaining the dataset. Demographic attributes are unavailable, so demographic fairness cannot be established. Store and department error comparisons will be reported as operational group analysis. No demographic attributes will be fabricated. Global SHAP, local explanations, counterfactual analysis and sensitivity checks remain planned work.

## Preserved submission

Module 3 code/evidence snapshot: `module-3-submission`, commit `df4b68dda36f53e04e4f35ce835f284e5bd60232`. The final presentation is submitted separately. This development branch starts from that code snapshot.

## Start on Windows / PyCharm

1. Download this `module-4-development` branch using Code > Download ZIP and extract into a separate folder. Open it in PyCharm and select the existing `retail-m5` Conda interpreter.
2. Copy the verified Module 3 `data/processed/prepared.parquet` into this folder's `data/processed/`. Raw data and prepared records remain local.
3. In the terminal, activate the intended environment and run:

```text
python -m pip install -r requirements-module4.txt
python -m pip check
python -m module4.setup_tracking
```

The script verifies the exact previously checked Parquet hash, checks the basic dataset structure and logs a clearly labelled setup run. It creates a SQLite tracking database and artifact folder under `tracking/`. It does not fit a model or report forecasting accuracy.

Start the UI from the same project folder:

```text
python -m mlflow server --backend-store-uri sqlite:///tracking/mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and select `BAN6800_Module4_M5_Forecasting`. Keep that terminal running while using the UI. Stop with Ctrl+C. Back up the full `tracking/` folder locally; it is excluded from public GitHub. The local URL is not accessible to the instructor, so provide actual screenshots and exported experiment evidence with the assignment.

## Modeling protocol to implement next

- Target: recorded daily units sold per product-store, with a 28-day forecast horizon. Sales are not unconstrained demand.
- Baseline: repeat the last observed week across the forecast horizon.
- Candidate algorithms: Random Forest and gradient boosting regression, with fixed seeds and modest hyperparameter searches.
- Select models using time-aware folds entirely inside the training period and the validation period. Keep the test period untouched for final evaluation after choices are frozen. Split by dates across all stores, never randomly by rows.
- Rebuild lag/rolling features for each forecast origin. Validation/test rows have intentionally masked historical features. Do not fill these using future target values. Exclude `zero_sales_cause` and other target-derived fields. Use only prices and calendar information available at the forecast origin.
- Define numerical acceptance thresholds before tuning. Compare MAE and RMSE against the baseline, together with signed error and group-level underprediction. No savings or accuracy improvement is claimed yet.
- Open an MLflow training run before every fit. Log parameters, fold dates, data/code hashes, metrics and artifacts. Keep setup runs separate. Register the selected model with a version and lifecycle annotation after validation.
- Remaining deliverables: SHAP notebook, counterfactuals, sensitivity and group error reports, Model Card, serialized model, `/predict` API, 1,800–2,200-word report, and separate AI disclosure.

## Sources

MLflow. (n.d.). Tracking quickstart. https://mlflow.org/docs/latest/ml/tracking/quickstart/ (accessed October 13, 2026).

MLflow. (n.d.). Model Registry. https://mlflow.org/docs/latest/ml/model-registry/ (accessed October 13, 2026).
