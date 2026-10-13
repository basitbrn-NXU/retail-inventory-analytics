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

## First forecasting experiment

Run `python -m module4.train_models` after data-readiness setup. The script trains Random Forest and histogram gradient boosting, using two predefined settings per algorithm and two expanding-window folds (origins January 31 and February 28, 2016). Each fold forecasts 28 days recursively. It refits each algorithm's best CV configuration through March 27, then compares its 28-day validation predictions against a weekly seasonal baseline through April 24.

Every fit starts inside an MLflow run. Each fit logs parameters, dates, metrics, the model and predictions. A parent run logs the predeclared protocol, code hash and final comparison. Detailed predictions and model files stay under ignored `models/module4/<run_id>/`. Ten fits, one baseline run and one parent run produce 12 MLflow runs for the standard grid.

Primary selection uses validation RMSE after within-training CV tuning. Provisional development targets are at least 5% RMSE improvement over the baseline and absolute aggregate signed bias no greater than 10%. Failure to meet these targets must be reported. They are project targets, not industry standards or proven savings. Final acceptance also needs untouched test evaluation, explainability and group review.

Input categories are one-hot encoded using each fit's training rows. Scaling is unnecessary for these tree models. Historical price is held at the last observed value during forecasting. Each later lag/rolling feature uses previous predictions, never future actual sales. Known calendar events/SNAP are assumed available at the forecast origin. The target-derived `zero_sales_cause` and future-derived `price_missing` fields are excluded. Gradient boosting early stopping is disabled to avoid a random internal validation split.

The fixed cohort was selected using the full Module 3 training interval; CV conclusions are conditional on this preselected cohort. The selected model is a one-step regressor used recursively over 28 days; explanations later must distinguish a step-level prediction explanation from accumulated horizon effects.

The script does not evaluate the final test period or claim demographic fairness. The best learned model is saved for later XAI even if the weekly baseline wins overall. See `validation_comparison.csv` and `selection.json` before selecting the next step. Actual MLflow model registration, final test evaluation, SHAP, counterfactuals, sensitivity, API and report remain later work.

Verification: synthetic automated tests cover the seasonal baseline, mutation of forbidden future inputs, date-gap rejection and an end-to-end MLflow run with unavailable final test targets. These are software checks, not M5 performance results.

### Frozen model final test

After the validation choice is fixed, run `python -m module4.evaluate_test --parent-run DEVELOPMENT_RUN_ID`. This evaluates the saved validation winner without retraining or tuning. The final origin is April 24, 2016; actual training and validation sales are legitimate history at that origin. Predictions for April 25–May 22 are generated recursively before test labels are joined for scoring. Prices remain at their last available training value because Module 3 withheld later prices. This evaluates a frozen model fitted through March 27, not a newly refitted model.

MLflow records a separate `frozen_model_final_test` run with the source development run, model/data hashes, forecast protocol, model and baseline metrics, predictions and store/department error summaries. Local outputs are under `models/module4/DEVELOPMENT_RUN_ID/test/`. Short artifact names avoid the observed Windows path-length issue. Existing test output blocks repeated evaluation: retain the first result and do not tune against this holdout. Accuracy and bias checks do not approve production deployment; explainability, group review, the Model Card and API remain outstanding.

### SHAP explainability

Install `module4/requirements-xai.txt` into the existing retail-m5 environment, then run `python -m module4.explain_model --parent-run DEVELOPMENT_RUN_ID`. This loads the same hashed model used for the final test and reconstructs the 2,800 validation forecast inputs from observed training history and saved recursive predictions. It checks those inputs reproduce the saved forecasts before computing TreeSHAP with the tree-path-dependent reference. No model fit, tuning or repeated final-test evaluation takes place.

Artifacts include `global.png`, `local.png`, `importance.csv`, `local.csv`, `inputs.csv`, `shap.npz` and `summary.json`. An independent MLflow run records the code, hashes and outputs. The global chart sums one-hot contributions within each original field before averaging absolute contributions over validation rows. The local example is fixed in advance as the first date and lexicographically first series. Additivity is checked for all original-field contributions. The example plot shows ten largest contributions; the full table includes all fields. Raw outputs are explained, with forecast clipping documented. Attributions depend on correlated inputs and the reference distribution and do not establish causal effects. `module4/shap_review.ipynb` reproduces contributions and displays the saved charts; notebook outputs remain empty until executed on the verified local dataset.

Method reference: SHAP documentation, TreeExplainer, https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html (consulted October 13, 2026).

### Fixed-model sensitivity

Run `python -m module4.sensitivity --parent-run DEVELOPMENT_RUN_ID --xai-run EXPLANATION_RUN_ID`. The script verifies that the hashed model and reconstructed validation inputs reproduce saved forecasts. It scales the three sales-history fields together by -20%, -10%, +10% and +20%, or scales price alone by -10% and +10%. IDs and calendar inputs remain fixed. It reports signed and absolute prediction changes, variation across rows and the fraction of scenarios outside each series' observed training input ranges. The range screen is descriptive and does not establish causal or operational plausibility.

This analysis perturbs individual forecast snapshots. It does not regenerate a full recursive horizon, estimate causal price elasticity, train a new model, evaluate the final test again or perform a counterfactual search. MLflow logs a separate `frozen_model_sensitivity` run with CSV results, `sens.png`, the protocol and code. Counterfactual analysis remains a separate deliverable.
