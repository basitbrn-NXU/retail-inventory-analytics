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

### DiCE counterfactual scenario

Run `python -m module4.counterfactual --parent-run DEVELOPMENT_RUN_ID` after installing `dice-ml==0.12`. The same first-date/first-series example as the local SHAP explanation is used. DiCE random search seeks three alternatives with predicted sales between 110% and 120% of the original prediction. A one-dimensional wrapper around the frozen model scales lag_7, lag_28 and rolling_mean_28 together, while keeping IDs, calendar, SNAP and price fixed. This coherent alternative history is bounded by a 0.5–1.5 multiplier and each history feature's observed training min/max for the same series. DiCE's interface uses a grid of hypothetical scale settings and original-model predictions; those are not real observations and no surrogate or new forecasting model is fitted.

Every returned alternative is verified through the original saved model. The closest returned scale is reported, with no claim of a globally minimal change. No solution within the finite search budget is recorded as a finding, not hidden by relaxing the target. Historical sales are not actionable recourse; training min/max does not guarantee causal or real-world plausibility. MLflow logs `frozen_model_counterfactual` with the model/data hashes, code, fixed target, constraints, original inputs, returned alternatives and interpretation. Source: DiCE project documentation, https://github.com/interpretml/DiCE (consulted October 13, 2026).

### Residuals and documentation

Run `python -m module4.review --parent-run DEVELOPMENT_RUN_ID` after completing the test, SHAP, sensitivity and DiCE stages. This reads saved outputs without generating new test forecasts or fitting models. It verifies that every stage uses the evaluated model hash and produces final-test residual, error-distribution, daily-total and store-bias charts. Positive residuals mean prediction minus actual is positive (overprediction).

The generated `model_card.md` records the actual metric tables and source run IDs; `fairness.md` reports store/department error differences, baseline comparisons and support flags. Demographic parity, equalized odds and disparate impact are not fabricated from store IDs or missing customer attributes. A null DiCE result is retained and finite-search limitations are explicit. The scripts propose monitoring and human-review controls without modifying the model after the final test. Outputs are logged under `frozen_model_documentation` and saved locally under the development folder's `review/REVIEW_RUN_ID/`. Registry and API implementation remain subsequent tasks; the Model Card describes academic use and does not approve deployment.

### Registry and local prediction API

First run `python -m module4.register_model --parent-run DEVELOPMENT_RUN_ID`. This locates the model already logged by the selected validation run and compares its predictions with the evaluated local serialized model. It registers that existing logged model, sets the `candidate` alias and attaches lifecycle/evidence/hash tags. Repeating registration for the same development run and model hash reuses the version. `registry.json` records the pinned version URI and cohort combinations; `api_request.json` provides a verified sample payload. MLflow aliases/tags replace deprecated legacy stage promotion. Academic candidate status does not imply Production approval. The generated Model Card remains linked to the registered version through its documentation run and registration artifacts.

Install `module4/requirements-api.txt`, then run `python -m module4.api --parent-run DEVELOPMENT_RUN_ID`. The API loads the pinned registered version, verifies its evidence hash tag and reproduces the sample prediction before listening at http://127.0.0.1:8000. `/health` reports readiness/version; `/predict` accepts up to 100 rows of the 13 model input fields. Negative/nonfinite inputs, invalid calendar fields, unknown cohort combinations and unseen event categories are rejected. Successful predictions include model version and request ID and produce a JSONL audit record without raw feature values. Swagger UI is available at `/docs`.

This local, unauthenticated demonstration binds only to loopback; it is not a public service or production deployment. The API predicts one day from explicitly supplied historical features. It does not calculate those features or generate a recursive 28-day forecast. `module4.train_models.forecast` provides the separate recursive routine. The fixed cohort restriction and calendar/event validation are deliberate limits on unreviewed inputs. No stock-order quantity, cost saving or customer-level fairness guarantee is returned.

Registry reference: MLflow Model Registry, https://www.mlflow.org/docs/latest/model-registry/ (consulted October 13, 2026).

API dependency correction: FastAPI 0.143.0, Starlette 1.7.0 and Uvicorn 0.54.0 preserve the versions previously installed in the retail-m5 environment and satisfy the observed Prefect 3.8.8 minimums (FastAPI >=0.139.0 and Starlette >=1.0.1). The initial older API pins caused a conflict and are superseded. After installing the corrected API requirements, run `python -m pip check` in the actual environment before starting the API. `requirements-module4.txt` now includes the XAI and API package lists for a complete Module 4 installation.
