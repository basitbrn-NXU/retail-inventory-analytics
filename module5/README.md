# Module 5 retail forecast dashboard

This dashboard presents the frozen Module 4 model and recorded evaluation to inventory planners, store managers and governance reviewers. It performs no training and sends no stock orders.

## Start on your PC

Open the module5 folder in PyCharm. Use a separate Python 3.12 environment so earlier modules retain their working dependencies.

Run these commands from the folder containing this README:

```powershell
& "C:\Users\ab_kb\anaconda3\envs\retail-m5\python.exe" -m venv "C:\Users\ab_kb\venvs\m5-dashboard"
& "C:\Users\ab_kb\venvs\m5-dashboard\Scripts\python.exe" -m pip install -r requirements.txt
& "C:\Users\ab_kb\venvs\m5-dashboard\Scripts\python.exe" -m streamlit run app.py
```

Open the Local URL printed in the terminal, normally http://localhost:8501. Keep that terminal running.

The model, predictions and three explanation examples are included. The model is stored as base64 text; the loader verifies the SHA-256 of the decoded bytes before loading. This is the original Module 4 model, not a retrained version. You do not need to copy raw Kaggle files, retrain the model, or rerun the final test.

## Pages

- **Forecast results:** store/department filters, actual-versus-forecast chart, baseline comparison and downloadable selected records.
- **Prediction examples:** three first-day validation forecasts, recorded sales and saved local/global SHAP explanations.
- **What-if explorer:** change history inputs together and historical price; obtain predictions through the unchanged Module 4 API factory.
- **Fairness and ethics:** Fairlearn MetricFrame group losses, sample coverage, bias flags and compliance status.
- **Sources and scope:** business context, references, limitations and provenance.

These historical retail records contain no customer demographics. Store/department errors are operational diagnostics, not demographic fairness measurements. The dashboard does not claim achieved cost savings, service levels or causal effects.

## API reuse

module4_api.py contains the Module 4 SalesRow and PredictRequest schemas and create_app factory copied verbatim. The loading adapter verifies the original model's SHA-256 and sample prediction. It uses the exported model rather than the Windows-local MLflow database; the version and registry manifest are retained.

Streamlit calls the same /predict route in-process through FastAPI's TestClient. This works after cloud deployment without trying to reach a laptop's localhost. This integration does not claim a separately hosted HTTP API.

For a real HTTP demonstration, run the following in a separate terminal (leave Streamlit running):

```powershell
& "C:\Users\ab_kb\venvs\m5-dashboard\Scripts\python.exe" serve_api.py
```

Then run this in another terminal:

```powershell
& "C:\Users\ab_kb\venvs\m5-dashboard\Scripts\python.exe" api_demo.py
```

The demo checks GET /health and POST /predict, including the original prediction 0.8481716946876521. If your original Module 4 API is already running on port 8000, run api_demo.py against it instead of starting a second server.

Only timestamp, request ID, model version/hash, row count and success status enter the temporary local audit log. Request feature values are not written to that log. Cloud temporary audit files are not durable operational logging.

## Deployment preparation

The assignment requires a live Streamlit Cloud or Hugging Face Spaces URL. This package is not a deployed dashboard and has no public URL yet.

For Streamlit Community Cloud, choose repository basitbrn-NXU/retail-inventory-analytics, branch module-5-dashboard, entrypoint module5/app.py and Python 3.12. Keep module5/requirements.txt beside app.py so this environment is separate from earlier-module requirements. Include the supplied data/ directory and model artifact. Do not upload the raw Kaggle archive, credentials or local tracking database.

The .streamlit/config.toml file belongs at the repository root. Test the deployed URL without signing in before adding it to the PPTX.

This branch adds the Module 5 dashboard. Earlier module files are retained. Public deployment must still be completed and checked separately.

## Tests

```powershell
& "C:\Users\ab_kb\venvs\m5-dashboard\Scripts\python.exe" -m pip install pytest==8.4.2
& "C:\Users\ab_kb\venvs\m5-dashboard\Scripts\python.exe" -m pytest tests -q
```

Tests cover source hashes, reproduced accuracy metrics, group-loss agreement, example/SHAP alignment, API validation/auditing, and interactive Streamlit views.

## Evidence boundaries

- Final test: 2,800 archived predictions, April 25–May 22, 2016.
- Explanation examples: March 28, 2016, selected by declared store/department order, not accuracy.
- Local what-if inputs are newly scored validation scenarios, not new final-test evaluations.
- Interface limits are not verified training-support bounds. Price changes can extrapolate.
- Local/global explanations are saved TreeSHAP outputs; they do not imply causation.
- No demographic attributes are invented and no post-test bias correction is applied.
- The original unsuccessful DiCE search is preserved as a null result.

## References

- Streamlit. App dependencies and deployment. https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- Fairlearn. MetricFrame. https://fairlearn.org/v0.13/api_reference/generated/fairlearn.metrics.MetricFrame.html
- SHAP. TreeExplainer. https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html
- Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and practice (3rd ed.). https://otexts.com/fpp3/
- See the dashboard Sources and scope page for the M5 and course-project references.
