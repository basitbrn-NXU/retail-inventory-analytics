# Retail Sales Forecasting and Inventory Replenishment Planning

BAN6800 Data Analytics Capstone · Abdul Basit Sheikh · Nexford University.

The `final-project` branch brings together the Module 3 pipeline, Module 4 forecasting and explainability source, Module 5 stakeholder dashboard, and the final-project API and monitoring service. Earlier submission branches retain their own histories. This is an academic decision-support prototype for 100 household product–store combinations across ten Walmart stores. It does not place stock orders or represent an authorized Walmart implementation.

## Live demonstrations

| Component | URL |
| --- | --- |
| Stakeholder and SHAP dashboard | https://retail-inventory-analytics-6xymcccu2rnbc9r3zxzdf9.streamlit.app/ |
| Public API documentation | https://ban6800-retail-api.onrender.com/docs |
| Public prediction endpoint | POST https://ban6800-retail-api.onrender.com/predict |
| Monitoring dashboard | https://ban6800-retail-api.onrender.com/monitoring |
| Prometheus exposition | https://ban6800-retail-api.onrender.com/metrics |

The Streamlit application remains deployed from `module-5-dashboard`. The API is deployed from `final-project`. Both use the same frozen Module 4 model hash, `764e977a39487fd64fa374b32e2b79c78a811fdfe8178bbde663a91243fbcad7`. The API example reproduces 0.8481716946876521 daily sales units. Free Render hosting sleeps when idle and runtime telemetry resets after restarts. The model's production approval remains false.

## Find the code and evidence

| Component | Entry point |
| --- | --- |
| Pipeline and quality checks | `flows/m5_pipeline.py`, `src/`, `tests/` |
| Cohort and chronology | `manifests/`, `docs/data_dictionary_module3.md` |
| Governance and pipeline bias | `docs/data_governance.md`, `docs/module3_evidence/` |
| Training and chronological validation | `module4/train_models.py` |
| Frozen final test | `module4/evaluate_test.py` |
| SHAP and sensitivity | `module4/explain_model.py`, `module4/sensitivity.py` |
| Counterfactual null result | `module4/counterfactual.py` |
| Model Card and group review generation | `module4/review.py` |
| Tracking and registry | `module4/setup_tracking.py`, `module4/register_model.py` |
| Original local registered-model API | `module4/api.py` |
| Stakeholder and XAI views | `module5/app.py`, `module5/core.py` |
| Public serving and monitoring | `deployment/cloud_api.py`, `deployment/README.md` |
| CI | `.github/workflows/final-project.yml` |
| Containers | `compose.yaml`, `Dockerfile`, `deployment/Dockerfile` |
| Integration audit and deployment evidence | `docs/final-project/` |

## Setup

Use separate Python 3.12 environments for the pipeline, modeling and dashboard/deployment requirements. Do not install all phase-specific dependency files into one existing working environment.

Pipeline:

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m flows.m5_pipeline --source /absolute/path/m5-forecasting-accuracy.zip
```

Modeling: follow `docs/module4/README.md` and use `requirements-module4.txt`. The source archive and prepared data remain local. Fresh MLflow runs receive new IDs; pass the generated parent run ID to subsequent commands.

Public API and monitoring:

```bash
pip install -r deployment/requirements.txt
python -m pytest deployment/test_cloud_api.py -q
uvicorn deployment.cloud_api:app --host 127.0.0.1 --port 8000 --workers 1
```

Dashboard:

```bash
pip install -r module5/requirements.txt
streamlit run module5/app.py
```

## Multi-container monitoring

Set `GRAFANA_ADMIN_PASSWORD` to a password of your choice before starting. In PowerShell, set it in the current terminal with `$env:GRAFANA_ADMIN_PASSWORD = 'your-chosen-password'`.

```bash
docker compose up --build -d api prometheus grafana
```

Open API/monitoring at localhost:8000, Prometheus at localhost:9090 and Grafana at localhost:3000. Grafana provisions the Retail API dashboard automatically. Its administrator username is `admin`. Local Prometheus data uses a named volume with a 30-day retention setting. Runtime API counters can reset while Prometheus retains prior scrapes.

For the optional pipeline container, set `M5_SOURCE_DIR` to the folder containing the locally obtained Kaggle ZIP. Then run `docker compose --profile prepare run --rm pipeline`. Raw source is mounted read-only; prepared output goes to a local Docker volume. No raw dataset is added to the public repository.

## Evidence and boundaries

The pipeline produced 78,600 records: 73,000 training, 2,800 validation and 2,800 final-test rows. The original Module 3 evidence records twelve tests and twelve quality expectations. Training ends 27 March 2016; validation ends 24 April; the final test ends 22 May. Recursive forecasts exclude future observed sales; historical price is carried forward.

The frozen gradient boosting model has final-test MAE 2.3337, RMSE 3.4014 and WAPE 54.56%, compared with baseline WAPE 67.00%. Overall bias is −1.37%, but HOUSEHOLD_2 underprediction is −32.23%. Store and department checks are operational equity diagnostics. Customer demographic fairness is unassessed. No post-test correction or promotion is applied.

The final inventory analysis is linked at `docs/final-project/inventory/README.md`. Its declared seven-day base case reduces assumed cost by 0.54%, achieves 94.68% fill rate and has a 10.77 percentage-point store gap; the original inventory targets are unmet. This is exploratory simulation on the existing test period, not realized business savings or a new independent evaluation. Uncertainty coverage remains unassessed. New performance and bias drift require observed future labels. The monitoring page separates these missing outcomes from historical scores and live input/activity checks. Automatic deployment is not enabled yet; passing CI verifies tests, image construction and live health checks of the API, Prometheus and Grafana containers.

## Instructor navigation

- Module 2: `module-2-submission` and fixed commit `2936b2a34d0c2fa8769b7feb97e20c2eee2bd6cb`.
- Module 3: `module-3-submission` and `docs/module3/README.md`.
- Module 4: `module-4-development` and `docs/module4/README.md`.
- Module 5: `module-5-dashboard` and `module5/README.md`.
- Final integration: `final-project`.
- Project board: https://github.com/users/basitbrn-NXU/projects/1.

The final report and AI disclosure are submitted through the course portal. Completion of those documents is tracked in the integration audit.
