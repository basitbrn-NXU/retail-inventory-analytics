# Final project API and monitoring

This deployment reuses the frozen Module 4 model, its request validation and prediction contract. The original model is loaded through `module5/core.py`, verified against SHA-256 `764e977a39487fd64fa374b32e2b79c78a811fdfe8178bbde663a91243fbcad7`, and checked against the saved prediction 0.8481716946876521. It is an academic candidate and has no authority to place orders.

## Start locally

Use Python 3.12 in a separate environment. From the repository root:

```bash
pip install -r deployment/requirements.txt
python -m pytest deployment/test_cloud_api.py -q
uvicorn deployment.cloud_api:app --host 127.0.0.1 --port 8000 --workers 1
```

Endpoints:

| Path | Purpose |
| --- | --- |
| `/health` | Model readiness and academic approval boundary |
| `/predict` | POST one to 100 valid feature rows using the original Module 4 schema |
| `/docs` | Interactive OpenAPI documentation |
| `/monitoring` | Human-readable monitoring dashboard |
| `/monitoring/data` | Aggregate monitoring JSON |
| `/metrics` | Prometheus counters and latency histogram |

## Monitoring interpretation

The historical accuracy reference contains the original 2,800 final-test predictions from 25 April to 22 May 2016. The group-loss assessment matches Module 4 and Module 5, including the CA_4 and HOUSEHOLD_2 coverage flags. These are historical evaluation results, not measurements of new live outcomes.

Live counters track successful prediction rows, HTTP status and latency. The rolling feature window stores bin indices for at most 1,000 accepted rows. A training-only aggregate reference from 73,000 rows through 27 March 2016 supports numeric Population Stability Index checks. At least 100 live rows are required; PSI over 0.20 triggers advisory review. This threshold is a project heuristic requiring validation, not a statistical guarantee. Scenario queries can change the input mix without model degradation. No feature drift claim is made before minimum support.

New overall/group forecast accuracy and bias drift require subsequently observed sales labels. They are explicitly unavailable in this demonstration. Customer demographics and a downstream binary allocation decision are absent, so demographic parity, equalized odds and disparate impact are unassessed.

Runtime counters and bin indices are process-local. Metadata audit records are temporary and exclude raw feature values. Render Free storage resets on restart, redeploy or sleep; no durable cloud audit history is claimed. Use one worker because telemetry is not shared across processes. Prometheus deployment with persistent storage is the next step for durable operational monitoring.

## Render configuration

`render.yaml` describes the free Python service. Build installs pinned dependencies and runs API tests before starting. Start uses `uvicorn deployment.cloud_api:app --host 0.0.0.0 --port $PORT --workers 1`. Set Python 3.12.15 and health check `/health`. The native-thread limit is one to control resource usage.

The GitHub Actions workflow runs integration checks and builds the Docker image. Render also runs the tests during its build. Automatic deployment is not yet claimed: a public-repository service may require a connected provider or a deployment hook. Earlier submission branches remain separate.

Free hosting can sleep and loses runtime history. This service demonstrates public deployment; operational production approval remains false.
