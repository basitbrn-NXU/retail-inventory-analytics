# AI assistance record for final project integration

This record supports the final NXU AI Disclosure Form. It records assistance actually used during the integration work; it is not a learner signature or a claim of independent learner verification.

## Tool and scope

ChatGPT with Codex workspace, GitHub and browser capabilities assisted with auditing Modules 1–5, comparing repository branches, inspecting existing model artifacts, preparing public API deployment, adding runtime/input monitoring, creating container configuration and writing automated tests. The existing trained model was reused; the assistant did not retrain it or create new final-test forecasts during this deployment step.

## Requests and iteration

The learner asked the assistant to work on the final project and identify required access, then supplied the assignment brief. Subsequent requests supplied the existing dashboard URL, confirmed that Docker had started, provided the Docker server version and confirmed creation of the Render account.

The assistant first audited source documents and code, identified the public API gap, and reconciled historical evaluation with the dashboard model. The implementation then reused the Module 4 request schema and model loader. A training-only reference was created for input distribution checks. Tests covered prediction parity, validation failures, live counter updates, minimum-support rules, historical-score boundaries and monitoring HTML.

Iteration included two observed limitations. Local asynchronous HTTP tests could not complete in the restricted execution session; direct model checks succeeded, and the API integration tests were subsequently executed on Render Linux. The first full Compose check reached a healthy API and ready Prometheus but queried Grafana during initialization; the readiness check was revised to retry transient connection resets within a bounded startup period. This was a test-orchestration change, not a model adjustment. The monitoring wording was also revised to explain that 100 accepted rows are needed for drift results.

## Critical evaluation and modifications

Useful AI contributions included linking the same model hash across report evidence, dashboard and API; identifying the one-day API versus 28-day recursive batch-forecast distinction; and keeping group-bias findings visible. AI-generated code still required execution and correction. A deployable configuration and a passing image build did not initially prove all containers were ready. Historical 2016 scores cannot be described as new live accuracy, and input drift alone cannot prove model degradation or discrimination.

No demographic variables, successful counterfactuals, achieved inventory savings or production approval were invented. The original unsuccessful counterfactual result and the absence of new outcome labels were preserved. The learner must review the final code, report, references, unresolved data-use exception and disclosure before submission, and provide any personal reflection in their own words.

## Verification evidence

- Frozen model SHA-256: `764e977a39487fd64fa374b32e2b79c78a811fdfe8178bbde663a91243fbcad7`.
- Public POST `/predict` returned HTTP 200 and 0.8481716946876521, matching the archived example.
- Render Linux build passed five API/monitoring tests in 1.64 seconds before initial deployment.
- Monitoring displayed the successful public request separately from dated historical metrics.
- Initial GitHub Actions run passed API tests and Docker image construction: https://github.com/basitbrn-NXU/retail-inventory-analytics/actions/runs/40063723810.
- Corrected full multi-container CI passed in 1m 58s, including the API, Prometheus and Grafana health checks: https://github.com/basitbrn-NXU/retail-inventory-analytics/actions/runs/40064799431.

## Sources consulted for deployment

Render. (n.d.). Deploy a FastAPI app. https://render.com/docs/deploy-fastapi

Render. (n.d.). Deploy for free. https://render.com/docs/free

The primary project evidence is the existing Module 1 and 2 documents, Module 3 quality and bias records, Module 4 model validation report and artifacts, Module 5 presentation and dashboard, and the repository source. References for the integrated analytical and regulatory discussion will be verified during report preparation.

## Inventory analysis added after deployment

The assistant subsequently implemented the previously planned inventory-policy comparison. Scenario assumptions were saved before execution; the 36 paired scenarios retained the frozen forecasts and did not tune a new policy from the outcomes. Five accounting and timing tests passed. The seven-day base case failed the original cost, fill-rate and store-gap targets, and these failures were preserved in the report evidence. The existing test outcomes were already known, so the analysis is explicitly exploratory. All stock, cost, lead-time, safety and additional-demand quantities are assumptions; no actual savings or stockouts are claimed.
