# Cross module integration audit

Audit date: 24 October 2026. This audit records the current evidence boundary before authoring the integrated report.

| Seam | Evidence and reconciliation | Remaining work |
| --- | --- | --- |
| Business goals to model scores | Original 10% relative WAPE target is met by the frozen evaluation. Better accuracy does not establish inventory savings. | Assess the original inventory cost, fill-rate and service-gap targets using declared scenario assumptions. |
| Pipeline to model chronology | 73,000 training, 2,800 validation and 2,800 test rows; training-only cohort selection and recursive withheld-period features. Observed validation history is legitimate at the final-test forecast origin. | Preserve chronology in any new-data workflow. |
| Pipeline representation to fairness | CA_4 has four series; HOUSEHOLD_2 has nine. The same coverage warnings appear beside historical group errors in monitoring. | New labels and representative data are needed for live group-error monitoring. |
| Model to dashboard and API | Frozen artifact hash and saved example agree. Public POST /predict returned HTTP 200 and 0.8481716946876521. | Maintain an explicit release record if the artifact changes. |
| Batch forecast to API contract | Module 4 produces recursive 28-day batch forecasts. The API takes one-day feature rows. | Report both interfaces accurately; do not call one feature row a 28-day forecast. |
| Fairness metric applicability | Store/department regression losses are available. Demographic parity, equalized odds and disparate impact are unassessed without protected attributes and a binary allocation decision. | Explain applicability, not fabricated demographic metrics. |
| Explanation to action | Global/local SHAP is available; DiCE search found no qualifying counterfactual in its declared search. Alternative past sales are not actionable recourse. | Preserve the null result and causal limitations. |
| Privacy plan to dashboard exports | Modules 1–2 prohibited row-level downloads. The existing Module 5 dashboard allows evaluation-record download for instructor review. | Resolve this exception and record a data-use decision before final portfolio publication. |
| Existing README to integrated code | Earlier README called modeling/dashboard future work. Final-branch README now indexes the inherited pipeline, modeling and dashboard plus serving/monitoring files. | Finish the final report, governance pack, board evidence and disclosure. |
| Static scores to monitoring | Historical scores are dated 2016. Live counters, latency and training-reference numeric PSI checks are separate. At least 100 accepted rows are needed for PSI; no fresh sales labels exist. | Add labeled new-period monitoring before claiming live performance or bias drift. |
| Cloud logging to audit retention | Render Free counters/log files reset on restart. The local Compose configuration adds volume-backed Prometheus history. | Verify multi-container execution; no durable public cloud history is claimed. |
| Container to deployment | Public Render build passed all five API tests; initial CI passed API tests and Docker build. Compose configuration parses successfully. | Validate the complete Compose stack in CI; automatic deployment remains separate. |
| Automation to model governance | CI verifies the deployed candidate; live PSI does not trigger model promotion. | Add new-data candidate retraining with quality checks, a new temporal evaluation and human release gate. |

## Open submission requirements

The 2,500–3,000-word integrated report, final AI disclosure, ethical framework, algorithmic impact assessment, regulatory applicability report, public trust statement, incident/decommissioning plans, and verified project-board/MLflow screenshots are still outstanding. This audit is not the final report and does not establish assignment completion.
