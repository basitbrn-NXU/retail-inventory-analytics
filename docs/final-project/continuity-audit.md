# Cross module integration audit

Audit date: 24 October 2026. This audit records the current evidence boundary before authoring the integrated report.

| Seam | Evidence and reconciliation | Remaining work |
| --- | --- | --- |
| Business goals to model scores | Original 10% relative WAPE target is met by the frozen evaluation. Better accuracy does not establish inventory savings. | Exploratory inventory comparison completed in `inventory/README.md`: all three base-case targets unmet. Obtain operational inputs and a new evaluation period before a pilot. |
| Pipeline to model chronology | 73,000 training, 2,800 validation and 2,800 test rows; training-only cohort selection and recursive withheld-period features. Observed validation history is legitimate at the final-test forecast origin. | Preserve chronology in any new-data workflow. |
| Pipeline representation to fairness | CA_4 has four series; HOUSEHOLD_2 has nine. The same coverage warnings appear beside historical group errors in monitoring. | New labels and representative data are needed for live group-error monitoring. |
| Model to dashboard and API | Frozen artifact hash and saved example agree. Public POST /predict returned HTTP 200 and 0.8481716946876521. | Maintain an explicit release record if the artifact changes. |
| Batch forecast to API contract | Module 4 produces recursive 28-day batch forecasts. The API takes one-day feature rows. | Report both interfaces accurately; do not call one feature row a 28-day forecast. |
| Fairness metric applicability | Store/department regression losses are available. Protected attributes and a relevant protected-group decision are absent; demographic parity, equalized odds and disparate impact are not claimed as demographic findings. | Equalized odds requires defined classification outcomes; avoid treating operational proxies as demographic fairness. |
| Explanation to action | Global/local SHAP is available; DiCE search found no qualifying counterfactual in its declared search. Alternative past sales are not actionable recourse. | Preserve the null result and causal limitations. |
| Privacy plan to dashboard exports | Modules 1–2 prohibited row-level downloads. The existing Module 5 dashboard allows evaluation-record download for instructor review. | Resolve this exception and record a data-use decision before final portfolio publication. |
| Existing README to integrated code | Earlier README called modeling/dashboard future work. Final-branch README now indexes the inherited pipeline, modeling and dashboard plus serving/monitoring files. | Finish the final report, governance pack, board evidence and disclosure. |
| Static scores to monitoring | Historical scores are dated 2016. Live counters, latency and training-reference numeric PSI checks are separate. At least 100 accepted rows are needed for PSI; no fresh sales labels exist. | Add labeled new-period monitoring before claiming live performance or bias drift. |
| Cloud logging to audit retention | Render Free counters/log files reset on restart. The local Compose configuration adds volume-backed Prometheus history. | Full API, Prometheus and Grafana stack verified in CI; no durable public cloud history is claimed. |
| Container to deployment | Public Render build passed all five API tests; initial CI passed API tests and Docker build. Compose configuration parses successfully. | Complete Compose stack passed CI run 40064799431. Automatic deployment remains separate. |
| Automation to model governance | New-period candidate code and five safeguard tests are implemented; original final-test reuse is rejected. Optional deployment-hook job follows passing CI. | Eligible new labels, full retraining execution and deployment-hook activation remain unverified. No automatic promotion. |

## Open submission requirements

Private review deliverables now include a 2,637-word integrated report, a filled university AI disclosure plus detailed reflection supplement, and the ethical framework/model card covering impact, regulatory scope, trust, incident response and retirement. Report screenshots show the actual MLflow and board states. Board duplicate cards remain unresolved. Latest integration CI passes fifteen tests and container runtime checks; optional CD is skipped until configured. Learner review, personal reflection and final reference-format confirmation remain necessary. This audit is not the final report and does not establish assignment completion.
