# Module 3 — Executed Pipeline Evidence

These files support Module 3 data pipeline implementation. For the earlier assignment, use the [Module 2 snapshot](https://github.com/basitbrn-NXU/retail-inventory-analytics/tree/2936b2a34d0c2fa8769b7feb97e20c2eee2bd6cb).

| File | Evidence |
| --- | --- |
| [learner_verification_2026-10-12.json](learner_verification_2026-10-12.json) | Learner Windows/PyCharm and Docker build, test, pipeline and matching-hash evidence |
| [run_summary.json](run_summary.json) | Counts, lineage, repeat-run result and execution limitations |
| [gx_suite.json](gx_suite.json) | Great Expectations suite configuration |
| [gx_validation.json](gx_validation.json) | Actual 12-check validation result |
| [gx_suite_report.html](gx_suite_report.html) | Rendered validation report |
| [gx_suite_results.png](gx_suite_results.png) | Image of report rendered from actual validation JSON |
| [pytest_results.txt](pytest_results.txt) | 12 passing Python tests |
| [bias_report.json](bias_report.json) | Fairlearn representation checks by store and department |
| [privacy_audit_example.jsonl](privacy_audit_example.jsonl) | Application access/transformation audit metadata |

Only aggregate results, configuration, hashes and metadata are published. Raw rows and prepared Parquet remain local. The initial workspace did not execute Docker. The learner's later Windows/Docker verification, dated October 12, 2026, is recorded separately above. That verification is transcribed from console output supplied by the learner; the assistant did not directly access the learner's computer. Run evidence refers to the implementation version linked in the [Module 3 index](../module3/README.md); later documentation labeling does not alter the executed code.
