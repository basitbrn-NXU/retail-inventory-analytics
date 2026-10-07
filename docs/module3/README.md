# Module 3 — Data Pipeline Implementation

BAN6800 Data Analytics Capstone · Abdul Basit Sheikh

## Scope and grading boundary

This index labels every file added or modified by the Module 3 implementation commit [`fb8cf24`](https://github.com/basitbrn-NXU/retail-inventory-analytics/commit/fb8cf244615e3521054a7c9cf4e7a4c3f4736ec5). It is a later stage of the same retail analytics project, distinct from Module 2 planning.

**For Module 2 grading, use the [fixed pre-Module-3 snapshot](https://github.com/basitbrn-NXU/retail-inventory-analytics/tree/2936b2a34d0c2fa8769b7feb97e20c2eee2bd6cb) or the [module-2-submission branch](https://github.com/basitbrn-NXU/retail-inventory-analytics/tree/module-2-submission).** These preserve the earlier requirements file, README, cohort audit and other setup documentation as they were before Module 3.

## Complete Module 3 file inventory

| File | Module label | Purpose |
| --- | --- | --- |
| [`.dockerignore`](../../.dockerignore) | Module 3 addition | Module 3 container build exclusions |
| [`.gitignore`](../../.gitignore) | Shared file updated in Module 3 | Shared repository exclusions updated for Module 3 local outputs |
| [`Dockerfile`](../../Dockerfile) | Module 3 addition | Module 3 container definition; execution not yet verified |
| [`README.md`](../../README.md) | Shared file updated in Module 3 | Shared project overview and module navigation |
| [`docs/README.md`](../../docs/README.md) | Shared file updated in Module 3 | Shared documentation index |
| [`docs/data_dictionary_module3.md`](../../docs/data_dictionary_module3.md) | Module 3 addition | Module 3 prepared-output field definitions |
| [`docs/data_governance.md`](../../docs/data_governance.md) | Module 3 addition | Module 3 privacy, access, retention and audit controls |
| [`docs/module3_evidence/bias_report.json`](../../docs/module3_evidence/bias_report.json) | Module 3 addition | Module 3 executed aggregate validation / audit evidence |
| [`docs/module3_evidence/gx_suite.json`](../../docs/module3_evidence/gx_suite.json) | Module 3 addition | Module 3 executed aggregate validation / audit evidence |
| [`docs/module3_evidence/gx_suite_report.html`](../../docs/module3_evidence/gx_suite_report.html) | Module 3 addition | Module 3 executed aggregate validation / audit evidence |
| [`docs/module3_evidence/gx_suite_results.png`](../../docs/module3_evidence/gx_suite_results.png) | Module 3 addition | Module 3 executed aggregate validation / audit evidence |
| [`docs/module3_evidence/gx_validation.json`](../../docs/module3_evidence/gx_validation.json) | Module 3 addition | Module 3 executed aggregate validation / audit evidence |
| [`docs/module3_evidence/privacy_audit_example.jsonl`](../../docs/module3_evidence/privacy_audit_example.jsonl) | Module 3 addition | Module 3 executed aggregate validation / audit evidence |
| [`docs/module3_evidence/pytest_results.txt`](../../docs/module3_evidence/pytest_results.txt) | Module 3 addition | Module 3 executed aggregate validation / audit evidence |
| [`docs/module3_evidence/run_summary.json`](../../docs/module3_evidence/run_summary.json) | Module 3 addition | Module 3 executed aggregate validation / audit evidence |
| [`docs/run_pipeline.md`](../../docs/run_pipeline.md) | Module 3 addition | Module 3 Anaconda, workflow and Docker reproduction guide |
| [`flows/__init__.py`](../../flows/__init__.py) | Module 3 addition | Module 3 workflow package |
| [`flows/m5_pipeline.py`](../../flows/m5_pipeline.py) | Module 3 addition | Module 3 Prefect orchestration |
| [`manifests/cohort_audit.json`](../../manifests/cohort_audit.json) | Shared file updated in Module 3 | Shared cohort audit reformatted in Module 3; Module 2 version is in its snapshot |
| [`notebooks/module3_walkthrough.ipynb`](../../notebooks/module3_walkthrough.ipynb) | Module 3 addition | Module 3 learner walkthrough |
| [`pytest.ini`](../../pytest.ini) | Module 3 addition | Module 3 test configuration |
| [`requirements.txt`](../../requirements.txt) | Shared file updated in Module 3 | Shared environment file updated to tested Module 3 package versions |
| [`src/__init__.py`](../../src/__init__.py) | Module 3 addition | Module 3 processing package |
| [`src/bias.py`](../../src/bias.py) | Module 3 addition | Module 3 Fairlearn representation checks |
| [`src/prepare.py`](../../src/prepare.py) | Module 3 addition | Module 3 ingestion, cleaning, integration, features and audit logging |
| [`src/validate.py`](../../src/validate.py) | Module 3 addition | Module 3 Great Expectations validation |
| [`tests/test_prepare.py`](../../tests/test_prepare.py) | Module 3 addition | Module 3 preparation, privacy and leakage tests |
| [`tests/test_validation.py`](../../tests/test_validation.py) | Module 3 addition | Module 3 quality-gate rejection tests |

The following navigation documents were added or updated afterward to label the module boundary: this index, [root README](../../README.md), [documentation index](../README.md) and [Module 3 evidence README](../module3_evidence/README.md).

## Reused Module 2 inputs

The existing [cohort manifest](../../manifests/cohort_manifest.csv), [Technical RAID and input dictionary](../Technical_RAID_and_Data_Dictionary.xlsx), [architecture image](../architecture.png) and [editable architecture](../architecture.drawio) support continuity with Module 2. Their presence on main does not make them new Module 3 deliverables.

## Evidence and remaining verification

The pipeline was executed on actual M5 inputs: 78,600 output records, 12 passing Great Expectations checks and 12 passing Python tests. Two complete runs produced matching Parquet hashes. Evidence is in [module3_evidence](../module3_evidence/). The Dockerfile is supplied, but its container build/run remains unverified. The PowerPoint is submitted separately through the course portal; it is not stored in this repository. Raw data and prepared Parquet remain local.

Forecasting, explainability, inventory simulation and the dashboard are later-module work.
