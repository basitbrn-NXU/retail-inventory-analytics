# Run and understand the Module 3 pipeline

## 1. Get the project and source files
Download the latest repository ZIP from GitHub using Code, Download ZIP.
Extract it to a folder you can find. The repository contains code and documentation,
not Walmart raw records. Keep your original M5 ZIP locally, or extract the three
CSVs into a local directory. Do not upload these files through the GitHub website.

## 2. Create an isolated environment
Open Anaconda Prompt. Change directory to the extracted repository root, where
requirements.txt and the src and flows folders appear. For example:

```bat
cd "C:\Users\ab_kb\Documents\Studies - Nexford\University Assignments\BAN6800-DataAnalyticsCapstone\retail-inventory-analytics-main"
conda create -n retail-m5 python=3.12 -y
conda activate retail-m5
python -m pip install -r requirements.txt
```

The environment separates the pipeline dependencies from your existing notebooks.
This Module 3 requirements file pins the packages used for the pipeline. Dashboard
dependencies such as Streamlit and Plotly can be added and tested in Module 5.

## 3. Run the validation tests

```bat
python -m pytest -q
```

Tests check date gaps, duplicate grain, PII screening and chronological splits.
Two mutation tests change current-day or future sales and check that prohibited
values do not enter the lag and rolling features. The Great Expectations negative
test deliberately supplies invalid data and requires the quality gate to reject it.

## 4. Run the Prefect workflow
Replace the path below with the location of your downloaded M5 ZIP:

```bat
python -m flows.m5_pipeline --source "C:\YOUR_FOLDER\m5-forecasting-accuracy.zip"
```

Alternatively pass an extracted directory containing sales_train_evaluation.csv,
calendar.csv and sell_prices.csv. Prefect starts a local temporary server for the
run. No Prefect Cloud account is required. First startup may take a few seconds.
If your company requires a network proxy, local server access needs the appropriate
localhost proxy exception. Do not disable organisational controls to run the project.

## 5. Inspect the actual outputs
Open data/processed/gx_report.html in your browser. It presents results generated
by the real Great Expectations suite, rather than manually assigned pass labels.
Inspect lineage.json for source hashes, versions, counts, split dates and code hashes.
Inspect bias_report.json for candidate and selected counts by store and department.
Inspect privacy_audit.jsonl for application access and transformation events.
prepared.parquet contains 78,600 daily observations for Module 4. Keep all of these
local outputs outside public GitHub. The local Parquet hash supports repeat-run checks.

Optional Jupyter exploration from this environment:

```bat
python -m pip install jupyterlab
jupyter lab
```

In a notebook started at the repository root:

```python
import pandas as pd
df = pd.read_parquet('data/processed/prepared.parquet')
df.shape
df.groupby('split').size()
df[['units', 'lag_7', 'lag_28', 'historical_sell_price']].isna().sum()
df.groupby('store_id')['id'].nunique()
```

Expected split counts: 73,000 training, 2,800 validation and 2,800 test records.
Some validation/test lags and all held-out selling prices are intentionally missing:
they exclude information that was unavailable at the forecast cutoff. Module 4
must construct recursive or direct multi-day model inputs without reading future
actual sales. Optional calendar events are legitimately nullable. Zero sales are
preserved and their cause remains unknown.

## 6. Docker reproduction
The learner verified Docker image construction, tests and the full pipeline on
October 12, 2026. The Windows and Docker output hashes matched. Docker execution
was unavailable in the original preparation workspace, so this later learner evidence
is recorded separately in docs/module3_evidence/learner_verification_2026-10-12.json.
Use Docker Desktop with the Linux engine to reproduce the container run.
Build from the repository root:

```bat
docker build -t retail-m5 .
docker run --rm --mount "type=bind,source=C:\YOUR_FOLDER\m5-forecasting-accuracy.zip,target=/inputs/m5.zip,readonly" --mount "type=bind,source=C:\YOUR_OUTPUT_FOLDER,target=/app/data/processed" retail-m5 --source /inputs/m5.zip
docker run --rm --entrypoint python retail-m5 -m pytest -q
```

Create the output directory first. On Linux/macOS use absolute native paths and
give the non-root container user UID 10001 write access to the output mount.
The input mount is read-only. Raw data never enters the image. Container build
and runtime success must be verified before claiming Docker reproduction.

## 7. Review before submission
Run these checks yourself and compare your results with the aggregate evidence
in docs/module3_evidence/. Keep a screenshot of your own Great Expectations report.
Read slide speaker notes for the explanation behind each pipeline stage.
Submit the 12-slide PowerPoint and the separately completed AI disclosure form;
the deck links to the repository's DAG, suite, tests, Dockerfile and governance files.
If local checks differ, record the discrepancy and correct it before claiming success.

## 8. Compare Windows and Docker outputs
Use separate data/processed and data/docker_processed folders so the two runs
can be compared. In PowerShell at the repository root, after both runs:

```powershell
Get-FileHash ".\data\processed\prepared.parquet", ".\data\docker_processed\prepared.parquet" -Algorithm SHA256 | Format-List Path, Hash
```

The learner's October 12 comparison produced identical hashes:
`2ef874771958b359766f1f10c10e146b230c3aa0041fcaca72baebc3e119dc58`.
This confirms identical output bytes for those two executions. It does not prove
future runs remain identical under all dependency or data changes.
