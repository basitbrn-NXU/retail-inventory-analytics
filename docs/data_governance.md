# Data governance and anonymization

Owner: Abdul Basit Sheikh, academic analyst. Scope: M5 retail forecasting prototype.

## Access and permitted use
Source data is acquired directly from Kaggle under the user's accepted terms.
The pipeline accepts a local ZIP or a directory containing the three CSVs.
Raw CSVs, prepared Parquet and detailed application logs remain in access-controlled
local storage. Public GitHub contains code, cohort identifiers, source hashes and
aggregate execution evidence. Users with GitHub write access review changes.
Raw files are excluded from both Git and the Docker build context. The container
runs as a non-root user and reads a read-only source mount. It requires no
Kaggle password, API token, customer information or production credentials.

## Anonymization plan
M5 inputs identify products and stores rather than individual customers.
The Python privacy stage removes columns explicitly named name, full_name,
email, phone, address, customer_id, national_id, passport or ip_address and
rejects contact-like values in retained text columns. Tests demonstrate both paths.
The actual M5 run removes zero PII columns. This is a screening safeguard, not
a guarantee of complete PII detection for arbitrary future datasets. New data
requires schema review and a privacy assessment before ingestion. If joins create
identifiable persons, use approved keyed pseudonymization, separate key storage,
and aggregate releases; do not label pseudonymized records anonymous.

## Logging and retention
The append-only JSONL application ledger records all explicit pipeline source
reads, transformations, validation, local publication and verification reads.
It records UTC time, run ID, actor role, stage, resource label, counts and outcomes.
It omits record values, personal names, credentials and absolute input paths.
Pipeline failures log the exception type. This covers application operations,
not arbitrary OS-level access or administrator activity; production deployment
requires host access logs and a protected central audit store.
Detailed logs and prepared records remain local through grading and any appeal
period, then unnecessary copies are deleted in accordance with university policy
and source terms. The institution's retention duration must be confirmed.
The local log is append-only by convention, not tamper-proof. Production access
would restrict alteration and provide retention-controlled archival storage.

## Quality and representation review
Source hashes lock the same Module 2 dataset. Unexpected identities, invalid
sales, duplicate keys, missing dates and failed Great Expectations checks stop
publication. Optional event fields remain nullable. Recorded zero sales stay
zero with an unknown cause. Missing selling prices stay unknown.
Fairlearn reports selection rates and counts across stores and departments.
Review thresholds are fewer than five selected series per store and fewer than
ten per department. These are prototype representation review thresholds,
not demographic fairness guarantees. The CA_4 and HOUSEHOLD_2 limitations
carry forward into Module 4 group performance analysis. Protected groups
cannot be assessed without appropriate demographic data.

## Responsibilities and change control
The learner reviews run results and limitations; the instructor reviews academic
evidence. A proposed business pilot assigns finance to cost assumptions,
planners to usefulness, and compliance to data-use approval. These roles have
not approved an operational deployment. Code changes use reviewed branches;
run lineage captures code hashes, source hashes, versions and split dates.
