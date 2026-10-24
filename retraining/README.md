# New-period challenger workflow

This watcher automates candidate creation when a changed private Parquet file supplies a complete 28-day labelled window starting after 22 May 2016. It does not promote models or modify the served version, frozen artifact, or registry aliases. The original test window cannot count as fresh evaluation.

Run from the repository root in the Module 4 MLflow environment:

```powershell
python -m retraining.watch --data C:/private/prepared.parquet --once
# Omit --once to poll hourly while this process remains running.
python -m retraining.candidate --data C:/private/prepared.parquet --window-start 2016-06-01
```

The input retains the exact original cohort and schema, at least 84 prior calendar days and 28 complete recent days for every history. Labels are separated before training; lag and rolling features are rebuilt from prior history. Changed identifiers, missing windows, negative targets, and duplicate keys stop processing. Calendar information must be known at the forecast origin. Do not upload the private source or generated model to Git.

A clone of the frozen model configuration is fit to available prior history. Champion, challenger and weekly baseline are scored recursively on the new window. Gates require at least 10% baseline-relative WAPE gain, no worse WAPE than the champion, absolute pooled bias within 10%, and no store/department absolute-bias deterioration above two percentage points. Group counts and contextual review remain necessary; passing these gates does not establish demographic fairness or business value.

The workflow writes candidate hashes, data hash, time cutoffs, scores and review status to MLflow and a local review JSON. Production approval remains false. A named learner/model owner must review support, inventory value, data rights, privacy, explanations, rollback and compatibility before a separate release decision. The watcher is an optional local process, not a provisioned cloud scheduler.

Validation: five boundary tests passed alongside five inventory accounting tests on 24 October 2026. No eligible new labelled period was supplied, so a newly trained challenger and end-to-end MLflow logging are not claimed as verified. The additional offline bundle contains a separate retrospective challenger; it is not used for serving or fresh-period evidence.
