"""Verify Module 3 input and create local MLflow tracking before training."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import pandas as pd
import mlflow

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASH = '2ef874771958b359766f1f10c10e146b230c3aa0041fcaca72baebc3e119dc58'
EXPERIMENT = 'BAN6800_Module4_M5_Forecasting'

def inspect_data(path):
    if not path.is_file():
        raise FileNotFoundError(f'Prepared data not found: {path}. Copy your verified Module 3 prepared.parquet here or pass --data PATH.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_HASH:
        raise ValueError('Prepared file differs from the verified Module 3 file. Review its provenance before proceeding.')
    df = pd.read_parquet(path)
    counts = df.groupby('split').size().to_dict()
    if counts != {'train': 73000, 'validation': 2800, 'test': 2800}:
        raise ValueError(f'Unexpected split counts: {counts}')
    if df.id.nunique() != 100 or df.duplicated(['id', 'date']).any():
        raise ValueError('Expected 100 unique histories and unique id/date rows.')
    if df.units.isna().any() or (df.units < 0).any():
        raise ValueError('Invalid target values.')
    return {'sha256': digest, 'rows': len(df), 'series': df.id.nunique(),
            'splits': counts, 'columns': list(df.columns),
            'date_min': str(df.date.min()), 'date_max': str(df.date.max()),
            'stage': 'data_readiness_only', 'models_trained': 0,
            'holdout_labels_used_for_model_selection': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/processed/prepared.parquet')
    args = parser.parse_args()
    summary = inspect_data(args.data)
    tracking = ROOT/'tracking'
    tracking.mkdir(exist_ok=True)
    uri = 'sqlite:///' + (tracking/'mlflow.db').as_posix()
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        eid = client.create_experiment(EXPERIMENT, artifact_location=(tracking/'artifacts').as_uri())
    else:
        eid = experiment.experiment_id
    with mlflow.start_run(experiment_id=eid, run_name='setup_data_readiness',
                          tags={'phase': 'setup', 'module': '4', 'training_run': 'false'}):
        mlflow.log_params({'data_sha256': summary['sha256'], 'series': summary['series'],
                           'forecast_horizon_days': 28, 'random_seed': 42})
        mlflow.log_dict(summary, 'data_readiness.json')
        mlflow.log_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), 'environment.txt')
        print('Setup run ID:', mlflow.active_run().info.run_id)
    (tracking/'data_readiness.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('\nMLflow setup complete. No forecasting model has been trained.')
    print('Next, from this project folder:')
    print('python -m mlflow server --backend-store-uri sqlite:///tracking/mlflow.db --host 127.0.0.1 --port 5000')
    print('Then open http://127.0.0.1:5000')

if __name__ == '__main__':
    main()
