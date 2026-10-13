"""Evaluate the frozen validation winner once, without fitting or tuning a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import mlflow
import pandas as pd
from module4.train_models import KNOWN_FUTURE, VALID_END, forecast, evaluate, group_scores
from module4.setup_tracking import ROOT, EXPERIMENT, inspect_data


def final_predictions(model, data):
    """Use observed history through April 24; test inputs are calendar/IDs only."""
    history = data.loc[data.date <= VALID_END].copy()
    future = data.loc[data['split'].eq('test'), KNOWN_FUTURE].copy()
    if history.date.max() != VALID_END or future.date.max() != pd.Timestamp('2016-05-22'):
        raise ValueError('Unexpected final-test dates.')
    return forecast(model, history, future), forecast(None, history, future)


def run_test(parent_run_id):
    folder = ROOT / 'models/module4' / parent_run_id
    selection = json.loads((folder / 'selection.json').read_text())
    if selection['parent_run_id'] != parent_run_id:
        raise ValueError('Selection does not match the requested development run.')
    if selection['validation_winner_including_baseline'] != selection['best_learned_model']:
        raise ValueError('Baseline won validation; review the frozen selection before using this script.')
    output = folder / 'test'
    if output.exists():
        raise ValueError('Final-test output already exists. Keep it as evidence; do not repeat test-based selection.')
    model_path = folder / 'selected_validation_model.joblib'
    # Only the project-owned model saved by train_models is loaded here.
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = joblib.load(model_path)
    data_path = ROOT / 'data/processed/prepared.parquet'
    readiness = inspect_data(data_path)
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data.date)
    mlflow.set_tracking_uri('sqlite:///' + (ROOT / 'tracking/mlflow.db').as_posix())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name='frozen_model_final_test', tags={
            'phase':'final_test', 'development_run_id':parent_run_id,
            'algorithm':selection['best_learned_model'], 'model_refitted':'false'}) as run:
        output.mkdir()
        protocol = {
            'model':selection['best_learned_model'], 'source_run_id':selection['source_run_id'],
            'development_run_id':parent_run_id, 'model_sha256':model_hash,
            'data_sha256':readiness['sha256'], 'forecast_origin':'2016-04-24',
            'test_start':'2016-04-25', 'test_end':'2016-05-22', 'horizon_days':28,
            'model_fit_end':'2016-03-27', 'refit':False, 'retuning':False,
            'history_policy':'Observed training and validation sales available at forecast origin; future test sales excluded from prediction inputs.',
            'price_policy':'Last available historical price carried forward; Module 3 withheld validation and test prices.',
            'interpretation':'Frozen-model holdout evaluation, not a retraining or production deployment.'}
        (output / 'freeze.json').write_text(json.dumps(protocol, indent=2))
        mlflow.log_dict(protocol, 'freeze.json')
        mlflow.log_artifact(__file__, artifact_path='code')
        mlflow.log_params({'model_sha256':model_hash, 'data_sha256':readiness['sha256'],
                           'forecast_origin':'2016-04-24', 'horizon_days':28, 'refit':False})
        print('Forecasting the final 28 days using the saved model...', flush=True)
        learned, baseline = final_predictions(model, data)
        # Test labels are joined only after predictions are generated.
        labels = data.loc[data['split'].eq('test'), ['id','date','units']]
        metrics, joined = evaluate(learned, labels)
        baseline_metrics, baseline_joined = evaluate(baseline, labels)
        mlflow.log_metrics({'test_'+key:value for key,value in metrics.items()})
        mlflow.log_metrics({'baseline_'+key:value for key,value in baseline_metrics.items()})
        table = pd.DataFrame([{'model':selection['best_learned_model'], **metrics},
                              {'model':'weekly_baseline', **baseline_metrics}])
        for name, frame in [('scores.csv',table), ('pred.csv',joined),
                            ('base.csv',baseline_joined), ('groups.csv',group_scores(joined))]:
            path = output / name
            frame.to_csv(path,index=False)
            mlflow.log_artifact(str(path))
        improvement = 1-metrics['rmse']/baseline_metrics['rmse'] if baseline_metrics['rmse']>0 else None
        summary = {**protocol, 'test_run_id':run.info.run_id, 'test_rows':len(joined),
                   'metrics':metrics, 'baseline_metrics':baseline_metrics,
                   'relative_rmse_improvement':improvement, 'final_test_evaluated':True,
                   'passes_accuracy_and_bias_targets':bool(improvement is not None and improvement>=0.05
                        and abs(metrics.get('aggregate_bias_fraction',float('inf')))<=0.10),
                   'deployment_approved':False,
                   'remaining':'Explainability, sensitivity/counterfactual analysis, group review, Model Card, registry and API.'}
        (output / 'summary.json').write_text(json.dumps(summary,indent=2))
        mlflow.log_dict(summary,'summary.json')
        print('\nFINAL TEST COMPARISON\n'+table[['model','mae','rmse','r2','mean_signed_error']].to_string(index=False))
        print('\n'+json.dumps(summary,indent=2))
        print('\nResults saved in:',output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run',required=True,help='Completed forecast_model_development run ID')
    args=parser.parse_args()
    if len(args.parent_run)!=32 or any(c not in '0123456789abcdef' for c in args.parent_run):
        parser.error('Expected a 32-character lowercase hexadecimal run ID.')
    run_test(args.parent_run)


if __name__=='__main__':
    main()
