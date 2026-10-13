"""Create residual plots, operational group review and Model Card from saved evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from module4.setup_tracking import ROOT,EXPERIMENT
from module4.train_models import group_scores


def markdown_table(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def latest_evidence(folder,stage,digest):
    files=list((folder/stage).glob('*/summary.json'))
    if not files:raise ValueError(f'Missing {stage} evidence')
    path=max(files,key=lambda p:p.stat().st_mtime)
    result=json.loads(path.read_text())
    if result['model_sha256']!=digest:raise ValueError(f'{stage} evidence uses a different model')
    return result


def assess_groups(predictions,baseline):
    groups=group_scores(predictions)
    other=group_scores(baseline)[['dimension','group','rmse']].rename(columns={'rmse':'baseline_rmse'})
    groups=groups.merge(other,on=['dimension','group'],validate='one_to_one')
    groups['rmse_improvement_percent']=100*(1-groups.rmse/groups.baseline_rmse.replace(0,np.nan))
    groups['small_sample_flag']=np.where(groups.dimension.eq('store_id'),groups.series<5,groups.series<10)
    # Descriptive diagnostic using the project aggregate-bias tolerance, not a demographic fairness test.
    groups['bias_review_flag']=groups.aggregate_bias_fraction.abs()>.10
    return groups


def render_plots(pred,groups,out):
    residual=pred.prediction-pred.units
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(pred.prediction,residual,alpha=.25,s=12,color='#008b8b')
    ax.axhline(0,color='#d05050');ax.set_xlabel('Predicted daily sales (units)');ax.set_ylabel('Prediction minus actual (units)')
    ax.set_title('Final-test residuals: positive = overprediction');fig.tight_layout();fig.savefig(out/'residual.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.hist(residual,bins=40,color='#008b8b',edgecolor='white')
    ax.axvline(0,color='#d05050');ax.set_xlabel('Prediction minus actual (units)');ax.set_ylabel('Forecast count')
    ax.set_title('Final-test error distribution');fig.tight_layout();fig.savefig(out/'errors.png',dpi=160);plt.close(fig)
    daily=pred.groupby('date')[['units','prediction']].sum()
    fig,ax=plt.subplots(figsize=(9,5));ax.plot(daily.index,daily.units,label='Actual',color='#23395b')
    ax.plot(daily.index,daily.prediction,label='Forecast',color='#008b8b');ax.set_ylabel('Total daily sales (selected 100 series)')
    ax.set_title('Final-test actual and forecast totals');ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'daily.png',dpi=160);plt.close(fig)
    stores=groups[groups.dimension.eq('store_id')].sort_values('aggregate_bias_fraction')
    fig,ax=plt.subplots(figsize=(8,5));ax.barh(stores.group,100*stores.aggregate_bias_fraction,color=np.where(stores.aggregate_bias_fraction>=0,'#008b8b','#d05050'))
    ax.axvline(0,color='gray');ax.set_xlabel('Total predicted minus actual sales (% of actual)');ax.set_title('Store-level forecast bias: final test')
    fig.tight_layout();fig.savefig(out/'stores.png',dpi=160);plt.close(fig)


def run_review(parent):
    folder=ROOT/'models/module4'/parent
    selection=json.loads((folder/'selection.json').read_text())
    test=json.loads((folder/'test/summary.json').read_text())
    digest=hashlib.sha256((folder/'selected_validation_model.joblib').read_bytes()).hexdigest()
    if selection['parent_run_id']!=parent or test['model_sha256']!=digest:raise ValueError('Evidence/model mismatch')
    xai=latest_evidence(folder,'xai',digest);sens=latest_evidence(folder,'sens',digest);cf=latest_evidence(folder,'cf',digest)
    predictions=pd.read_csv(folder/'test/pred.csv',parse_dates=['date'])
    baseline=pd.read_csv(folder/'test/base.csv',parse_dates=['date'])
    if len(predictions)!=test['test_rows'] or predictions.duplicated(['id','date']).any():raise ValueError('Invalid saved test predictions')
    groups=assess_groups(predictions,baseline)
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'tracking/mlflow.db').as_posix());mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name='frozen_model_documentation',tags={'phase':'documentation','development_run_id':parent}) as run:
        out=folder/'review'/run.info.run_id;out.mkdir(parents=True)
        mlflow.log_params({'model_sha256':digest,'test_run_id':test['test_run_id'],'test_rows':len(predictions),'model_refitted':False})
        mlflow.log_artifact(__file__,artifact_path='code')
        render_plots(predictions,groups,out)
        groups.to_csv(out/'groups.csv',index=False)
        display=groups[['dimension','group','series','mae','rmse','wape','aggregate_bias_fraction','rmse_improvement_percent','small_sample_flag','bias_review_flag']]
        disparities={}
        for dimension,group in groups.groupby('dimension'):
            disparities[dimension]={'mae_max_minus_min':float(group.mae.max()-group.mae.min()),
                'rmse_max_minus_min':float(group.rmse.max()-group.rmse.min()),
                'wape_max_minus_min':float(group.wape.max()-group.wape.min()),
                'aggregate_bias_max_minus_min':float(group.aggregate_bias_fraction.max()-group.aggregate_bias_fraction.min())}
        fairness=f'''# Operational group-performance review

This review describes forecast quality across stores and departments. It does not establish demographic fairness. Group metrics and regression loss are selected to assess differences in service quality; metric selection must reflect the application and available attributes (Fairlearn, n.d.).

## Results

{markdown_table(display)}

WAPE and aggregate bias are fractions in this table. Negative bias means underprediction. Small-sample flags use fewer than five store series or fewer than ten department series, consistent with the Module 3 representation review. The absolute 10% bias diagnostic applies the project's aggregate tolerance descriptively; it is not a protected-group fairness test or independently validated group acceptance criterion. Rows across dates are dependent and do not represent independent customers.

## Required demographic metrics and applicability

Demographic parity, equalized odds and disparate impact ratio are not estimated. M5 contains no customer-level protected attributes and this model outputs sales quantities, not a defined binary allocation decision. Store/state identifiers are not substitutes for customer demographics. No attributes or fairness scores are fabricated. The professor approved continued use of the dataset subject to demonstrating explainability; this is not treated as an explicit waiver of every fairness requirement.

## Interpretation and mitigation plan

Aggregate overprediction can offset underprediction elsewhere. These group results therefore constrain use even when the overall model beats the baseline. Underprediction could contribute to insufficient stock and overprediction to excess inventory, but neither stockouts nor carrying costs are measured. Group differences may reflect product mix, volume, history and cohort selection; they do not alone prove discriminatory treatment.

No post-test calibration, reweighting or retraining has been applied to the frozen model. Proposed operational controls are human review of flagged groups, reporting group-specific errors alongside totals and obtaining broader data and a new chronological evaluation period before developing mitigation. Customer-level fairness would require appropriate consented demographic data and a clearly defined downstream decision, with a separate assessment. The current evidence is limited to one 28-day test period and the selected household cohort.

## Source

Fairlearn. (n.d.). *Fairness in machine learning*. https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html (consulted October 13, 2026).
'''
        (out/'fairness.md').write_text(fairness)
        source_run=mlflow.MlflowClient().get_run(selection['source_run_id'])
        params=source_run.data.params
        performance=pd.read_csv(folder/'validation_comparison.csv')[['model','mae','rmse','r2']]
        test_table=pd.read_csv(folder/'test/scores.csv')[['model','mae','rmse','r2']]
        card=f'''# Model Card: M5 household sales forecasting

## Model details

Academic project owner: Abdul Basit Sheikh. Algorithm: {selection['best_learned_model']} from scikit-learn, with one-hot categorical encoding and recursive 28-day forecasting. Model SHA-256: `{digest}`. Development run: `{parent}`. Training settings recorded in MLflow: `{json.dumps(params,sort_keys=True)}`. This card follows the documentation principles of Mitchell et al. (2019).

## Intended use

Educational comparison of daily recorded-sales forecasts for the fixed 100 product–store household series. Results may inform future inventory scenarios under explicit assumptions. The model is not approved for autonomous stock ordering, customer decisions, causal pricing recommendations or generalization to all M5 products/stores. Actual inventory levels, replenishment lead times, costs, lost sales and weather are unavailable; stockout and storage-cost savings have not been demonstrated.

## Data and development

Source: Kaggle M5 Forecasting Accuracy historical sales, calendar and price files, prepared by the Module 3 pipeline. Verified data SHA-256: `{test['data_sha256']}`. Training comprises 73,000 rows through March 27, 2016; validation comprises 2,800 rows from March 28–April 24; final test comprises 2,800 rows from April 25–May 22. Hyperparameters were compared at two chronological forecast origins within training before validation model selection. The cohort was fixed using the full training period, so rolling-origin comparisons are conditional on that selection. Zero sales are retained with unknown cause; they are not labelled as confirmed stockouts.

## Evaluation

Validation comparison:

{markdown_table(performance)}

Frozen-model final test:

{markdown_table(test_table)}

Final-test RMSE improvement over the weekly baseline: {100*test['relative_rmse_improvement']:.2f}%. Final-test WAPE: {100*test['metrics']['wape']:.2f}%. Aggregate forecast bias: {100*test['metrics']['aggregate_bias_fraction']:.2f}%. The model was not refitted after selection. Observed sales through April 24 provided legitimate forecast history; test sales were joined only after predictions. Prices remained at the last available training value. Test results are final evidence, not tuning feedback.

## Explainability and robustness

TreeSHAP explains all {xai['explanation_rows']} validation forecast snapshots. Largest influences: {', '.join(r['feature'] for r in xai['top_features'])}. Grouped contributions reconstruct raw output with maximum error {xai['max_additivity_error']:.3g}. Correlated inputs and the tree-path reference affect attribution. Raw predictions are clamped at zero for served forecasts.

Sensitivity varies history fields jointly and price separately at fixed forecast snapshots; it is not a new recursive forecast or causal price elasticity. Training-range extrapolation is reported. DiCE searched the fixed local SHAP example for a 10–20% higher prediction using bounded, coherent history scaling. A qualifying alternative was {'found' if cf['found'] else 'not found'} within the declared search budget. A null result does not prove infeasibility. Historical inputs are explanatory alternatives, not actionable recourse. No model changes were made.

## Fairness and limitations

`fairness.md` and `groups.csv` show store/department MAE, RMSE, WAPE, bias, baseline comparisons and sample support. Protected-attribute demographic metrics cannot be established from this dataset and are not fabricated. Small groups and substantial underprediction in some groups limit operational use despite favourable aggregate results. Model performance reflects recorded sales, which can be constrained by unobserved availability. No uncertainty intervals, causal claims, customer fairness guarantees or observed cost savings are provided.

## Monitoring and governance

Proposed checks: input completeness, unseen IDs, historical-price age, data/feature drift, overall and group MAE/RMSE/WAPE/bias after labels arrive, and forecast changes against the weekly baseline. Proposed aggregate acceptance targets were at least 5% RMSE improvement and absolute aggregate bias no greater than 10%; passing these does not approve deployment. Require human review of group disparities, validation on a fresh temporal period for any revised model and explicit inventory-policy testing before operational adoption. Registry and API implementation are subsequent project deliverables. Production approval remains false.

## Evidence and sources

Performance and interpretation are project results from saved predictions and run artifacts, not claims derived from external sources. Final-test run: `{test['test_run_id']}`. SHAP run: `{xai['xai_run_id']}`. Sensitivity run: `{sens['sensitivity_run_id']}`. DiCE run: `{cf['counterfactual_run_id']}`.

Mitchell, M., et al. (2019). *Model cards for model reporting*. https://doi.org/10.1145/3287560.3287596

Fairlearn. (n.d.). *Fairness in machine learning*. https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html

SHAP. (n.d.). *TreeExplainer*. https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html

DiCE. (n.d.). *Generate diverse counterfactual explanations*. https://github.com/interpretml/DiCE

Kaggle. (n.d.). *M5 Forecasting Accuracy*. https://www.kaggle.com/competitions/m5-forecasting-accuracy

Online documentation consulted October 13, 2026.
'''
        (out/'model_card.md').write_text(card)
        summary={'review_run_id':run.info.run_id,'development_run_id':parent,'model_sha256':digest,
                 'final_test_run_id':test['test_run_id'],'test_rows':len(predictions),
                 'flagged_groups':display[display.small_sample_flag|display.bias_review_flag].to_dict('records'),
                 'group_metric_disparities':disparities,'protected_attribute_fairness_assessed':False,
                 'model_refitted':False,'deployment_approved':False}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        for path in out.iterdir():mlflow.log_artifact(str(path))
        print('\nDOCUMENTATION COMPLETE')
        print('Created: residual.png, errors.png, daily.png, stores.png, model_card.md, fairness.md, groups.csv, summary.json')
        print('\nGROUP REVIEW FLAGS\n'+display[display.small_sample_flag|display.bias_review_flag].to_string(index=False))
        print('\nResults saved in:',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run',required=True)
    args=parser.parse_args()
    if len(args.parent_run)!=32 or any(c not in '0123456789abcdef' for c in args.parent_run):parser.error('Invalid run ID')
    run_review(args.parent_run)


if __name__=='__main__':main()
