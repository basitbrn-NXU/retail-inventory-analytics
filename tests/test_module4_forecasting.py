"""Synthetic checks for forecast leakage and logged experiment execution."""
import json
import numpy as np
import pandas as pd
import pytest
from module4 import train_models as m


def fixture_data():
    dates=pd.date_range('2014-03-29','2016-05-22')
    frames=[]
    for j in range(2):
        d=pd.DataFrame({'date':dates,'id':f's{j}','item_id':f'i{j}',
                        'store_id':f'S{j}','dept_id':'H','state_id':'CA'})
        d['units']=5+j+d.date.dt.dayofweek.to_numpy()
        d['wday']=d.date.dt.dayofweek+1;d['month']=d.date.dt.month;d['snap']=0
        d['event_name_1']=None;d['event_name_2']=None
        d['lag_7']=d.units.shift(7).fillna(5+j)
        d['lag_28']=d.units.shift(28).fillna(5+j)
        d['rolling_mean_28']=d.units.shift(1).rolling(28,min_periods=1).mean().fillna(5+j)
        d['historical_sell_price']=2.0+j
        d['split']=np.where(d.date<=m.TRAIN_END,'train',np.where(d.date<=m.VALID_END,'validation','test'))
        frames.append(d)
    return pd.concat(frames,ignore_index=True)


def test_weekly_baseline_and_future_labels_cannot_change_forecast():
    df=fixture_data();hist=df[df.split.eq('train')];future=df[df.split.eq('validation')].copy()
    expected=m.forecast(None,hist,future)
    future['units']=999999;future['lag_7']=-999;future['historical_sell_price']=50000
    pd.testing.assert_frame_equal(expected,m.forecast(None,hist,future))
    merged=expected.merge(df[df.split.eq('validation')][['id','date','units']],on=['id','date'])
    np.testing.assert_array_equal(merged.prediction,merged.units)


def test_learned_forecast_ignores_future_labels_and_derived_fields():
    df=fixture_data();hist=df[df.split.eq('train')];future=df[df.split.eq('validation')].copy()
    model=m.build_model('random_forest',{'n_estimators':3,'max_depth':3,'min_samples_leaf':2})
    model.fit(m.features(hist),hist.units)
    expected=m.forecast(model,hist,future)
    future['units']=0;future['lag_28']=1e6;future['rolling_mean_28']=1e6
    future['historical_sell_price']=1e6;future['zero_sales_cause']='invented'
    pd.testing.assert_frame_equal(expected,m.forecast(model,hist,future))


def test_calendar_gap_rejected():
    df=fixture_data();hist=df[df.split.eq('train')];future=df[df.split.eq('validation')].copy()
    future.loc[future.date.eq(future.date.min()),'date']-=pd.Timedelta(days=1)
    with pytest.raises(ValueError):m.forecast(None,hist,future)


def test_tracked_workflow_uses_no_test_targets(tmp_path,monkeypatch):
    df=fixture_data();df.loc[df.split.eq('test'),'units']=np.nan
    source=tmp_path/'fixture.parquet';df.to_parquet(source)
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'inspect_data',lambda path:{'sha256':'synthetic_fixture_only'})
    monkeypatch.setattr(m,'CANDIDATES',{
        'random_forest':[{'n_estimators':3,'max_depth':3,'min_samples_leaf':2}],
        'gradient_boosting':[{'max_iter':3,'max_leaf_nodes':5,'learning_rate':0.1,'l2_regularization':1.0}]})
    (tmp_path/'tracking').mkdir()
    m.mlflow.set_tracking_uri('sqlite:///'+(tmp_path/'tracking/mlflow.db').as_posix())
    m.mlflow.set_experiment(m.EXPERIMENT)
    m.run_experiment(source,tmp_path/'models')
    decisions=list((tmp_path/'models').glob('*/selection.json'))
    assert len(decisions)==1
    assert json.loads(decisions[0].read_text())['final_test_evaluated'] is False
    experiment=m.mlflow.get_experiment_by_name(m.EXPERIMENT)
    runs=m.mlflow.search_runs([experiment.experiment_id])
    assert len(runs)==8
    assert set(runs.status)=={'FINISHED'}
