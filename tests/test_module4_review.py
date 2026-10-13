"""Check error-direction flags and documented group support."""
import numpy as np
import pandas as pd
from module4.review import assess_groups,markdown_table


def test_group_flags_distinguish_bias_and_small_samples():
    pred=pd.DataFrame({'id':['a','b','c','d','e','f'], 'date':pd.Timestamp('2016-04-25'),
                       'store_id':['S1']*5+['S2'],'dept_id':['H1']*6,'units':[10.]*6,
                       'prediction':[8.]*5+[10.]})
    baseline=pred.assign(prediction=5.)
    result=assess_groups(pred,baseline)
    s1=result[result.group.eq('S1')].iloc[0];s2=result[result.group.eq('S2')].iloc[0]
    assert bool(s1.bias_review_flag) and not bool(s1.small_sample_flag)
    assert not bool(s2.bias_review_flag) and bool(s2.small_sample_flag)
    assert np.isclose(s1.aggregate_bias_fraction,-.2)
    assert np.isclose(s1.rmse_improvement_percent,60.)
    assert '| group |' in markdown_table(pd.DataFrame({'group':['S1']}))


def test_review_creates_plots_card_and_honest_null_counterfactual(tmp_path,monkeypatch):
    import json,hashlib
    from module4 import review as r
    parent='f'*32;folder=tmp_path/'models/module4'/parent
    (folder/'test').mkdir(parents=True);(tmp_path/'tracking').mkdir()
    model=folder/'selected_validation_model.joblib';model.write_bytes(b'synthetic software-test model hash only')
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    r.mlflow.set_tracking_uri('sqlite:///'+(tmp_path/'tracking/mlflow.db').as_posix());r.mlflow.set_experiment(r.EXPERIMENT)
    with r.mlflow.start_run(run_name='synthetic_selection') as run:
        source_run=run.info.run_id;r.mlflow.log_param('synthetic_fixture',True)
    selection={'parent_run_id':parent,'source_run_id':source_run,'best_learned_model':'gradient_boosting'}
    (folder/'selection.json').write_text(json.dumps(selection))
    test={'model_sha256':digest,'test_rows':6,'test_run_id':'synthetic_test','data_sha256':'synthetic_data',
          'relative_rmse_improvement':.2,'metrics':{'wape':.5,'aggregate_bias_fraction':-.1}}
    (folder/'test/summary.json').write_text(json.dumps(test))
    pred=pd.DataFrame({'id':list('abcdef'),'date':pd.Timestamp('2016-04-25'),
         'store_id':['S1']*3+['S2']*3,'dept_id':['H1']*6,'units':[10.]*6,'prediction':[8.]*6})
    pred.to_csv(folder/'test/pred.csv',index=False);pred.assign(prediction=5.).to_csv(folder/'test/base.csv',index=False)
    scores=pd.DataFrame({'model':['gradient_boosting','weekly_baseline'],'mae':[2.,5.],'rmse':[2.,5.],'r2':[0.,0.]})
    scores.to_csv(folder/'validation_comparison.csv',index=False);scores.to_csv(folder/'test/scores.csv',index=False)
    for stage,meta in [('xai',{'explanation_rows':56,'top_features':[{'feature':'rolling_mean_28'}],'max_additivity_error':1e-14,'xai_run_id':'synthetic_xai'}),
                       ('sens',{'sensitivity_run_id':'synthetic_sens'}),('cf',{'counterfactual_run_id':'synthetic_cf','found':False})]:
        path=folder/stage/'fixture';path.mkdir(parents=True)
        (path/'summary.json').write_text(json.dumps({'model_sha256':digest,**meta}))
    monkeypatch.setattr(r,'ROOT',tmp_path)
    r.run_review(parent)
    out=next((folder/'review').iterdir())
    assert all((out/name).is_file() for name in ['residual.png','errors.png','daily.png','stores.png','model_card.md','fairness.md'])
    assert 'not found' in (out/'model_card.md').read_text()
    summary=json.loads((out/'summary.json').read_text())
    assert summary['deployment_approved'] is False
    assert summary['protected_attribute_fairness_assessed'] is False
