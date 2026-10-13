"""Check input reconstruction and grouped SHAP additivity on synthetic data."""
import numpy as np
import pandas as pd
from test_module4_forecasting import fixture_data
from module4 import train_models as t
from module4.explain_model import reconstruct_inputs,grouped_shap


def test_reconstruction_matches_recursive_predictions_and_ignores_future_sales():
    data=fixture_data();hist=data[data.split.eq('train')];valid=data[data.split.eq('validation')]
    model=t.build_model('gradient_boosting',{'max_iter':5,'max_leaf_nodes':5,'learning_rate':.1,'l2_regularization':1.0})
    model.fit(t.features(hist),hist.units)
    saved=t.forecast(model,hist,valid)
    inputs=reconstruct_inputs(data,saved)
    expected=inputs[['id','date']].merge(saved,on=['id','date']).prediction
    np.testing.assert_allclose(np.maximum(model.predict(t.features(inputs)),0),expected)
    data.loc[data.split.ne('train'),['units','lag_7','lag_28','rolling_mean_28']]=9999
    pd.testing.assert_frame_equal(inputs,reconstruct_inputs(data,saved))
    values,base,raw=grouped_shap(model,t.features(inputs).head(5))
    np.testing.assert_allclose(base+values.sum(axis=1),raw,atol=1e-5)
    assert values.shape==(5,len(t.FEATURES))


def test_full_explanation_outputs_are_logged(tmp_path,monkeypatch):
    import json,hashlib,joblib
    from module4 import explain_model as e
    data=fixture_data();hist=data[data.split.eq('train')];valid=data[data.split.eq('validation')]
    model=t.build_model('gradient_boosting',{'max_iter':5,'max_leaf_nodes':5,'learning_rate':.1,'l2_regularization':1.0})
    model.fit(t.features(hist),hist.units)
    parent='c'*32;folder=tmp_path/'models/module4'/parent;folder.mkdir(parents=True)
    joblib.dump(model,folder/'selected_validation_model.joblib')
    digest=hashlib.sha256((folder/'selected_validation_model.joblib').read_bytes()).hexdigest()
    (folder/'selection.json').write_text(json.dumps({'parent_run_id':parent}))
    (folder/'test').mkdir();(folder/'test/summary.json').write_text(json.dumps({'model_sha256':digest}))
    t.forecast(model,hist,valid).to_csv(folder/'selected_validation_predictions.csv',index=False)
    source=tmp_path/'data/processed/prepared.parquet';source.parent.mkdir(parents=True);data.to_parquet(source)
    (tmp_path/'tracking').mkdir()
    monkeypatch.setattr(e,'ROOT',tmp_path);monkeypatch.setattr(e,'inspect_data',lambda path:{'sha256':'synthetic_only'})
    e.run_analysis(parent)
    output=next((folder/'xai').iterdir())
    assert (output/'global.png').is_file() and (output/'local.png').is_file()
    summary=json.loads((output/'summary.json').read_text())
    assert summary['explanation_rows']==56
    assert summary['max_additivity_error']<1e-5
