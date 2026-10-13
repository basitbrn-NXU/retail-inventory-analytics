"""Verify sensitivity scenarios alter only declared inputs and measure known changes."""
import numpy as np
import pandas as pd
from module4.sensitivity import perturb,sensitivity_table,HISTORY_FEATURES

class KnownModel:
    def predict(self,x):return x.rolling_mean_28.to_numpy()+.1*x.historical_sell_price.to_numpy()


def test_changes_are_joint_and_inputs_remain_unchanged():
    x=pd.DataFrame({'lag_7':[10.],'lag_28':[20.],'rolling_mean_28':[15.],'historical_sell_price':[2.],'store_id':['CA_1']})
    original=x.copy()
    altered=perturb(x,'history',.2)
    np.testing.assert_allclose(altered[HISTORY_FEATURES],[[12.,24.,18.]])
    assert altered.historical_sell_price.iloc[0]==2
    assert altered.store_id.iloc[0]=='CA_1'
    pd.testing.assert_frame_equal(original,x)
    assert perturb(x,'price',-.1).historical_sell_price.iloc[0]==1.8


def test_table_reports_known_changes_and_training_range_screen():
    x=pd.DataFrame({'lag_7':[10.],'lag_28':[20.],'rolling_mean_28':[15.],'historical_sell_price':[2.]})
    train=x.assign(id='s')
    table,rows=sensitivity_table(KnownModel(),x,pd.Series(['s']),train)
    history=table[table.scenario.eq('history_+20%')].iloc[0]
    assert np.isclose(history.mean_prediction_change_units,3.)
    assert history.outside_training_range_fraction==1.
    price=table[table.scenario.eq('price_+10%')].iloc[0]
    assert np.isclose(price.mean_prediction_change_units,.02)
    assert len(rows)==6


def test_full_workflow_reconstructs_inputs_instead_of_csv(tmp_path,monkeypatch):
    import json,hashlib,joblib
    from test_module4_forecasting import fixture_data
    from module4 import train_models as t
    from module4 import sensitivity as s
    data=fixture_data();hist=data[data.split.eq('train')];valid=data[data.split.eq('validation')]
    model=t.build_model('gradient_boosting',{'max_iter':5,'max_leaf_nodes':5,'learning_rate':.1,'l2_regularization':1.})
    model.fit(t.features(hist),hist.units)
    parent='d'*32;xid='e'*32
    folder=tmp_path/'models/module4'/parent;xai=folder/'xai'/xid;xai.mkdir(parents=True)
    joblib.dump(model,folder/'selected_validation_model.joblib')
    digest=hashlib.sha256((folder/'selected_validation_model.joblib').read_bytes()).hexdigest()
    (xai/'summary.json').write_text(json.dumps({'model_sha256':digest,'development_run_id':parent}))
    # An invalid CSV must not influence the reconstructed forecast inputs.
    (xai/'inputs.csv').write_text('wrong,values\n999,999\n')
    t.forecast(model,hist,valid).to_csv(folder/'selected_validation_predictions.csv',index=False)
    source=tmp_path/'data/processed/prepared.parquet';source.parent.mkdir(parents=True);data.to_parquet(source)
    (tmp_path/'tracking').mkdir()
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'inspect_data',lambda path:{'sha256':'synthetic_only'})
    s.run_analysis(parent,xid)
    output=next((folder/'sens').iterdir())
    assert (output/'sens.png').is_file()
    summary=json.loads((output/'summary.json').read_text())
    assert summary['rows']==56 and len(summary['scenarios'])==6
