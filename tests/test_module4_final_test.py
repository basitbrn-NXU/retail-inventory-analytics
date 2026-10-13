"""Verify frozen-model final-test forecasting cannot see future labels."""
import pandas as pd
from module4 import train_models as training
from module4.evaluate_test import final_predictions
from test_module4_forecasting import fixture_data


def test_final_forecast_uses_past_observations_and_ignores_test_targets():
    data=fixture_data()
    history=data[data.split.eq('train')]
    model=training.build_model('random_forest',{'n_estimators':3,'max_depth':3,'min_samples_leaf':2})
    model.fit(training.features(history),history.units)
    learned,baseline=final_predictions(model,data)
    mutated=data.copy()
    is_test=mutated.split.eq('test')
    mutated.loc[is_test,['units','lag_7','lag_28','rolling_mean_28','historical_sell_price']]=999999
    other_learned,other_baseline=final_predictions(model,mutated)
    pd.testing.assert_frame_equal(learned,other_learned)
    pd.testing.assert_frame_equal(baseline,other_baseline)
    # The baseline starts from the last observed week in validation, not old training dates.
    data.loc[data.date.gt(pd.Timestamp('2016-04-17')) & ~is_test,'units']=123
    _,updated=final_predictions(model,data)
    assert (updated.prediction==123).all()


def test_final_evaluation_logs_results_and_blocks_duplicate(tmp_path,monkeypatch):
    import json
    import joblib
    import pytest
    from module4 import evaluate_test as final
    data=fixture_data()
    source=tmp_path/'data/processed/prepared.parquet'
    source.parent.mkdir(parents=True)
    data.to_parquet(source)
    parent='a'*32
    folder=tmp_path/'models/module4'/parent
    folder.mkdir(parents=True)
    selection={'parent_run_id':parent,'best_learned_model':'random_forest',
               'validation_winner_including_baseline':'random_forest','source_run_id':'b'*32}
    (folder/'selection.json').write_text(json.dumps(selection))
    train=data[data.split.eq('train')]
    model=training.build_model('random_forest',{'n_estimators':3,'max_depth':3,'min_samples_leaf':2})
    model.fit(training.features(train),train.units)
    joblib.dump(model,folder/'selected_validation_model.joblib')
    (tmp_path/'tracking').mkdir()
    monkeypatch.setattr(final,'ROOT',tmp_path)
    monkeypatch.setattr(final,'inspect_data',lambda path:{'sha256':'synthetic_only'})
    final.run_test(parent)
    summary=json.loads((folder/'test/summary.json').read_text())
    assert summary['final_test_evaluated'] is True
    assert summary['refit'] is False
    assert summary['test_rows']==56
    assert summary['deployment_approved'] is False
    with pytest.raises(ValueError,match='already exists'):
        final.run_test(parent)
