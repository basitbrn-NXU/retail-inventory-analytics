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
