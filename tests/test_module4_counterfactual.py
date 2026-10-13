"""Verify DiCE finds a bounded alternative without breaking history consistency."""
import numpy as np
import pandas as pd
from module4.counterfactual import HistoryScaleModel,search,HISTORY

class KnownModel:
    def predict(self,x):return x.rolling_mean_28.to_numpy()


def test_joint_scaling_keeps_other_inputs_fixed():
    query=pd.DataFrame({'lag_7':[8.],'lag_28':[12.],'rolling_mean_28':[10.],'store_id':['CA_1'],'historical_sell_price':[2.]})
    class CheckingModel:
        def predict(self,x):
            assert (x.store_id=='CA_1').all() and (x.historical_sell_price==2.).all()
            np.testing.assert_allclose(x.lag_7/x.rolling_mean_28,.8)
            np.testing.assert_allclose(x.lag_28/x.rolling_mean_28,1.2)
            return x.rolling_mean_28.to_numpy()
    wrapper=HistoryScaleModel(CheckingModel(),query)
    np.testing.assert_allclose(wrapper.predict(pd.DataFrame({'history_scale':[1.,1.2]})),[10.,12.])


def test_dice_solution_hits_target_and_respects_bounds():
    query=pd.DataFrame({'lag_7':[8.],'lag_28':[12.],'rolling_mean_28':[10.]})
    train=pd.DataFrame({'lag_7':[4.,16.],'lag_28':[6.,24.],'rolling_mean_28':[5.,20.]})
    info,candidates=search(KnownModel(),query,train)
    assert info['found']
    assert candidates.verified_prediction.between(11.,12.).all()
    assert candidates.history_scale.between(.5,1.5).all()
    assert np.isclose(info['nearest_found_prediction'],10*info['nearest_found_scale'])
