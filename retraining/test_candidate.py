from pathlib import Path
import pandas as pd
import pytest
from retraining.candidate import validate_window,rebuild_features,gates
from retraining.watch import check


def sample():
    dates=pd.date_range("2016-03-01",periods=120)
    data=pd.DataFrame([dict(id="x",item_id="a",store_id="s",dept_id="d",state_id="z",date=d,units=i%9,
                           wday=1,month=d.month,snap=0,event_name_1="No_event",event_name_2="No_event",historical_sell_price=2.) for i,d in enumerate(dates)])
    return data,[dict(item_id="a",store_id="s",dept_id="d",state_id="z")]


def test_original_test_is_not_a_fresh_window():
    data,cohort=sample()
    with pytest.raises(ValueError,match="original held-out"):validate_window(data,"2016-04-25",cohort)


def test_valid_new_window_separates_history_and_labels():
    data,cohort=sample();history,window=validate_window(data,"2016-06-01",cohort)
    assert history.date.max()<window.date.min()
    assert window.date.nunique()==28
    changed=data.copy();changed.loc[changed.date.ge("2016-06-01"),"units"]=900
    old,_=validate_window(changed,"2016-06-01",cohort)
    pd.testing.assert_frame_equal(rebuild_features(history),rebuild_features(old))


def test_duplicate_and_incomplete_windows_fail():
    data,cohort=sample()
    with pytest.raises(ValueError):validate_window(pd.concat([data,data.iloc[[0]]]),"2016-06-01",cohort)
    with pytest.raises(ValueError):validate_window(data.iloc[:-1],"2016-06-01",cohort)


def test_group_deterioration_blocks_gate():
    s={"baseline":{"wape":1.},"champion":{"wape":.8},"challenger":{"wape":.7,"aggregate_bias_fraction":.01}}
    g=gates(s,{"s":.10},{"s":.20})
    assert g["not_worse_than_champion_wape"] is True
    assert g["no_group_absolute_bias_deterioration_above_2points"] is False


def test_watcher_waits_without_new_data(tmp_path):
    assert check(tmp_path/"missing.parquet",tmp_path/"state.json",tmp_path,"sqlite:///none.db")["status"]=="waiting_for_private_dataset"
