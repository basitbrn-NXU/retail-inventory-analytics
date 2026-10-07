import json
import numpy as np
import pandas as pd
import pytest
from src.prepare import (TRAIN_START, TRAIN_END, TEST_END, prepare_features,
                         integrate, anonymize, Audit, read_inputs)

def daily():
    dates=pd.date_range(TRAIN_START-pd.Timedelta(days=28),TEST_END)
    return pd.DataFrame(dict(id='series',item_id='item',dept_id='HOUSEHOLD_1',cat_id='HOUSEHOLD',store_id='CA_1',state_id='CA',
        date=dates,d=['d_'+str(i) for i in range(len(dates))],wm_yr_wk=1,
        units=np.arange(len(dates))%13,sell_price=2.5,wday=dates.dayofweek+1,month=dates.month,year=dates.year,
        event_name_1=None,event_type_1=None,event_name_2=None,event_type_2=None,snap_CA=0,snap_TX=0,snap_WI=0))

def test_holdout_target_mutation_cannot_change_features():
    source=daily(); changed=source.copy()
    changed.loc[changed.date>TRAIN_END,'units']=999999
    a,b=prepare_features(source),prepare_features(changed)
    pd.testing.assert_frame_equal(a[['lag_7','lag_28','rolling_mean_28']],b[['lag_7','lag_28','rolling_mean_28']])

def test_current_target_does_not_enter_same_day_features():
    source=daily(); changed=source.copy(); day=TRAIN_START+pd.Timedelta(days=100)
    changed.loc[changed.date.eq(day),'units']=5000
    a,b=prepare_features(source),prepare_features(changed)
    pd.testing.assert_frame_equal(a.loc[a.date.eq(day),['lag_7','rolling_mean_28']],b.loc[b.date.eq(day),['lag_7','rolling_mean_28']])

def test_future_price_is_masked():
    frame=prepare_features(daily())
    assert frame.loc[frame.date>TRAIN_END,'historical_sell_price'].isna().all()

def test_split_is_chronological():
    frame=prepare_features(daily())
    assert frame.groupby('split').size().to_dict()=={'test':28,'train':730,'validation':28}

def test_zero_labels_are_retained():
    source=daily(); frame=prepare_features(source)
    assert frame.units.eq(0).sum()==source.loc[source.date>=TRAIN_START,'units'].eq(0).sum()
    assert frame.loc[frame.units.eq(0),'zero_sales_cause'].eq('unknown').all()

def test_duplicate_daily_grain_rejected():
    frame=daily()
    with pytest.raises(ValueError,match='Duplicate daily'):
        prepare_features(pd.concat([frame,frame.iloc[:1]]))

def test_missing_date_rejected():
    with pytest.raises(ValueError,match='date gaps'):
        prepare_features(daily().drop(index=50))

def test_anonymization_drops_known_pii():
    clean,removed=anonymize(pd.DataFrame({'item_id':['safe'],'email':['a@example.com'],'phone':['+254712345678']}))
    assert clean.columns.tolist()==['item_id']
    assert set(removed)=={'email','phone'}

def test_contact_information_in_unexpected_field_rejected():
    with pytest.raises(ValueError,match='Potential contact'):
        anonymize(pd.DataFrame({'comment':['a@example.com']}))

def test_audit_has_metadata_without_row_values(tmp_path):
    a=Audit(tmp_path/'audit.jsonl','test-run');a.record('read','sales',rows=10)
    event=json.loads((tmp_path/'audit.jsonl').read_text())
    assert event['run_id']=='test-run' and event['rows']==10
    assert 'units' not in event and 'email' not in event

def test_wrong_source_hash_fails_before_processing(tmp_path):
    manifest=tmp_path/'manifest.csv'
    pd.DataFrame({'id':[f'id{i}' for i in range(100)]}).to_csv(manifest,index=False)
    expected=tmp_path/'expected.json';expected.write_text(json.dumps({'source_sha256':{'sales_train_evaluation.csv':'bad'}}))
    (tmp_path/'sales_train_evaluation.csv').write_text('untrusted data')
    with pytest.raises(ValueError,match='hash mismatch'):
        read_inputs(tmp_path,manifest,expected,Audit(tmp_path/'audit.jsonl','test'))
