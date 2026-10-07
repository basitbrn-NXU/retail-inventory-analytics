import pandas as pd
import pytest
from src.validate import run_validation

def test_gx_rejects_bad_row_count_and_negative_units(tmp_path):
    frame=pd.DataFrame({'id':['a'],'item_id':['i'],'store_id':['CA_1'],
        'date':pd.to_datetime(['2016-01-01']),'units':[-1],'split':['train'],
        'cat_id':['HOUSEHOLD'],'zero_sales_cause':['not_zero']})
    with pytest.raises(ValueError,match='quality gate failed'):
        run_validation(frame,tmp_path)
    import json
    result=json.loads((tmp_path/'gx_validation.json').read_text())
    assert result['success'] is False
    assert result['statistics']['unsuccessful_expectations']>=2
