import numpy as np
import pytest
from inventory.simulate import simulate


def test_weekly_review_ledger_balances_receipts_and_orders():
    totals, daily = simulate([3,5,4,0], [4,4,4,4], 4, 0, lead=1, review=2, safety=0)
    assert daily.closing_stock.tolist() == [9,4,0,8]
    assert daily.placed.tolist() == [0,0,8,0]
    assert totals["cost"] == 21
    assert totals["fill_rate"] == 1
    assert totals["initial_stock"]+totals["receipts"]-totals["fulfilled"] == totals["ending_stock"]


def test_order_cannot_fulfil_sales_before_arrival():
    totals, daily = simulate([5,5,5], [5,5,5], 5, 0, lead=1, review=1, safety=0, initial_stock=0)
    assert daily.receipts.tolist() == [0,10,0]
    assert totals["fulfilled"] == 10
    assert totals["unmet"] == 5
    assert totals["pipeline_stock"] == 5
    assert totals["fill_rate"] == pytest.approx(2/3)


def test_equal_forecasts_give_identical_policy_results():
    a,_=simulate([1,0,9,2]*7,[3]*28,3,1,lead=7,policy="forecast")
    b,_=simulate([1,0,9,2]*7,[3]*28,3,1,lead=7,policy="historical_average")
    assert a==b


def test_no_lookahead_and_same_initial_inventory():
    a,x=simulate([1]*28,[2]*28,2,1,lead=3)
    b,y=simulate([100]*28,[2]*28,2,1,lead=3)
    assert a["initial_stock"]==b["initial_stock"]
    assert x.placed.iloc[0]==y.placed.iloc[0]
    assert np.allclose(y.opening_stock+y.receipts-y.fulfilled,y.closing_stock)


def test_rejects_invalid_inputs_and_handles_zero_demand():
    for demand,forecast in [([-1],[1]),([1],[float('nan')]),([1,2],[1])]:
        with pytest.raises(ValueError): simulate(demand,forecast,1,0,lead=1)
    totals,_=simulate([0]*28,[0]*28,0,0,lead=14)
    assert totals["fill_rate"] is None
    assert totals["cost"] == 0
