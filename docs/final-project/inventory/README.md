# Inventory policy comparison

This exploratory analysis connects the frozen Module 4 forecasts to the inventory goals stated in Modules 1 and 2. Improved forecast accuracy did not deliver all three original inventory targets under the declared scenarios. No model was retrained, corrected or selected using these results.

## Main findings

The declared base case uses a seven-day lead time, weekly review, assumed holding cost of one cost unit per on-hand unit-day, unmet-unit penalty of 20 cost units and safety multiplier 1.645. The costs are hypothetical scenario units, not actual currency or realized savings. Both policies have the same initial stock and safety buffer.

| Original goal | Forecast policy base case | Result |
| --- | --- | --- |
| Reduce simulated holding plus shortage cost by at least 5% | 0.54% lower than historical-average ordering | Not met |
| At least 95% simulated unit fill rate | 94.68% | Not met |
| No more than five percentage points between store fill rates | 10.77 percentage points | Not met |

The forecast policy costs 126,284 scenario units versus 126,964 for the historical-average policy. Lower holding cost saves 900 units, while the increased unmet-demand penalty adds 220. It fulfils 11,339 of 11,976 scenario-demand units, compared with 11,350 for the baseline. The small cost improvement therefore comes with slightly lower service; it does not establish a superior replenishment policy.

HOUSEHOLD_2 has a simulated fill rate of 74.50% versus 96.63% for HOUSEHOLD_1 in the forecast-policy base case. HOUSEHOLD_2 contains only nine selected series. Among stores, CA_3 has 88.83% and CA_1 has 99.60%. These are conditional operational service diagnostics; they do not measure actual customer availability or protected-group fairness.

## Lead time sensitivity

Other inputs held at the declared base-case values:

| Lead time | Cost reduction | Forecast policy fill rate | Store fill rate gap |
| --- | --- | --- | --- |
| 3 days | 2.44% | 95.88% | 8.20 percentage points |
| 7 days | 0.54% | 94.68% | 10.77 percentage points |
| 14 days | -0.25% | 94.81% | 11.98 percentage points |

![Lead time comparison](lead_time_comparison.svg)

Thirty-six paired scenarios combine lead times of 3, 7 and 14 days, shortage penalties of 5, 20 and 50, safety multipliers of zero and 1.645, and demand-proxy multipliers of 1.0 and 1.2. None of the 36 forecast-policy scenarios meets the cost, fill-rate and store-gap targets simultaneously. The analysis does not select a new preferred policy by searching these outcomes.

## Method and boundaries

The forecast-based policy reviews stock every seven days and orders up to the saved expected sales over lead time plus the next review period, with a history-based safety buffer. The comparison uses the fixed mean of the 28 observed days before the test origin under identical stock, cost and timing assumptions. Neither policy updates its demand forecast from future observed sales. Orders are nonnegative whole units and arrive after the stated lead time. Daily events receive due orders, review/place orders, fulfil demand, then charge holding and shortage costs. Unmet demand is lost, not backordered.

History from 28 March through 24 April 2016 determines the fixed mean and sample standard deviation. Initial stock equals the rounded-up historical protection-period requirement plus the common buffer; initial open orders are zero. The multiplier 1.645 is a declared heuristic applied to daily-sales variability. No empirical 95% forecast-interval coverage is claimed. Storage capacity is unobserved, so no capacity limit is invented. Beyond the 28 saved forecasts, the last predicted week is repeated only to calculate later review targets. Actual future sales never fill this extension.

Observed test sales are the base-case demand proxy. The 1.2 stress multiplier represents assumed additional unit-equivalent demand, not observed lost sales or item transactions. Starting stock, lead times and costs are scenarios. Ending on-hand and outstanding pipeline stock are reported, without an invented salvage value or terminal acquisition charge. The short horizon and substantial assumed initial inventory particularly limit steady-state interpretation for 14-day lead times.

The existing test outcomes and model limitations were already known before this work. Consequently, these are exploratory integration results, not a new independent evaluation. The assumptions file was saved in commit `a86724d8300825617535cc4a4ea8a11961abdd83` before simulation execution. That records the analysis choices; it is not a claim of pre-registration before the original test evaluation. No cost or safety parameter was changed afterward to claim the targets were achieved.

## Reproduce and inspect

From the `final-project` branch, use the verified Module 3 Parquet file locally:

```bash
python -m pytest inventory/test_simulate.py -q
python -m inventory.simulate --prepared data/processed/prepared.parquet
```

The simulator verifies the archived prediction hash, exact product/date alignment and agreement of saved test observations with the local pipeline output. Five tests cover inventory conservation, delayed receipts, outstanding orders, equivalent policies, absence of look-ahead in the opening order, invalid inputs and zero demand. All five passed before batch execution.

- `inventory/scenarios.json` fixes assumptions and the base case.
- `inventory/simulate.py` implements the comparison without model training.
- `scenario_summary.csv` contains pooled results for both policies in all 36 scenarios.
- `base_case_groups.csv` contains store/department aggregates for the base case.
- `summary.json` records source hashes, base-case results and limitations.
- The simulator also generates full aggregate `group_fill_rates.csv` locally. Per-series ledgers and source records are not published here.

The defensible business recommendation is to retain human review, investigate the under-served groups, obtain stock/receipts/cost evidence, and evaluate a revised policy on a new period before an operational pilot. The original inventory targets remain unmet under this analysis; the report should state that clearly.
