# Module 3 prepared data dictionary

Grain: one product-store-date. File: local data/processed/prepared.parquet.
Rows: 78,600. Type: Parquet with native timestamp and numeric fields.
The Module 2 Excel workbook remains the input dictionary and RAID reference.

| Fields | Meaning and type | Null policy / modelling role |
|---|---|---|
| id, item_id, dept_id, cat_id, store_id, state_id | String product/store hierarchy | Required identifiers, no customer identities |
| date | Timestamp for the observation day | Required, unique together with id |
| d | String M5 day identifier | Required calendar join lineage |
| wm_yr_wk | Integer M5 week identifier | Required weekly price join lineage |
| units | Integer observed daily sales | Required nonnegative evaluation label; zero cause unknown |
| wday, month, year | Integer calendar features | Required; verify availability at forecast time in Module 4 |
| event_name_1, event_type_1, event_name_2, event_type_2 | Optional calendar event strings | Null is legitimate absence of a recorded event |
| snap | Integer 0/1 state calendar flag | State-level scheduled feature, no individual customer data |
| lag_7, lag_28 | Sales from 7 or 28 days earlier | Training complete; held-out inputs unavailable after frozen cutoff become null |
| rolling_mean_28 | Mean of preceding 28 daily observations | Shifted by one day; full 28-day window required |
| historical_sell_price | Weekly selling price on training observations | Validation/test null deliberately excludes realised future prices |
| price_missing | Boolean unmatched original weekly price indicator | Required; actual run has zero unmatched prices |
| zero_sales_cause | unknown for zero units, otherwise not_zero | Required; does not infer stock availability |
| split | train, validation or test | Train 2014-03-29 to 2016-03-27, validation to 2016-04-24, test to 2016-05-22 |

Never feed held-out units to a forecasting model as a predictor. All post-training
actual sales are masked before lag construction. Subsequent multi-day forecasting
must generate horizon-specific inputs recursively or with direct models. The
prepared label table is not evidence of realised stockouts or inventory savings.
