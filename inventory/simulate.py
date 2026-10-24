"""Fixed-forecast inventory scenarios; no training or policy tuning on test outcomes."""
from __future__ import annotations
import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def simulate(demand, forecast, history_mean, history_std, *, lead, review=7,
             safety=1.645, holding=1., penalty=20., policy="forecast", initial_stock=None):
    demand = np.asarray(demand, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(demand) == 0 or len(demand) != len(forecast):
        raise ValueError("Demand and forecast must align over a nonempty horizon.")
    if (not np.isfinite(demand).all() or not np.isfinite(forecast).all()
            or np.any(demand < 0) or np.any(forecast < 0)):
        raise ValueError("Demand and forecast must be finite and nonnegative.")
    if not isinstance(lead, int) or lead < 1 or not isinstance(review, int) or review < 1:
        raise ValueError("Lead and review must be positive integer days.")
    parameters = [history_mean, history_std, safety, holding, penalty]
    if not np.isfinite(parameters).all() or min(parameters) < 0:
        raise ValueError("History and scenario parameters must be finite and nonnegative.")
    if policy not in {"forecast", "historical_average"}:
        raise ValueError("Unknown policy.")
    protection = lead + review
    buffer = safety * history_std * math.sqrt(protection)
    starting_stock = math.ceil(history_mean * protection + buffer) if initial_stock is None else float(initial_stock)
    if not np.isfinite(starting_stock) or starting_stock < 0:
        raise ValueError("Invalid starting stock.")
    on_hand = starting_stock
    orders = {}
    ledger = []

    def at(t):
        # Repeat the last forecast week, never any held-out actual sales.
        return forecast[t] if t < len(forecast) else forecast[max(0, len(forecast)-7) + ((t-len(forecast)) % min(7, len(forecast)))]

    for day, required in enumerate(demand):
        opening = on_hand
        receipts = orders.pop(day, 0.)
        on_hand += receipts
        placed = 0
        if day % review == 0:
            expected = sum(at(t) for t in range(day, day+protection)) if policy == "forecast" else history_mean * protection
            target = math.ceil(expected + buffer)
            position = on_hand + sum(orders.values())
            placed = max(0, math.ceil(target - position))
            if placed:
                orders[day+lead] = orders.get(day+lead, 0.) + placed
        fulfilled = min(on_hand, required)
        unmet = required - fulfilled
        on_hand -= fulfilled
        if not math.isclose(opening + receipts - fulfilled, on_hand, abs_tol=1e-9):
            raise AssertionError("Inventory accounting failed.")
        ledger.append({"day": day, "opening_stock": opening, "receipts": receipts, "placed": placed,
                       "demand": required, "fulfilled": fulfilled, "unmet": unmet, "closing_stock": on_hand,
                       "holding_cost": holding*on_hand, "shortage_cost": penalty*unmet})
    daily = pd.DataFrame(ledger)
    totals = {name: float(daily[name].sum()) for name in ["demand", "fulfilled", "unmet", "holding_cost", "shortage_cost", "placed", "receipts"]}
    totals.update(initial_stock=float(starting_stock), ending_stock=float(on_hand), pipeline_stock=float(sum(orders.values())),
                  mean_inventory=float(daily.closing_stock.mean()), cost=float(daily.holding_cost.sum()+daily.shortage_cost.sum()),
                  fill_rate=float(daily.fulfilled.sum()/daily.demand.sum()) if daily.demand.sum() else None)
    return totals, daily


def run(prepared_path, predictions_path, config_path, output):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    prepared = pd.read_parquet(prepared_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["date"], float_precision="round_trip")
    expected_digest = "a7c74ac569cd255b87183a3c0e032818c3ecc55a84dd70296d26580b8ff754fa"
    if hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest() != expected_digest:
        raise ValueError("Use the unchanged archived Module 4 final-test predictions.")
    if predictions.duplicated(["id", "date"]).any():
        raise ValueError("Duplicate forecast keys.")
    expected_dates = pd.date_range("2016-04-25", "2016-05-22")
    if len(predictions) != 2800 or predictions.id.nunique() != 100:
        raise ValueError("Expected the fixed 100-series, 2,800-row final-test cohort.")
    recent = prepared.loc[prepared.date.between("2016-03-28", "2016-04-24")]
    truth = prepared.loc[prepared.split.eq("test"), ["id", "date", "units"]]
    check = predictions.merge(truth, on=["id", "date"], suffixes=("", "_source"), validate="one_to_one")
    if len(check) != 2800 or not np.array_equal(check.units, check.units_source):
        raise ValueError("Saved final-test observations differ from the verified pipeline.")
    series = []
    for sid, frame in predictions.groupby("id", sort=True):
        frame = frame.sort_values("date")
        history = recent.loc[recent.id.eq(sid)].sort_values("date")
        expected_history = pd.Series(pd.date_range("2016-03-28", "2016-04-24"))
        if (not frame.date.reset_index(drop=True).equals(pd.Series(expected_dates))
                or not history.date.reset_index(drop=True).equals(expected_history)):
            raise ValueError("Series requires complete aligned test and prior 28-day history.")
        series.append((sid, frame, float(history.units.mean()), float(history.units.std(ddof=1))))
    records = []
    scenario_dimensions = product(cfg["lead_times_days"], cfg["shortage_penalties_per_unmet_unit"],
                                  cfg["safety_multipliers"], cfg["demand_multipliers"])
    for lead, penalty, safety, demand_multiplier in scenario_dimensions:
        for sid, frame, history_mean, history_std in series:
            for policy in ["forecast", "historical_average"]:
                totals, _ = simulate(frame.units.to_numpy()*demand_multiplier, frame.prediction,
                    history_mean, history_std, lead=lead, review=cfg["review_period_days"], safety=safety,
                    holding=cfg["holding_cost_per_unit_day"], penalty=penalty, policy=policy)
                records.append({"lead": lead, "penalty": penalty, "safety": safety, "demand_multiplier": demand_multiplier,
                                "policy": policy, "id": sid, "store_id": frame.store_id.iloc[0],
                                "dept_id": frame.dept_id.iloc[0], **totals})
    detail = pd.DataFrame(records)
    keys = ["lead", "penalty", "safety", "demand_multiplier", "policy"]
    summary = detail.groupby(keys, as_index=False).agg(cost=("cost", "sum"), demand=("demand", "sum"),
        fulfilled=("fulfilled", "sum"), unmet=("unmet", "sum"), mean_inventory=("mean_inventory", "sum"),
        initial_stock=("initial_stock", "sum"), ending_stock=("ending_stock", "sum"), pipeline_stock=("pipeline_stock", "sum"))
    summary["fill_rate"] = summary.fulfilled / summary.demand
    comparison_keys = keys[:-1]
    baseline = summary.loc[summary.policy.eq("historical_average"), comparison_keys+["cost"]].rename(columns={"cost":"baseline_cost"})
    summary = summary.merge(baseline, on=comparison_keys, validate="many_to_one")
    summary["cost_reduction"] = 1-summary.cost/summary.baseline_cost.replace(0,np.nan)
    summary["cost_target_met"] = summary.cost_reduction >= cfg["targets"]["cost_reduction_fraction"]
    summary["fill_target_met"] = summary.fill_rate >= cfg["targets"]["unit_fill_rate"]
    groups = []
    for dim in ["store_id", "dept_id"]:
        group = detail.groupby(keys+[dim], as_index=False).agg(demand=("demand", "sum"), fulfilled=("fulfilled", "sum"),
            cost=("cost", "sum"), series=("id", "nunique"))
        group["fill_rate"] = group.fulfilled / group.demand
        group["dimension"] = dim
        groups.append(group.rename(columns={dim:"group"}))
    groups = pd.concat(groups, ignore_index=True)
    store_gap = groups.loc[groups.dimension.eq("store_id")].groupby(keys).fill_rate.agg(lambda a: a.max()-a.min()).rename("store_fill_gap").reset_index()
    summary = summary.merge(store_gap, on=keys, validate="one_to_one")
    summary["store_gap_target_met"] = summary.store_fill_gap <= cfg["targets"]["maximum_store_fill_rate_gap"]
    base = cfg["base_case"]
    base_rows = summary.loc[summary.lead.eq(base["lead_time_days"]) & summary.penalty.eq(base["shortage_penalty_per_unmet_unit"]) & summary.safety.eq(base["safety_multiplier"]) & summary.demand_multiplier.eq(base["demand_multiplier"])]
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output/"scenario_summary.csv", index=False)
    groups.to_csv(output/"group_fill_rates.csv", index=False)
    # Per-series and daily ledgers remain local; publish only aggregate scenario outputs.
    evidence = {"analysis_type": cfg["analysis_type"], "base_case": base_rows.to_dict(orient="records"),
                "source_hashes": {str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [prepared_path,predictions_path,config_path]},
                "scenario_count": len(summary)//2, "series_count": 100, "days":28,
                "limitations":["Observed sales are a demand proxy, not measured stockouts or unconstrained demand.",
                  "All stock, costs, lead times, safety buffers and capacity assumptions are scenarios.",
                  "The same starting stock and safety buffer are used within each policy pair.",
                  "The short 28-day horizon and assumed starting stock limit steady-state interpretation, especially at 14-day lead time.",
                  "No statistical interval coverage or realized business savings is claimed.",
                  "Existing test outcomes were already known; this is exploratory integration analysis and does not provide a new untouched evaluation."]}
    (output/"summary.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    print(base_rows.to_string(index=False))
    print(f"Completed {len(summary)//2} paired scenarios; aggregate results saved to {output}.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--predictions",default="module5/data/test_predictions.csv")
    parser.add_argument("--config",default="inventory/scenarios.json")
    parser.add_argument("--output",default="docs/final-project/inventory")
    a=parser.parse_args(); run(a.prepared,a.predictions,a.config,a.output)
