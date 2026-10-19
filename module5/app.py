"""Stakeholder exploration of frozen Module 4 evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import (DATA, LABELS, api_predict, group_review, load_model,
                  load_test, metrics, read_json, scenario_row)

st.set_page_config(page_title="Retail forecast insights", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;padding-bottom:2rem;max-width:1400px}
h1{letter-spacing:-.035em} [data-testid="stMetricValue"]{font-size:1.9rem}
</style>""", unsafe_allow_html=True)

@st.cache_data
def data():
    return load_test(), read_json("examples.json"), read_json("test_summary.json")

@st.cache_resource
def frozen_model():
    return load_model()

def plot(fig, key):
    fig.update_layout(template="plotly_white", font=dict(size=14, color="#172B3A"),
                      margin=dict(l=15,r=20,t=45,b=20), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, key=key)

def source_note(text):
    st.caption("Source: " + text)

try:
    test, examples, summary = data()
    model, manifest = frozen_model()
except Exception as exc:
    st.error("The verified project files could not be loaded. Check the complete dashboard package and pinned dependencies.")
    st.code(str(exc))
    st.stop()

st.sidebar.title("Retail forecast insights")
page = st.sidebar.radio("Explore", [
    "Forecast results", "Prediction examples", "What-if explorer",
    "Fairness and ethics", "Sources and scope"], key="page")
st.sidebar.caption("100 product–store combinations · 10 stores · household products")
st.sidebar.caption("Historical study, 2016 · academic candidate")
st.sidebar.markdown("**Intended audience:** inventory planners, store managers and governance reviewers.")
st.sidebar.caption("No model training or stock orders are performed here.")

if page == "Forecast results":
    st.title("How closely did forecasts follow sales?")
    st.write("Explore the held-out 28 days, April 25–May 22, 2016. The comparison repeats the latest weekday sales pattern as a simple baseline.")
    left, right = st.columns(2)
    stores = left.multiselect("Stores", sorted(test.store_id.unique()),
                              default=sorted(test.store_id.unique()), key="stores")
    depts = right.multiselect("Departments", sorted(test.dept_id.unique()),
                              default=sorted(test.dept_id.unique()), key="depts")
    selected = test[test.store_id.isin(stores) & test.dept_id.isin(depts)]
    if selected.empty:
        st.info("No records match these filters. Select at least one store and department with shared records.")
        st.stop()
    m = metrics(selected.units, selected.prediction)
    b = metrics(selected.units, selected.prediction_baseline)
    improvement = 100*(1-m["RMSE"]/b["RMSE"]) if b["RMSE"] else np.nan
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Typical absolute error", f'{m["MAE"]:.2f} units')
    c2.metric("RMSE improvement over baseline", f"{improvement:.2f}%")
    c3.metric("Absolute error / sales", f'{100*m["WAPE"]:.2f}%')
    c4.metric("Net forecast bias", f'{100*m["Bias"]:+.2f}%')
    st.caption(f"Current selection: {selected.id.nunique()} product–store combinations, {len(selected):,} daily records. Negative bias means total forecasts were below total sales.")
    daily = selected.groupby("date")[["units","prediction","prediction_baseline"]].sum().reset_index()
    daily = daily.rename(columns={"units":"Actual sales","prediction":"Model forecast",
                                 "prediction_baseline":"Weekly baseline"})
    fig = px.line(daily,x="date",y=["Actual sales","Model forecast","Weekly baseline"],
                  color_discrete_map={"Actual sales":"#172B3A","Model forecast":"#007F82","Weekly baseline":"#8C6D31"},
                  labels={"date":"Date","value":"Daily sales units"},
                  title="Total sales and forecasts for the selected products and stores")
    fig.update_yaxes(rangemode="tozero")
    plot(fig,"daily")
    st.write("Totals can hide errors for individual products or stores. Accuracy gains do not establish lower storage costs, fewer stockouts or achieved service levels.")
    with st.expander("How to read these measures"):
        st.write("Typical absolute error is the average size of a daily product–store error. RMSE gives more weight to large misses. Absolute error / sales (WAPE) divides all absolute errors by total observed sales; it is not a percentage of correct predictions. Net bias allows over- and underprediction to cancel.")
    st.download_button("Download the selected evaluation records",
                       selected.to_csv(index=False).encode(), "selected_evaluation.csv", "text/csv")
    source_note("Module 4 frozen final-test predictions; metric definitions: Hyndman & Athanasopoulos (2021).")

elif page == "Prediction examples":
    st.title("What does an individual forecast mean?")
    st.write("Three examples from the first validation day, March 28, 2016. These illustrate forecast behaviour, not the final-test accuracy estimate.")
    index = st.selectbox("Example", range(len(examples)),
                         format_func=lambda i: examples[i]["id"].replace("_evaluation",""), key="example")
    e = examples[index]
    c1,c2,c3=st.columns(3)
    c1.metric("Forecast daily sales",f'{e["prediction"]:.2f} units')
    c2.metric("Recorded sales",f'{e["actual_units"]} units')
    c3.metric("Forecast minus sales",f'{e["prediction"]-e["actual_units"]:+.2f} units')
    st.write(f'The model estimated {e["prediction"]:.2f} units for this product at {e["inputs"]["store_id"]}. Recorded sales were {e["actual_units"]} units. A fractional forecast expresses an expected quantity; it is not a stock-order instruction.')
    contributions = pd.DataFrame({"Input":[LABELS[k] for k in e["shap"]],
                                   "Contribution":list(e["shap"].values())})
    contributions = contributions.assign(magnitude=lambda x:x.Contribution.abs()).nlargest(7,"magnitude").sort_values("Contribution")
    fig=px.bar(contributions,x="Contribution",y="Input",orientation="h",
               color=contributions.Contribution.ge(0).map({True:"Raises forecast",False:"Lowers forecast"}),
               color_discrete_map={"Raises forecast":"#007F82","Lowers forecast":"#AB4B4B"},
               title="Largest contributions to this forecast",
               labels={"Contribution":"Contribution in daily sales units"})
    fig.add_vline(x=0,line_color="#172B3A")
    plot(fig,"local_shap")
    st.write(f'The explanation starts from a reference value of {e["base_value"]:.2f} units. All 13 input contributions together produce the forecast. The chart shows the seven largest; it describes the model, not causes of customer purchases.')
    st.subheader("What influenced forecasts across validation?")
    importance = pd.read_csv(DATA/"importance.csv").head(7).iloc[::-1]
    importance["Input"]=importance.feature.map(LABELS)
    plot(px.bar(importance,x="mean_absolute_shap",y="Input",orientation="h",
                labels={"mean_absolute_shap":"Average absolute contribution (units)"},
                color_discrete_sequence=["#007F82"]), "global_shap")
    st.caption("Global chart: all 2,800 validation forecasts. Later-horizon history inputs can contain earlier model predictions. Correlated inputs affect attribution.")
    with st.expander("View the example's original inputs"):
        st.json(e["inputs"])
    source_note("Module 4 validation predictions and saved TreeSHAP contributions; SHAP TreeExplainer documentation.")

elif page == "What-if explorer":
    st.title("How does the model respond to different inputs?")
    st.write("Change one validation snapshot while keeping the trained model fixed. History changes are hypothetical: past sales cannot be changed.")
    index=st.selectbox("Example",range(len(examples)),
                       format_func=lambda i:examples[i]["id"].replace("_evaluation",""),key="scenario_example")
    e=examples[index]
    c1,c2=st.columns(2)
    history=c1.slider("Change the three sales-history inputs together (%)",-20,20,0,5,key="history")
    price=c2.slider("Change the last historical price (%)",-10,10,0,5,key="price")
    altered=scenario_row(e,history,price)
    try:
        response=api_predict(model,manifest,altered)
    except Exception as exc:
        st.error("Prediction could not be obtained from the model API.")
        st.code(str(exc));st.stop()
    prediction=response["predictions"][0]
    a,b,c=st.columns(3)
    a.metric("Original forecast",f'{e["prediction"]:.2f} units')
    b.metric("Scenario forecast",f"{prediction:.2f} units")
    c.metric("Forecast change",f'{prediction-e["prediction"]:+.2f} units')
    st.warning("This is a model-response illustration, not a causal estimate or a replenishment recommendation. The scenario does not regenerate a 28-day forecast. Training-range coverage is not verified for this individual or combined scenario.")
    with st.expander("Compare original and scenario inputs"):
        cols=["lag_7","lag_28","rolling_mean_28","historical_sell_price"]
        st.dataframe(pd.DataFrame({"Input":[LABELS[k] for k in cols],
                      "Original":[e["inputs"][k] for k in cols],
                      "Scenario":[altered[k] for k in cols]}),hide_index=True)
    st.subheader("Context from the original sensitivity study")
    st.write("Across all validation snapshots, changing history by −20% and +20% changed total model forecasts by −17.41% and +17.67%. Price scenarios had small responses, but 85–95% lay outside observed series-specific training ranges. This cannot establish price elasticity.")
    st.caption("Serving status: same Module 4 /predict contract and verified model export, executed within this app. The HTTP demo can be run separately; no connection to your laptop is needed for a cloud dashboard.")
    source_note("Module 4 sensitivity evidence; frozen model and Module 4 API factory. No new final-test predictions or model fitting.")

elif page == "Fairness and ethics":
    st.title("Do errors differ across stores and departments?")
    st.write("This is an operational performance comparison. The dataset has no customer demographic attributes, so demographic fairness cannot be assessed.")
    dimension=st.radio("Compare",["Stores","Departments"],horizontal=True,key="dimension")
    table=group_review(test,"store_id" if dimension=="Stores" else "dept_id")
    chart=table.copy()
    chart["Net forecast bias (%)"]=100*chart.Bias
    chart["Coverage"]=np.where(chart["Limited coverage"],"Limited coverage","Other groups")
    fig=px.bar(chart,x="Group",y="Net forecast bias (%)",color="Coverage",
               color_discrete_map={"Limited coverage":"#8C6D31","Other groups":"#007F82"},
               title="Net overprediction and underprediction by group",
               hover_data=["Series","Rows"])
    fig.add_hline(y=10,line_dash="dash",line_color="#666666")
    fig.add_hline(y=-10,line_dash="dash",line_color="#666666")
    plot(fig,"bias")
    st.caption("Dashed lines mark a ±10% diagnostic review threshold, not proof of fairness or unfairness. Negative values mean underprediction. All groups use the same held-out period.")
    display=table.copy()
    display["WAPE"]=display.WAPE.map(lambda x:f"{100*x:.2f}%")
    display["Bias"]=display.Bias.map(lambda x:f"{100*x:+.2f}%")
    display[["MAE","RMSE"]]=display[["MAE","RMSE"]].round(3)
    st.dataframe(display,hide_index=True,use_container_width=True)
    st.write("HOUSEHOLD_2 was underpredicted by 32.23% across nine series. CA_4 has only four series. Uneven coverage and product mix limit interpretation. A date row is not an independent customer.")
    st.subheader("Ethical compliance status")
    st.dataframe(pd.DataFrame([
        ["Explainability","Available","Global and local SHAP explanations are archived."],
        ["Protected-group fairness","Not measurable","No customer demographic attributes; store IDs are not substitutes."],
        ["Bias mitigation","Review proposed","Group gaps identified; no post-test adjustment or demonstrated mitigation."],
        ["Privacy","Limited inputs","No customer PII in M5; API audit excludes raw feature values."],
        ["Inventory and cost outcomes","Not evaluated","Stock levels, lead times, cost savings and fill rate are unverified."],
        ["Production approval","Not granted","Academic candidate; predictions are not stock orders."]
    ],columns=["Control","Status","Evidence or boundary"]),hide_index=True,use_container_width=True)
    st.write("The Module 1 ethical principles and Module 2 privacy approach carry forward through traceable outputs, restricted input fields and transparent limitations. The professor approved using M5 subject to demonstrating explainability; this does not establish demographic fairness.")
    source_note("Module 4 held-out predictions and group review; Fairlearn MetricFrame computes the displayed group losses; Module 1 Ethical AI Charter and Module 2 Privacy Plan.")

else:
    st.title("Scope, provenance and sources")
    st.write("Business context: balancing product availability against excess inventory and storage costs. This dashboard communicates model evidence neutrally to inventory planners, store managers and ethics/governance reviewers.")
    st.write("The model learns recurring patterns from recent sales, weekday, product/store identity and calendar inputs. A fixed model was selected through chronological development; the 28-day final test was kept separate.")
    st.subheader("What the evidence covers")
    st.write("100 household product–store combinations; ten stores; 78,600 prepared records. Final test: April 25–May 22, 2016. Explanation examples: March 28, 2016. Improved accuracy is measured against a weekly baseline.")
    st.subheader("What the evidence cannot establish")
    st.write("Sales are not unconstrained demand: zero sales may reflect missing stock or no purchases. There is no inventory, lead-time, weather or customer-demographic information. Cost savings, service-level improvements, price elasticity and demographic fairness are not demonstrated.")
    st.write("The model and held-out results remain unchanged. The unsuccessful DiCE search is retained as a null finding; its target was not relaxed. What-if controls explain one forecast snapshot and do not change the historical evaluation.")
    with st.expander("Technical provenance"):
        st.json({k:manifest[k] for k in ["registered_model_name","version","development_run_id","model_sha256","data_sha256","deployment_approved"]})
        st.write("The dashboard verifies the frozen model hash and original API sample prediction on load. The Module 4 request schemas and API factory are reused verbatim; the cloud-compatible loader uses the verified exported artifact instead of a PC-specific MLflow database.")
    st.subheader("References")
    for text in [
        "Sheikh, A. B. (2026). BAN6800 Module 1 Vision, Module 2 Project Overview, Module 3 Pipeline and Module 4 Model Validation Report. Nexford University.",
        "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013",
        "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and practice (3rd ed.). https://otexts.com/fpp3/",
        "SHAP. (n.d.). TreeExplainer. https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html",
        "Fairlearn. (n.d.). MetricFrame. https://fairlearn.org/v0.13/api_reference/generated/fairlearn.metrics.MetricFrame.html"
    ]: st.markdown(text)
    st.markdown("[Project repository](https://github.com/basitbrn-NXU/retail-inventory-analytics)")
