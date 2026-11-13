"""Streamlit app: Telangana cotton price forecast and price-spike alert."""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import build_features, forecast, next_sunday

BASE = Path(__file__).parent
st.set_page_config(page_title="Telangana Cotton Price Forecast", page_icon="🌾", layout="wide")


@st.cache_resource
def load_models():
    # Only load pickle files you created yourself or trust.
    with open(BASE / "cotton_models.pkl", "rb") as fh:
        return pickle.load(fh)


@st.cache_data
def load_history():
    return pd.read_csv(BASE / "weekly_history.csv", index_col=0, parse_dates=True)


models = load_models()
history = load_history()
tm = models["test_metrics"]
thr = models["threshold"] * 100

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About the model")
    st.write(f"**Price model:** {models['reg_name']}")
    st.write(f"**Movement model:** {models['clf_name']}")
    st.write(f"Trained on {models['n_weeks']} weekly rows, up to {models['trained_until']}.")
    st.subheader("Test-period scores")
    st.caption("Apr to Oct 2025, models trained on earlier weeks only.")
    st.write(f"Price RMSE: **₹{tm['reg']['RMSE']:.0f}** per quintal "
             f"(naive 'same as this week': ₹{tm['naive']['RMSE']:.0f})")
    st.write(f"Movement accuracy: **{tm['clf']['Accuracy']:.0%}** "
             f"(always 'Stable': {tm['majority']['Accuracy']:.0%})")
    st.write(f"Spike alerts: {tm['alerts_correct']} of {tm['alerts_raised']} correct; "
             f"caught {tm['alerts_correct']} of {tm['rise_weeks_in_test']} real rises.")
    st.warning("A decision aid, not financial advice. Accuracy is modest and the model "
               "was trained on only about 90 weeks of data.")

# ------------------------------------------------------------------ header
st.title("🌾 Telangana Cotton Price Forecast & Price-Spike Alert")
st.write(
    "Predicts **next week's modal price** (₹ per quintal) and whether it will **Rise**, "
    f"**Fall** or stay **Stable** (±{thr:.0f}%), using weekly Agmarknet data (CEDA)."
)
st.caption(
    "Prices are a weekly average across Telangana mandis (the model was trained on a "
    "market-adjusted weekly price index). Arrivals are in tonnes."
)

mode = st.radio("Input", ["Use the latest data in the model", "Enter recent weeks myself"], horizontal=True)

hist = None
n_markets = 10
if mode.startswith("Use"):
    hist = history.tail(5)[["p_modal", "p_min", "p_max", "qty"]]
    n_markets = float(history.n_markets.iloc[-1])
    st.subheader("Last 5 weeks used")
    st.dataframe(hist.rename(columns={"p_modal": "Modal price (₹)", "p_min": "Min price (₹)",
                                      "p_max": "Max price (₹)", "qty": "Arrivals (tonnes)"})
                 .round(1))
    week_end = hist.index[-1]
    go = True
else:
    d = history.tail(5)
    last = history.iloc[-1]
    with st.form("inputs"):
        picked = st.date_input("Date inside the latest week", value=history.index[-1].date())
        labels = ["4 weeks ago", "3 weeks ago", "2 weeks ago", "Last week", "Latest week"]
        cols = st.columns(5)
        prices, qtys = [], []
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"**{labels[i]}**")
                prices.append(st.number_input("Modal price (₹/quintal)", 1000.0, 20000.0,
                                              float(d.p_modal.iloc[i]), 10.0, key=f"p{i}"))
                qtys.append(st.number_input("Arrivals (tonnes)", 0.0, 100000.0,
                                            float(d.qty.iloc[i]), 10.0, key=f"q{i}"))
        c1, c2, c3 = st.columns(3)
        pmin = c1.number_input("Latest week: minimum price (₹)", 1000.0, 20000.0, float(last.p_min), 10.0)
        pmax = c2.number_input("Latest week: maximum price (₹)", 1000.0, 20000.0, float(last.p_max), 10.0)
        n_markets = c3.number_input("Markets reporting in latest week", 1, 60, int(last.n_markets))
        go = st.form_submit_button("Forecast next week")
    if go:
        week_end = next_sunday(picked)
        idx = pd.date_range(end=week_end, periods=5, freq="7D")
        hist = pd.DataFrame({"p_modal": prices, "p_min": [float("nan")] * 4 + [pmin],
                             "p_max": [float("nan")] * 4 + [pmax], "qty": qtys}, index=idx)
        if not (pmin <= prices[-1] <= pmax):
            st.warning("The latest modal price is outside your min-max range. Check the numbers.")

# ------------------------------------------------------------------ result
if go and hist is not None:
    X = build_features(hist, n_markets)
    price, move, proba = forecast(models, X)
    now = float(hist.p_modal.iloc[-1])
    change = price / now - 1

    st.subheader(f"Forecast for the week ending {(week_end + pd.Timedelta(days=7)).date()}")
    a, b, c = st.columns(3)
    a.metric("Latest modal price", f"₹{now:,.0f}")
    b.metric("Forecast modal price", f"₹{price:,.0f}", f"{change:+.1%}")
    c.metric("Expected movement", move)

    if move == "Rise":
        st.warning(f"📈 **Price-spike alert:** the model expects a rise of more than {thr:.0f}% next week.")
    elif move == "Fall":
        st.info(f"📉 The model expects a fall of more than {thr:.0f}% next week.")
    else:
        st.success(f"➡️ The model expects the price to stay within ±{thr:.0f}% next week.")
    st.caption("The price forecast and the movement class come from two separate models, "
               "so they can occasionally disagree.")

    left, right = st.columns(2)
    with left:
        chart = pd.DataFrame({"Actual": hist.p_modal})
        nxt = week_end + pd.Timedelta(days=7)
        chart.loc[nxt, "Forecast"] = price
        chart.loc[week_end, "Forecast"] = now
        st.line_chart(chart.sort_index())
    with right:
        if proba:
            st.write("Model confidence for each movement")
            st.bar_chart(pd.Series(proba, name="Probability"))

    with st.expander("How reliable is this?"):
        st.write(
            f"On the Apr to Oct 2025 test weeks the price model had an RMSE of about "
            f"₹{tm['reg']['RMSE']:.0f} per quintal, against ₹{tm['naive']['RMSE']:.0f} for simply "
            "repeating this week's price. In time-series cross-validation on earlier weeks, no model "
            "clearly beat that simple forecast, so treat the numbers as a rough guide. The movement "
            f"model was right about {tm['clf']['Accuracy']:.0%} of the time, and the spike alert caught "
            f"only {tm['alerts_correct']} of {tm['rise_weeks_in_test']} real rises."
        )
