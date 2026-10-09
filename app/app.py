"""UK grid demand forecasting dashboard.

Run from the project root:   streamlit run app/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import HORIZON, PREDICTIONS_DIR, TARGET          # noqa: E402
from src.live import (build_live_dataset, fetch_live_raw, forecast_lstm,  # noqa: E402
                      forecast_prophet, forecast_seasonal_naive, load_lstm_ensemble)

MODELS = {                                   # predictions file name -> display name
    "lstm_tuned": "LSTM (ensemble)",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "seasonal_naive": "Seasonal naive",
}
LIVE_MODELS = ["LSTM (ensemble)", "Prophet", "Seasonal naive"]
COLOURS = {"LSTM (ensemble)": "#d62728", "SARIMA": "#1f77b4",
           "Prophet": "#2ca02c", "Seasonal naive": "#7f7f7f"}
BAND_FILL = {"LSTM (ensemble)": "rgba(214,39,40,0.15)", "SARIMA": "rgba(31,119,180,0.15)",
             "Prophet": "rgba(44,160,44,0.15)", "Seasonal naive": "rgba(127,127,127,0.15)"}
BAND_QUANTILES = (0.05, 0.95)                # middle 90% of 2025 errors
LIVE_CACHE_SECONDS = 3 * 3600                # re-download NESO data at most every 3 hours

st.set_page_config(page_title="UK Grid Demand Forecasting", layout="wide")

theme = getattr(getattr(st.context, "theme", None), "type", "light")
ACTUAL_COLOUR = "white" if theme == "dark" else "black"


# ================================================================== shared data
@st.cache_data
def load_predictions():
    frames = []
    for file_name, label in MODELS.items():
        path = PREDICTIONS_DIR / f"{file_name}.csv"
        if not path.exists():
            st.error(f"Missing predictions file: {path.relative_to(PROJECT_ROOT)}")
            st.stop()
        frames.append(pd.read_csv(path, parse_dates=["origin", "date"]).assign(model=label))
    preds = pd.concat(frames, ignore_index=True)
    preds["error"] = preds["y_true"] - preds["y_pred"]
    preds["abs_error"] = preds["error"].abs()
    return preds


@st.cache_data
def error_bands(preds):
    """For each model and horizon: the range covering the middle 90% of its 2025 forecast errors."""
    lo, hi = BAND_QUANTILES
    return (preds.groupby(["model", "horizon"])["error"]
                 .quantile([lo, hi]).unstack()
                 .rename(columns={lo: "err_lo", hi: "err_hi"})
                 .reset_index())


def scores(df):
    err = df["y_true"] - df["y_pred"]
    return {"MAE": err.abs().mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "MAPE": (err.abs() / df["y_true"]).mean() * 100}


def add_band(fig, dates, y_pred, err_lo, err_hi, model):
    fig.add_trace(go.Scatter(x=dates, y=y_pred + err_hi, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=y_pred + err_lo, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=BAND_FILL[model],
                             name=f"{model}: 90% error range", hoverinfo="skip"))


preds = load_predictions()
bands = error_bands(preds)

# ================================================================== header
st.title("UK National Grid: daily demand forecasts")
st.caption("Daily mean National Demand (MW) for Great Britain. Data: NESO historic and live demand data.")

live_tab, backtest_tab, about_tab = st.tabs(["Next 7 days", "2025 backtest", "How it works"])


# ================================================================== live tab
@st.cache_data(ttl=LIVE_CACHE_SECONDS, show_spinner="Downloading the latest NESO data…")
def get_live_data():
    raw = fetch_live_raw()
    return build_live_dataset(raw)


@st.cache_resource
def get_ensemble():
    return load_lstm_ensemble()


@st.cache_data(ttl=LIVE_CACHE_SECONDS, show_spinner="Running the forecasts…")
def get_live_forecasts(data, origin):
    return pd.DataFrame({
        "LSTM (ensemble)": forecast_lstm(data, origin, get_ensemble()),
        "Prophet": forecast_prophet(data, origin),
        "Seasonal naive": forecast_seasonal_naive(data, origin),
    })


with live_tab:
    live_ok = True
    try:
        data, last_actual, gaps = get_live_data()
    except Exception as e:                   # network down, NESO file moved, format changed...
        st.error("Couldn't load the latest NESO data, so the live forecast is unavailable right now. "
                 "The 2025 backtest tab still works.")
        st.caption(f"Details: {type(e).__name__}: {e}")
        live_ok = False

    if live_ok:
        origin = last_actual + pd.Timedelta(days=1)
        forecasts = get_live_forecasts(data, origin)

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest complete day of data", last_actual.strftime("%a %d %b %Y"))
        c2.metric("Forecast period", f"{origin:%d %b} – {forecasts.index[-1]:%d %b %Y}")
        c3.metric("LSTM forecast, next 7 days (average)", f"{forecasts['LSTM (ensemble)'].mean():,.0f} MW")
        if gaps:
            st.warning(f"{gaps} missing day(s) in the live data were filled by interpolation.")

        left, right = st.columns([1, 3])
        shown = left.multiselect("Models", LIVE_MODELS, default=LIVE_MODELS[:2], key="live_models")
        band_for = left.selectbox("Error band for", ["None"] + shown, index=1 if shown else 0, key="live_band",
                                  help="Middle 90% of this model's 2025 errors at each horizon.")
        history_days = left.slider("Days of history to show", 7, 90, 28, key="live_history")

        fig = go.Figure()
        if band_for != "None":
            b = bands[bands["model"] == band_for].sort_values("horizon")
            add_band(fig, forecasts.index, forecasts[band_for].values, b["err_lo"].values, b["err_hi"].values, band_for)
        recent = data.loc[last_actual - pd.Timedelta(days=history_days - 1):last_actual, TARGET]
        fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines",
                                 line=dict(color=ACTUAL_COLOUR, width=2.5), name="Actual"))
        for model in shown:
            fig.add_trace(go.Scatter(x=forecasts.index, y=forecasts[model], mode="lines+markers",
                                     line=dict(color=COLOURS[model], width=2), name=model))
        fig.add_vline(x=origin - pd.Timedelta(hours=12), line_dash="dot", line_color="grey")
        fig.update_layout(title=f"Forecast for {origin:%d %b} – {forecasts.index[-1]:%d %b %Y}",
                          yaxis_title="Daily mean demand (MW)", hovermode="x unified", height=460,
                          margin=dict(t=60, b=20), legend=dict(orientation="h", y=-0.15))
        right.plotly_chart(fig, width="stretch")

        table = forecasts.round(0).astype(int)
        table.insert(0, "Day", table.index.strftime("%a"))
        hol = data.loc[forecasts.index, "holiday_name"].replace("", np.nan)
        if hol.notna().any():
            table["Bank holiday"] = hol.fillna("")
        table.index = table.index.strftime("%d %b %Y")
        st.dataframe(table, width="stretch")
        st.caption("SARIMA isn't run live: it needs a full refit on all data up to today, which is too slow "
                   "for a web page. Its 2025 results are in the backtest tab. NESO publishes actual demand "
                   "with a short delay, so the forecast starts from the day after the latest complete day.")


# ================================================================== backtest tab
with backtest_tab:
    st.caption("Every model made 359 rolling 7-day forecasts across 2025, each using only data up to the day before.")
    model_names = list(MODELS.values())
    dates = preds["date"].dt.date

    c1, c2, c3, c4 = st.columns([1.2, 2, 1.3, 1.6])
    horizon = c1.slider("Days ahead", 1, HORIZON, 1, key="bt_horizon",
                        help="Each point is a forecast made this many days before the date it predicts.")
    selected = c2.multiselect("Models", model_names, default=model_names[:3], key="bt_models")
    band_model = c3.selectbox("Error band for", ["None"] + selected, index=1 if selected else 0, key="bt_band")
    date_range = c4.date_input("Date range", value=(pd.Timestamp("2025-10-01").date(), dates.max()),
                               min_value=dates.min(), max_value=dates.max(), key="bt_dates")

    if not selected:
        st.info("Pick at least one model.")
    elif len(date_range) != 2:               # the picker returns one date while the user is mid-selection
        st.info("Pick an end date for the range.")
    else:
        start, end = date_range
        in_range = preds["date"].dt.date.between(start, end)
        view = preds[(preds["horizon"] == horizon) & in_range & preds["model"].isin(selected)]

        naive_mae = scores(preds[(preds["horizon"] == horizon) & in_range
                                 & (preds["model"] == "Seasonal naive")])["MAE"]
        for col, model in zip(st.columns(len(selected)), selected):
            s = scores(view[view["model"] == model])
            col.metric(f"{model}: MAE", f"{s['MAE']:,.0f} MW",
                       delta=f"{(s['MAE'] / naive_mae - 1) * 100:+.1f}% vs naive", delta_color="inverse",
                       help=f"RMSE {s['RMSE']:,.0f} MW · MAPE {s['MAPE']:.2f}%")

        fig = go.Figure()
        if band_model != "None":
            b = view[view["model"] == band_model].merge(bands, on=["model", "horizon"]).sort_values("date")
            add_band(fig, b["date"], b["y_pred"], b["err_lo"], b["err_hi"], band_model)
        actual = view.drop_duplicates("date").sort_values("date")
        fig.add_trace(go.Scatter(x=actual["date"], y=actual["y_true"], mode="lines",
                                 line=dict(color=ACTUAL_COLOUR, width=2.5), name="Actual"))
        for model in selected:
            m = view[view["model"] == model].sort_values("date")
            fig.add_trace(go.Scatter(x=m["date"], y=m["y_pred"], mode="lines",
                                     line=dict(color=COLOURS[model], width=1.5), name=model))
        fig.update_layout(title=f"Actual demand vs {horizon}-day-ahead forecasts",
                          yaxis_title="Daily mean demand (MW)", hovermode="x unified", height=480,
                          margin=dict(t=60, b=20), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

        if band_model != "None":
            inside = ((b["y_true"] >= b["y_pred"] + b["err_lo"]) & (b["y_true"] <= b["y_pred"] + b["err_hi"])).mean()
            st.caption(f"Shaded band: the middle 90% of {band_model}'s 2025 errors at this horizon. "
                       f"In the selected dates, {inside:.0%} of actual values fall inside it. The band comes from "
                       "the same year it's shown on, so it describes typical error rather than a tested prediction interval.")

        left, right = st.columns([3, 2])
        h_table = (preds[preds["model"].isin(selected) & in_range]
                   .groupby(["model", "horizon"])["abs_error"].mean().unstack())
        fig_h = go.Figure()
        for model in selected:
            fig_h.add_trace(go.Scatter(x=h_table.columns, y=h_table.loc[model], mode="lines+markers",
                                       line=dict(color=COLOURS[model]), name=model))
        fig_h.add_vline(x=horizon, line_dash="dot", line_color="grey")
        fig_h.update_layout(title="Error by horizon (selected dates)", xaxis_title="Days ahead",
                            yaxis_title="MAE (MW)", height=380, margin=dict(t=60, b=20),
                            legend=dict(orientation="h", y=-0.25))
        left.plotly_chart(fig_h, width="stretch")

        overall = pd.DataFrame({m: scores(preds[(preds["model"] == m) & in_range])
                                for m in model_names}).T.sort_values("MAE")
        right.subheader("All horizons, selected dates")
        right.dataframe(overall.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE": "{:.2f}%"}),
                        width="stretch")


# ================================================================== about tab
with about_tab:
    st.markdown("""
**Data.** NESO half-hourly National Demand from 2021, aggregated to daily mean MW. The live tab downloads
the current year's file and NESO's demand update (cached for 3 hours) and processes them with the same
code as the training data.

**Evaluation (2025 backtest).** Models were trained on 2021–2024 (the LSTM on 2021–2023, with 2024 used
only to decide when to stop training) and tested on 2025. Every day from 1 Jan to 25 Dec 2025, each model
forecast the next 7 days using only data up to the day before: 359 forecasts per model.

**Models.**
- **Seasonal naive:** "same day last week". The benchmark.
- **SARIMA:** ARIMA(1,0,0)(2,1,1)[7] with bank-holiday and Christmas inputs, updated with new data each day.
- **Prophet:** weekly and yearly seasonality plus bank holidays, refitted each day.
- **LSTM:** 14 days of history plus the upcoming calendar (day of week, day of year, holidays);
  the average of 5 models trained with different random seeds.

**Main finding.** SARIMA is most accurate one day ahead. The LSTM is most accurate for days 2–7 and
overall (MAE 1,324 MW, 11% lower than SARIMA and Prophet).

**Error bands.** The middle 90% of each model's 2025 forecast errors at each horizon. They show typical
error, not a formally tested prediction interval.
""")
