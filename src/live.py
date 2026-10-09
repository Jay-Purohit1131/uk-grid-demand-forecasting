"""Live forecasting: fetch the latest NESO data and forecast the next 7 days.

Used by the dashboard. Every step reuses the same data-prep, calendar and model code as the
notebooks, so live forecasts are produced exactly the way the backtest forecasts were.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import DEMAND_UPDATE_URL, HORIZON, LIVE_YEAR_URL, MODELS_DIR, PROCESSED_DATA, TARGET
from src.data_prep import MIN_PERIODS_PER_DAY, add_calendar_features, clean_half_hourly, parse_settlement_dates, to_daily
from src.lstm_data import add_calendar_encodings
from src.lstm_model import DemandLSTM

ACTUAL, FORECAST = "A", "F"


# ------------------------------------------------------------------ data
def fetch_live_raw(urls=(LIVE_YEAR_URL, DEMAND_UPDATE_URL)):
    """Download the current-year and demand-update files (both raw half-hourly NESO format).
    The update file also contains rows marked "F" for the coming week; those rows hold only
    embedded wind/solar forecasts (demand is 0), so only "A" (actual) rows are ever used."""
    frames = [pd.read_csv(url).assign(source=name) for name, url in zip(["year", "update"], urls)]
    raw = pd.concat(frames, ignore_index=True)
    if "FORECAST_ACTUAL_INDICATOR" not in raw.columns:
        raw["FORECAST_ACTUAL_INDICATOR"] = ACTUAL
    return raw


def live_actuals_daily(raw):
    """Daily actual demand from the live files. Where both files cover the same half-hour,
    the update file (listed second, more recent) wins. Incomplete days are dropped."""
    actual = raw[raw["FORECAST_ACTUAL_INDICATOR"] == ACTUAL].copy()
    actual["_date"] = parse_settlement_dates(actual["SETTLEMENT_DATE"])
    actual = (actual.drop_duplicates(["_date", "SETTLEMENT_PERIOD"], keep="last")
                    .drop(columns="_date"))
    return to_daily(clean_half_hourly(actual), drop_incomplete=True)


def build_live_dataset(raw, history_path=PROCESSED_DATA):
    """Saved history (2021-2025) + live actuals since, as one continuous daily series with
    calendar features, extended HORIZON days past the last actual (for the forecast inputs)."""
    history = pd.read_csv(history_path, index_col="date", parse_dates=True)
    live = live_actuals_daily(raw)
    live = live[live.index > history.index.max()]
    combined = pd.concat([history[[TARGET]], live[[TARGET]]]).sort_index()

    last_actual = combined.index.max()
    full_index = pd.date_range(combined.index.min(), last_actual + pd.Timedelta(days=HORIZON))
    combined = combined.reindex(full_index)
    combined.index.name = "date"

    gaps = combined.loc[:last_actual, TARGET].isna().sum()
    if gaps:
        combined.loc[:last_actual, TARGET] = combined.loc[:last_actual, TARGET].interpolate()
    return add_calendar_features(combined), last_actual, int(gaps)


# ------------------------------------------------------------------ models
def forecast_seasonal_naive(data, origin):
    dates = pd.date_range(origin, periods=HORIZON)
    return pd.Series([data.loc[d - pd.Timedelta(days=7), TARGET] for d in dates], index=dates)


def forecast_prophet(data, origin):
    from prophet import Prophet          # imported here: slow to import, only needed for this model
    import logging
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").setLevel(logging.WARNING)

    history = data.loc[:origin - pd.Timedelta(days=1)]
    hol = data[data["is_holiday"] == 1]
    holidays_df = pd.DataFrame({"ds": hol.index, "holiday": hol["holiday_name"].values})

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                holidays=holidays_df, uncertainty_samples=0)
    m.add_regressor("is_christmas_period")
    m.fit(pd.DataFrame({"ds": history.index, "y": history[TARGET].values,
                        "is_christmas_period": history["is_christmas_period"].values}))

    dates = pd.date_range(origin, periods=HORIZON)
    future = pd.DataFrame({"ds": dates, "is_christmas_period": data.loc[dates, "is_christmas_period"].values})
    return pd.Series(m.predict(future)["yhat"].values, index=dates)


def load_lstm_ensemble(models_dir=MODELS_DIR, pattern="lstm_tuned_seed*.pt"):
    """Load every saved ensemble member. Returns (models, lookback, scaler_min, scaler_max)."""
    paths = sorted(Path(models_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No LSTM checkpoints matching {pattern} in {models_dir}")
    models, settings = [], set()
    for p in paths:
        ckpt = torch.load(p, weights_only=False, map_location="cpu")
        cfg = ckpt["config"]
        model = DemandLSTM(n_past_features=7, n_future_features=6, horizon=HORIZON,
                           hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
        settings.add((cfg["lookback"], ckpt["scaler_min"], ckpt["scaler_max"]))
    if len(settings) != 1:
        raise ValueError("Ensemble members were trained with different lookbacks or scalers")
    lookback, smin, smax = settings.pop()
    return models, lookback, smin, smax


def forecast_lstm(data, origin, ensemble):
    """Average of the ensemble members' forecasts, using the training-period scaler."""
    models, lookback, smin, smax = ensemble
    frame = add_calendar_encodings(data)
    frame["demand_scaled"] = (frame[TARGET] - smin) / (smax - smin)

    from src.lstm_data import FUTURE_COLS, PAST_COLS
    pos = frame.index.get_loc(origin)
    x_past = torch.from_numpy(frame[PAST_COLS].iloc[pos - lookback:pos].to_numpy(np.float32))[None]
    x_future = torch.from_numpy(frame[FUTURE_COLS].iloc[pos:pos + HORIZON].to_numpy(np.float32))[None]
    if torch.isnan(x_past).any():
        raise ValueError("Missing demand values in the LSTM input window")

    with torch.no_grad():
        scaled = np.mean([m(x_past, x_future).numpy()[0] for m in models], axis=0)
    return pd.Series(scaled * (smax - smin) + smin, index=pd.date_range(origin, periods=HORIZON))
