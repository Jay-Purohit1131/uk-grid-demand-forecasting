"""Turns the daily series into LSTM training windows.

For each forecast origin (the first day being forecast), one sample contains:
  X_past   - the LOOKBACK days before the origin: scaled demand + calendar features
  X_future - the HORIZON days being forecast: calendar features only (known in advance)
  y        - the HORIZON days of scaled demand to predict
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CALENDAR_COLS = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday", "is_christmas_period"]
PAST_COLS = ["demand_scaled"] + CALENDAR_COLS
FUTURE_COLS = CALENDAR_COLS


def add_calendar_encodings(daily):
    """Encode day-of-week and day-of-year as points on a circle (sin/cos), so that
    Sunday sits next to Monday and 31 Dec sits next to 1 Jan. Plain numbers (0-6, 1-366)
    would tell the model that Sunday and Monday are as far apart as possible."""
    out = daily.copy()
    dow = out.index.dayofweek
    doy = out.index.dayofyear
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def fit_scaler(daily, target, fit_end):
    """Fit MinMaxScaler on the training period ONLY (up to and including fit_end)."""
    scaler = MinMaxScaler()
    scaler.fit(daily.loc[:fit_end, [target]])
    return scaler


def make_windows(frame, origins, lookback, horizon):
    """Build (X_past, X_future, y) for each origin. `frame` must contain PAST_COLS and
    FUTURE_COLS and have one row per day. Origins without enough history or future are skipped."""
    positions = frame.index.get_indexer(pd.DatetimeIndex(origins))
    past_values = frame[PAST_COLS].to_numpy(dtype=np.float32)
    future_values = frame[FUTURE_COLS].to_numpy(dtype=np.float32)
    target_values = frame["demand_scaled"].to_numpy(dtype=np.float32)

    X_past, X_future, y, kept = [], [], [], []
    for origin, p in zip(origins, positions):
        if p < lookback or p + horizon > len(frame):
            continue
        X_past.append(past_values[p - lookback:p])          # days origin-lookback .. origin-1
        X_future.append(future_values[p:p + horizon])      # days origin .. origin+horizon-1
        y.append(target_values[p:p + horizon])
        kept.append(origin)
    return np.stack(X_past), np.stack(X_future), np.stack(y), pd.DatetimeIndex(kept)


def origins_between(index, first, last_target_day, horizon):
    """Every daily origin from `first` whose whole forecast window ends on or before last_target_day."""
    last_origin = pd.Timestamp(last_target_day) - pd.Timedelta(days=horizon - 1)
    return [d for d in index if pd.Timestamp(first) <= d <= last_origin]
