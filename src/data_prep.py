"""Turns raw NESO half-hourly demand files into the clean daily series used by every model.

This is the same logic as notebooks/02_data_prep.ipynb, as reusable functions, so the
dashboard processes newly downloaded data exactly the way the training data was processed.
"""
import holidays
import pandas as pd

KEEP_COLUMNS = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND", "TSD",
                "EMBEDDED_SOLAR_GENERATION", "EMBEDDED_SOLAR_CAPACITY"]
SPIKE_THRESHOLD_MW = 8000
MIN_PERIODS_PER_DAY = 46          # clock-change days have 46 or 50; fewer means an incomplete day


def parse_settlement_dates(raw_dates):
    """Parse NESO date strings with explicit formats, so day and month can never be swapped.
    Known formats: 01-JAN-2022, 01-Jan-23, 2025-01-08. Anything else raises an error."""
    s = raw_dates.astype(str).str.strip().str.strip('"')
    fmt_long = s.str.fullmatch(r"\d{2}-[A-Za-z]{3}-\d{4}")
    fmt_short = s.str.fullmatch(r"\d{2}-[A-Za-z]{3}-\d{2}")
    fmt_iso = s.str.match(r"\d{4}-\d{2}-\d{2}")

    dates = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    dates[fmt_long] = pd.to_datetime(s[fmt_long], format="%d-%b-%Y")
    dates[fmt_short] = pd.to_datetime(s[fmt_short], format="%d-%b-%y")
    dates[fmt_iso] = pd.to_datetime(s[fmt_iso].str[:10], format="%Y-%m-%d")

    unknown = dates.isna()
    if unknown.any():
        raise ValueError(f"Unrecognised SETTLEMENT_DATE format, e.g. {s[unknown].iloc[0]!r}")
    return dates


def clean_half_hourly(raw):
    """Parse dates, check order, build DST-safe timestamps, and clean isolated spikes.
    `raw` holds the raw NESO columns for one or more files, in their original row order."""
    df = raw[[c for c in KEEP_COLUMNS if c in raw.columns]].copy()
    df["date"] = parse_settlement_dates(df["SETTLEMENT_DATE"])
    if "source" in raw.columns:
        df["source"] = raw["source"].values
        in_order = df.groupby("source", sort=False)["date"].apply(lambda d: d.is_monotonic_increasing)
        if not in_order.all():
            raise ValueError(f"Dates go backwards after parsing in: {list(in_order[~in_order].index)}")

    offset = pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")
    df["datetime_utc"] = df["date"].dt.tz_localize("Europe/London").dt.tz_convert("UTC") + offset
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    rolling_median = df["ND"].rolling(window=7, center=True, min_periods=1).median()
    df["is_spike"] = (df["ND"] - rolling_median).abs() > SPIKE_THRESHOLD_MW
    df["ND_clean"] = df["ND"].mask(df["is_spike"]).interpolate(limit_direction="both")
    return df


def to_daily(half_hourly, drop_incomplete=True):
    """Daily mean demand (the model target) plus peak, energy and solar columns.
    Incomplete days (e.g. today, still in progress) are dropped by default."""
    daily = (half_hourly.groupby("date")
             .agg(demand_mw=("ND_clean", "mean"),
                  peak_mw=("ND_clean", "max"),
                  energy_mwh=("ND_clean", lambda s: s.sum() * 0.5),
                  n_periods=("ND_clean", "size"),
                  solar_capacity_mw=("EMBEDDED_SOLAR_CAPACITY", "mean"),
                  solar_generation_mw=("EMBEDDED_SOLAR_GENERATION", "mean")))
    if drop_incomplete:
        daily = daily[daily["n_periods"] >= MIN_PERIODS_PER_DAY]
    daily.index.name = "date"
    return daily


def add_calendar_features(daily):
    """Day of week, month, weekend, England & Wales bank holidays, Christmas period.
    Works for any dates, including future ones (holidays are generated, not looked up)."""
    out = daily.copy()
    idx = out.index
    years = range(idx.year.min(), idx.year.max() + 1)
    eng_holidays = holidays.country_holidays("GB", subdiv="ENG", years=years)

    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    out["day_of_year"] = idx.dayofyear
    out["year"] = idx.year
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = idx.isin(pd.to_datetime(list(eng_holidays.keys()))).astype(int)
    out["holiday_name"] = [eng_holidays.get(d.date(), "") for d in idx]
    out["is_christmas_period"] = (((idx.month == 12) & (idx.day >= 24)) |
                                  ((idx.month == 1) & (idx.day <= 2))).astype(int)
    return out


def build_daily_from_files(paths):
    """Full pipeline from raw yearly CSV files to the processed daily dataset."""
    raw = pd.concat([pd.read_csv(p).assign(source=str(p)) for p in paths], ignore_index=True)
    daily = to_daily(clean_half_hourly(raw)).asfreq("D")
    daily.index.name = "date"
    return add_calendar_features(daily)
