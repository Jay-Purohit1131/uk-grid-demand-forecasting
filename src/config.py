"""Shared settings for every model, so all of them are trained and scored identically."""
from pathlib import Path

import pandas as pd

# Paths are built from this file's location, so they work from notebooks/, src/ or the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "daily_demand.csv"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET = "demand_mw"
SEASONAL_PERIOD = 7          # weekly cycle (days)

# Evaluation design: train on 2021-2024, test on 2025.
# Every 7 days in the test year, each model forecasts the next HORIZON days
# using only data up to that point (a rolling forecast origin).
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")
HORIZON = 7                  # forecast 1-7 days ahead
STEP = 1                     # move the forecast origin forward one week at a time


def forecast_origins():
    """Dates on which a 7-day forecast is made. Each forecast covers origin .. origin + HORIZON - 1."""
    last_start = TEST_END - pd.Timedelta(days=HORIZON - 1)
    return list(pd.date_range(TEST_START, last_start, freq=f"{STEP}D"))

# ---- Live data (dashboard) ----
# NESO publishes the current year's half-hourly demand as a growing file, plus a "Demand Data Update"
# file with the most recent weeks of actuals. The year-specific link changes each January:
# update LIVE_YEAR_URL for 2027.
LIVE_YEAR_URL = ("https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/"
                 "resource/8a4a771c-3929-4e56-93ad-cdf13219dea5/download/demanddataupdate")
DEMAND_UPDATE_URL = ("https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/"
                     "resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate")