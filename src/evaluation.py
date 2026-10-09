"""Shared scoring, saving and comparison, so every model is evaluated by identical code."""
import pandas as pd

from src.config import PREDICTIONS_DIR, SEASONAL_PERIOD
from src.metrics import score

PRED_COLUMNS = ["model", "origin", "date", "horizon", "y_true", "y_pred"]


def evaluate_and_save(preds, name, y_train):
    """Score a model's predictions, show error by horizon, and save them.
    y_train is used only to scale MASE; pass the SAME series for every model."""
    missing = set(PRED_COLUMNS) - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")

    scores = score(preds["y_true"], preds["y_pred"], y_train=y_train, m=SEASONAL_PERIOD)
    print(f"=== {name} ===")
    for k, v in scores.items():
        print(f"  {k:5s} {v:,.3f}")

    by_horizon = (preds.assign(abs_error=(preds["y_true"] - preds["y_pred"]).abs())
                       .groupby("horizon")["abs_error"].mean().round(0))
    print("\n  MAE by horizon (days ahead):")
    print(by_horizon.to_string())

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = PREDICTIONS_DIR / f"{name}.csv"
    preds[PRED_COLUMNS].to_csv(out, index=False)
    print(f"\n  Saved {len(preds)} rows to {out}")
    return scores


def load_predictions():
    """Load every saved model's predictions into one DataFrame."""
    files = sorted(PREDICTIONS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {PREDICTIONS_DIR}")
    return pd.concat([pd.read_csv(f, parse_dates=["origin", "date"]) for f in files], ignore_index=True)


def comparison_table(all_preds, y_train):
    """One row per model: MAE, RMSE, MAPE, MASE, sorted by MAE."""
    rows = {m: score(g["y_true"], g["y_pred"], y_train=y_train, m=SEASONAL_PERIOD)
            for m, g in all_preds.groupby("model")}
    return pd.DataFrame(rows).T.sort_values("MAE")


def horizon_table(all_preds):
    """MAE by model (rows) and horizon (columns)."""
    return (all_preds.assign(abs_error=(all_preds["y_true"] - all_preds["y_pred"]).abs())
                     .pivot_table(index="model", columns="horizon", values="abs_error", aggfunc="mean"))
