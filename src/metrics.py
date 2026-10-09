"""Forecast error metrics shared by every model."""
import numpy as np


def _as_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaN found in y_true or y_pred")
    return y_true, y_pred


def mae(y_true, y_pred):
    """Mean absolute error, in MW."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error, in MW. Punishes large misses more than MAE."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    """Mean absolute percentage error, in %. Safe here because demand is never near zero."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mase(y_true, y_pred, y_train, m=7):
    """Mean absolute scaled error: MAE divided by the in-sample MAE of a seasonal-naive
    forecast ('same day last week'). Below 1 means better than that naive benchmark."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def score(y_true, y_pred, y_train=None, m=7):
    """All metrics in one dict. MASE is included only when y_train is given."""
    out = {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)}
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train, m)
    return out
