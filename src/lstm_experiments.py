"""Helpers for LSTM tuning and seed checks: build data for any lookback, train one run, predict in MW."""
import numpy as np
import pandas as pd
import torch

from src.config import HORIZON, TRAIN_END, forecast_origins
from src.lstm_config import VAL_START
from src.lstm_data import make_windows, origins_between
from src.lstm_model import DemandLSTM, make_loader, set_seed, train_model


def build_splits(frame, lookback):
    """Train / validation / test windows for a given lookback. `frame` must already contain demand_scaled."""
    lookback = int(lookback)
    train_end = VAL_START - pd.Timedelta(days=1)
    origins = {
        "train": origins_between(frame.index, frame.index[lookback], train_end, HORIZON),
        "val": origins_between(frame.index, VAL_START, TRAIN_END, HORIZON),
        "test": forecast_origins(),
    }
    splits = {}
    for name, o in origins.items():
        Xp, Xf, y, kept = make_windows(frame, o, lookback, HORIZON)
        splits[name] = {"X_past": Xp, "X_future": Xf, "y": y, "origins": kept}
    assert list(splits["test"]["origins"]) == list(origins["test"]), "test origins changed"
    return splits


def train_one(splits, hidden_size, num_layers, lr, seed, mw_per_unit,
              batch_size=32, max_epochs=300, patience=25, dropout=0.0):
    """Train one model with early stopping on validation. Returns (model, best_epoch, best_val_mae_mw)."""
    # Settings may arrive as numpy types (e.g. read from a pandas index); PyTorch needs plain Python numbers
    hidden_size, num_layers, lr, seed = int(hidden_size), int(num_layers), float(lr), int(seed)
    set_seed(seed)
    tr, va = splits["train"], splits["val"]
    train_loader = make_loader(tr["X_past"], tr["X_future"], tr["y"], batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(va["X_past"], va["X_future"], va["y"], batch_size, shuffle=False)
    model = DemandLSTM(
        n_past_features=tr["X_past"].shape[2], n_future_features=tr["X_future"].shape[2],
        horizon=HORIZON, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
    )
    history, best_epoch = train_model(model, train_loader, val_loader, mw_per_unit, lr=lr,
                                      max_epochs=max_epochs, patience=patience, verbose=False)
    return model, best_epoch, history["val_mae_mw"][best_epoch - 1]


def predict_mw(model, X_past, X_future, scaler):
    """Predict and convert back to MW. Returns an array shaped (n_windows, HORIZON)."""
    model.eval()
    with torch.no_grad():
        scaled = model(torch.from_numpy(X_past), torch.from_numpy(X_future)).numpy()
    return scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape)
