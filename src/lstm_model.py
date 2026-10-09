"""The LSTM forecaster and its training loop."""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DemandLSTM(nn.Module):
    """Reads the past LOOKBACK days with an LSTM, then combines what it learned with the
    calendar features of the HORIZON days ahead, and predicts all HORIZON days at once."""

    def __init__(self, n_past_features, n_future_features, horizon,
                 hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_past_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,                       # inputs shaped (batch, time, features)
        )
        # Linear head: no sigmoid, because scaled test values can fall outside 0-1
        self.head = nn.Linear(hidden_size + horizon * n_future_features, horizon)

    def forward(self, x_past, x_future):
        # x_past:   (batch, lookback, n_past_features)
        # x_future: (batch, horizon, n_future_features)
        _, (h_n, _) = self.lstm(x_past)              # h_n: (num_layers, batch, hidden)
        summary = h_n[-1]                            # last layer's final state: (batch, hidden)
        future_flat = x_future.flatten(start_dim=1)  # (batch, horizon * n_future_features)
        return self.head(torch.cat([summary, future_flat], dim=1))   # (batch, horizon)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X_past, X_future, y, batch_size, shuffle, seed=None):
    dataset = TensorDataset(torch.from_numpy(X_past), torch.from_numpy(X_future), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_abs_err, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x_past, x_future, y in loader:
            pred = model(x_past, x_future)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # guards against exploding gradients
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_abs_err += (pred - y).abs().sum().item()
            n += len(y)
    horizon = loader.dataset.tensors[2].shape[1]
    return total_loss / n, total_abs_err / (n * horizon)   # mean loss, mean absolute error (scaled units)


def train_model(model, train_loader, val_loader, mw_per_unit, lr=1e-3,
                max_epochs=300, patience=25, verbose_every=10, verbose=True):
    """Train with early stopping on validation loss. Restores the best weights at the end.
    mw_per_unit converts scaled errors back to MW (the scaler's max - min), for readable logs."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae_mw": []}
    best_val, best_epoch, best_state = float("inf"), 0, None
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_mw"].append(val_mae * mw_per_unit)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % verbose_every == 0):
            print(f"epoch {epoch:3d}  train loss {train_loss:.5f}  val loss {val_loss:.5f}  "
                  f"val MAE {val_mae * mw_per_unit:,.0f} MW  (best epoch {best_epoch})")

        if epoch - best_epoch >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs.")
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"Best epoch {best_epoch}: val loss {best_val:.5f}, "
              f"val MAE {history['val_mae_mw'][best_epoch - 1]:,.0f} MW  ({time.time() - t0:.0f}s)")
    return history, best_epoch
