"""LSTM-specific settings.

Shared settings (paths, target, test year, horizon, forecast origins) stay in config.py,
so the LSTM is still scored on exactly the same 359 forecasts as the baselines.
"""
import pandas as pd

# Split: the LSTM learns from 2021-2023 and uses 2024 to decide when to stop training.
# 2025 remains the untouched test year (defined in config.py).
VAL_START = pd.Timestamp("2024-01-01")

# Days of history in each training example (tuned on Day 21)
LOOKBACK = 28

# Model size
HIDDEN_SIZE = 64        # size of the LSTM's memory vector
NUM_LAYERS = 1          # stacked LSTM layers
DROPOUT = 0.0           # only applies between layers, so it has no effect when NUM_LAYERS = 1

# Training
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 300        # upper limit; early stopping normally ends training well before this
PATIENCE = 25           # stop after this many epochs without a new best validation loss
SEED = 42               # fixes random weight initialisation and shuffling, so runs are repeatable
