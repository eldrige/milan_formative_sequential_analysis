"""
Section 4 — Model 2: LSTM.

Uses the SAME windowed (X, y) data as the shared pipeline (seq_len steps of
history -> next-step prediction), scaled with the SAME train-only scaler as
the other models, so the comparison across models is fair.

Architecture: a small single-layer LSTM (64 hidden units) + a linear output
head. Kept deliberately simple/shallow given the size of a single area's
series (a handful of weeks of 10-min data) -- a large/deep LSTM would be prone
to overfitting a series this short, per the "LSTM needs substantial data"
limitation discussed in Section 3.

Training: Adam optimizer, MSE loss, early stopping on a held-out validation
slice (the last few days of the training period) to avoid picking an
arbitrary fixed epoch count.

One-step-ahead evaluation over the test week uses the TRUE past values at each
step (X_test windows are built from ground truth, never from the model's own
prior predictions) -- consistent with the SARIMA script and the assignment's
formal definition of one-step-ahead forecasting.

Usage:
    python3 lstm_forecast.py \
        --parquet milan_out/milan_internet_traffic.parquet \
        --ranked-csv eda_figures/total_traffic_by_area.csv \
        --out-dir results_lstm
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared_pipeline import DEFAULT_SEQ_LEN, prepare_area

HIDDEN_SIZE = 64
NUM_LAYERS = 1
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 5
VAL_DAYS = 5  # last N days of the training period held out for early stopping
LEARNING_RATE = 1e-3


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # final hidden state
        return self.head(last_step).squeeze(-1)


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (n, seq_len, 1)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_model(X_train, y_train, seq_len, device):
    # carve out the last VAL_DAYS of training data as a validation split for
    # early stopping (chronological, not random, to respect time ordering)
    val_size = VAL_DAYS * 144
    if val_size >= len(X_train):
        val_size = max(1, len(X_train) // 10)

    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]

    train_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = LSTMForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_run = 0

    t0 = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))
        epochs_run = epoch

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"    early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_time, epochs_run, best_val_loss


def run_area(df: pd.DataFrame, square_id: int, out_dir: str, device, seq_len: int = DEFAULT_SEQ_LEN):
    print(f"\n=== LSTM — square {square_id} ===")
    data = prepare_area(df, square_id, seq_len=seq_len)

    print(f"  training on {len(data['X_train']):,} windows, device={device}")
    model, train_time, epochs_run, best_val_loss = train_model(
        data["X_train"], data["y_train"], seq_len, device
    )
    print(f"  train time: {train_time:.2f}s over {epochs_run} epochs "
          f"(best val MSE, scaled: {best_val_loss:.5f})")

    # --- Inference: one batched forward pass over all test windows (true-history inputs) ---
    t0 = time.perf_counter()
    model.eval()
    X_test_t = torch.tensor(data["X_test"], dtype=torch.float32).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred_scaled = model(X_test_t).cpu().numpy()
    inference_time = time.perf_counter() - t0
    print(f"  inference time ({len(pred_scaled)} one-step forecasts): {inference_time:.3f}s")

    predicted = data["scaler"].inverse_transform(pred_scaled)
    actual = data["scaler"].inverse_transform(data["y_test"])

    result_df = pd.DataFrame({
        "timestamp": data["test_timestamps"],
        "actual": actual,
        "predicted": predicted,
    })
    pred_path = os.path.join(out_dir, f"predictions_lstm_square_{square_id}.csv")
    result_df.to_csv(pred_path, index=False)
    print(f"  Saved -> {pred_path}")

    err = result_df["actual"] - result_df["predicted"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    nonzero = result_df["actual"].abs() > 1e-6
    mape = (err[nonzero].abs() / result_df["actual"][nonzero].abs()).mean() * 100

    metrics = {
        "square_id": square_id,
        "model": "LSTM",
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "fit_time_s": train_time,
        "inference_time_s": inference_time,
        "epochs_run": epochs_run,
        "n_test_points": len(result_df),
    }
    print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print(f"  torch threads: {torch.get_num_threads()}")

    df = pd.read_parquet(args.parquet)
    ranked = pd.read_csv(args.ranked_csv, index_col=0)
    top3 = ranked.index[:3].tolist()

    all_metrics = []
    for sq in top3:
        metrics = run_area(df, int(sq), args.out_dir, device, seq_len=args.seq_len)
        all_metrics.append(metrics)

    metrics_path = os.path.join(args.out_dir, "metrics_lstm.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
