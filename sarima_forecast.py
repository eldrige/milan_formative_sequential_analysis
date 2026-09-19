"""
Section 4 — Model 1: SARIMA, implemented as harmonic (Fourier) regression with
ARMA errors rather than native seasonal SARIMA.

WHY NOT NATIVE SEASONAL SARIMA: with a seasonal period of 144 (daily, at 10-min
resolution), a native SARIMAX(seasonal_order=(P,D,Q,144)) blows up the
state-space dimension and does not converge in a reasonable time (tested: did
not finish fitting in 5 minutes on 45 days of training data). Representing the
daily/weekly seasonality as deterministic Fourier terms (sin/cos regressors)
instead, with a small ARMA(p,q) on the residual, captures the same seasonal
structure in ~2-3 seconds and is a standard, well-documented alternative for
long seasonal periods (sometimes called "dynamic harmonic regression").

One-step-ahead forecasting over the test week is done correctly (using true
past values at every step, not the model's own prior predictions) via
statsmodels' `.append(refit=False)` + `.get_prediction()`, which filters the
already-fitted parameters through the new true observations rather than
re-optimizing or forecasting blindly forward.

Usage:
    python3 sarima_forecast.py \
        --parquet milan_out/milan_internet_traffic.parquet \
        --ranked-csv eda_figures/total_traffic_by_area.csv \
        --out-dir results_sarima
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from shared_pipeline import DEFAULT_SEQ_LEN, get_train_test_split, load_area_series

DAILY_PERIOD = 144
WEEKLY_PERIOD = 144 * 7
ORDER = (2, 0, 2)          # non-seasonal ARMA order on top of the Fourier terms
DAILY_HARMONICS = 2        # number of sin/cos pairs for the daily cycle
WEEKLY_HARMONICS = 1       # number of sin/cos pairs for the weekly cycle


def fourier_terms(t: np.ndarray, period: int, n_harmonics: int) -> np.ndarray:
    cols = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)


def build_exog(t: np.ndarray) -> np.ndarray:
    daily = fourier_terms(t, DAILY_PERIOD, DAILY_HARMONICS)
    weekly = fourier_terms(t, WEEKLY_PERIOD, WEEKLY_HARMONICS)
    return np.hstack([daily, weekly])


def run_area(df: pd.DataFrame, square_id: int, out_dir: str, seq_len: int = DEFAULT_SEQ_LEN):
    print(f"\n=== SARIMA (Fourier) — square {square_id} ===")
    series = load_area_series(df, square_id)
    train, _, test_scored = get_train_test_split(series, seq_len)

    n_train = len(train)
    n_test = len(test_scored)
    t_train = np.arange(n_train)
    t_test = np.arange(n_train, n_train + n_test)  # continuous index -> phase continuity

    exog_train = build_exog(t_train)
    exog_test = build_exog(t_test)

    # --- Fit ---
    t0 = time.perf_counter()
    model = SARIMAX(
        train.values,
        order=ORDER,
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=100)
    fit_time = time.perf_counter() - t0
    print(f"  fit time: {fit_time:.2f}s")

    # --- One-step-ahead forecasts across the test week, using TRUE past values ---
    t0 = time.perf_counter()
    extended = res.append(test_scored.values, exog=exog_test, refit=False)
    pred = extended.get_prediction(start=n_train, end=n_train + n_test - 1)
    predicted_mean = pred.predicted_mean
    inference_time = time.perf_counter() - t0
    print(f"  inference time (whole test week, {n_test} one-step forecasts): {inference_time:.2f}s")

    # --- Save predictions in the shared format used by all 3 models ---
    result_df = pd.DataFrame({
        "timestamp": test_scored.index,
        "actual": test_scored.values,
        "predicted": predicted_mean,
    })
    pred_path = os.path.join(out_dir, f"predictions_sarima_square_{square_id}.csv")
    result_df.to_csv(pred_path, index=False)
    print(f"  Saved -> {pred_path}")

    # --- Metrics ---
    err = result_df["actual"] - result_df["predicted"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    # avoid div-by-zero in MAPE for near-zero actuals
    nonzero = result_df["actual"].abs() > 1e-6
    mape = (err[nonzero].abs() / result_df["actual"][nonzero].abs()).mean() * 100

    metrics = {
        "square_id": square_id,
        "model": "SARIMA (Fourier)",
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "fit_time_s": fit_time,
        "inference_time_s": inference_time,
        "n_test_points": n_test,
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

    df = pd.read_parquet(args.parquet)
    ranked = pd.read_csv(args.ranked_csv, index_col=0)
    top3 = ranked.index[:3].tolist()

    all_metrics = []
    for sq in top3:
        metrics = run_area(df, int(sq), args.out_dir, seq_len=args.seq_len)
        all_metrics.append(metrics)

    metrics_path = os.path.join(args.out_dir, "metrics_sarima.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
