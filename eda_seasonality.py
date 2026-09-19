"""
Section 2, item 3: two additional analyses on the highest-traffic area —
(a) ACF/PACF to quantify daily/weekly periodicity and inform sequence length
    choices for the forecasting models,
(b) STL seasonal decomposition to separate trend/seasonal/residual and check
    stationarity of the residual (ADF test).

Also computes a weekly-seasonality "strength" comparison across all 5 areas
(top-3 + 4159 + 4556), to put a number behind any weekday/weekend contrast
you noticed visually in the time series plots.

Data is at 10-minute resolution -> 144 samples/day, 1008 samples/week.

Usage:
    python3 eda_seasonality.py \
        --parquet milan_out/milan_internet_traffic.parquet \
        --ranked-csv eda_figures/total_traffic_by_area.csv \
        --out-dir eda_figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

SAMPLES_PER_DAY = 144
SAMPLES_PER_WEEK = 144 * 7
FIXED_AREAS = [4159, 4556]


def get_area_series(df: pd.DataFrame, square_id: int) -> pd.Series:
    subset = df[df["square_id"] == square_id].sort_values("timestamp")
    s = subset.set_index("timestamp")["internet_traffic"]
    # enforce a regular 10-min index so ACF/STL lags line up with real time,
    # filling any small gaps by interpolation (report this as a limitation)
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    n_missing = len(full_index) - len(s)
    if n_missing:
        print(f"    square {square_id}: {n_missing} missing 10-min slots, interpolating")
    s = s.reindex(full_index).interpolate(limit_direction="both")
    return s


def plot_acf_pacf(series: pd.Series, square_id: int, out_dir: str, lags: int = SAMPLES_PER_WEEK + 100):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].axvline(SAMPLES_PER_DAY, color="red", linestyle="--", alpha=0.5, label="1 day")
    axes[0].axvline(SAMPLES_PER_WEEK, color="green", linestyle="--", alpha=0.5, label="1 week")
    axes[0].legend()
    axes[0].set_title(f"ACF — square {square_id}")

    plot_pacf(series, lags=200, ax=axes[1], method="ywm")
    axes[1].axvline(SAMPLES_PER_DAY, color="red", linestyle="--", alpha=0.5, label="1 day")
    axes[1].legend()
    axes[1].set_title(f"PACF — square {square_id} (first 200 lags)")

    fig.tight_layout()
    path = os.path.join(out_dir, f"acf_pacf_square_{square_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def run_stl(series: pd.Series, square_id: int, out_dir: str, period: int = SAMPLES_PER_DAY):
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"STL decomposition (daily period) — square {square_id}", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"stl_square_{square_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")

    # ADF test on the residual component -> is what's left after removing
    # trend+seasonality stationary?
    resid = result.resid.dropna()
    adf_stat, adf_p, *_ = adfuller(resid)
    print(f"  ADF test on STL residual: stat={adf_stat:.3f}, p-value={adf_p:.4f} "
          f"({'stationary' if adf_p < 0.05 else 'NOT stationary'} at 5%)")

    return result


def weekly_strength(series: pd.Series, period: int = SAMPLES_PER_DAY) -> dict:
    """Run STL with a WEEKLY period and report the seasonal component's variance
    relative to the deseasonalized series — a simple 'how much does the weekly
    pattern matter here' score, comparable across areas."""
    if len(series) < SAMPLES_PER_WEEK * 2:
        return {"weekly_strength": None, "note": "series too short for weekly STL"}

    stl = STL(series, period=SAMPLES_PER_WEEK, robust=True)
    result = stl.fit()
    # strength measure from Hyndman & Athanasopoulos: 1 - Var(resid) / Var(seasonal + resid)
    var_resid = result.resid.var()
    var_deseasonalized = (result.seasonal + result.resid).var()
    strength = max(0.0, 1 - var_resid / var_deseasonalized) if var_deseasonalized > 0 else None
    return {"weekly_strength": strength}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    ranked = pd.read_csv(args.ranked_csv, index_col=0)
    top_area = int(ranked.index[0])
    top3 = ranked.index[:3].tolist()
    print(f"Top-traffic area: {top_area}")

    # --- Primary required analyses: ACF/PACF + STL on the top-traffic area ---
    print(f"\n=== ACF/PACF: square {top_area} ===")
    top_series = get_area_series(df, top_area)
    plot_acf_pacf(top_series, top_area, args.out_dir)

    print(f"\n=== STL decomposition (daily): square {top_area} ===")
    run_stl(top_series, top_area, args.out_dir)

    # --- Bonus: weekly-seasonality-strength comparison across all 5 areas ---
    print("\n=== Weekly seasonality strength comparison (top-3 + 4159 + 4556) ===")
    compare_areas = list(dict.fromkeys(top3 + FIXED_AREAS))  # dedupe, keep order
    rows = []
    for sq in compare_areas:
        if sq not in df["square_id"].values:
            print(f"  square {sq} not found, skipping")
            continue
        s = get_area_series(df, int(sq))
        res = weekly_strength(s)
        rows.append({"square_id": sq, **res})
        ws = res["weekly_strength"]
        print(f"  square {sq}: weekly seasonality strength = "
              f"{ws:.3f}" if ws is not None else f"  square {sq}: n/a")

    summary = pd.DataFrame(rows).sort_values("weekly_strength", ascending=False)
    summary_path = os.path.join(args.out_dir, "weekly_seasonality_strength.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved comparison table -> {summary_path}")


if __name__ == "__main__":
    main()
