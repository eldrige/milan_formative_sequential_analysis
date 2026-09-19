"""
Section 4 shared pipeline — used by all three forecasting models (SARIMA, LSTM, TCN)
so preprocessing, train/test split, and windowing are identical and comparable
across models.

Train/test split: train on everything BEFORE 2013-12-16, evaluate one-step-ahead
forecasts across the week 2013-12-16 to 2013-12-22 (inclusive), per the assignment.

Sequence models (LSTM/TCN) need a windowed (X, y) representation; SARIMA works
directly on the raw series, so this module exposes both.

Import this module from your model-training scripts:
    from shared_pipeline import load_area_series, get_train_test_split, make_windows, ForecastScaler
"""

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2013-12-16")
TEST_END = pd.Timestamp("2013-12-23")  # exclusive upper bound -> covers Dec 16-22 inclusive

DEFAULT_SEQ_LEN = 144  # 1 day at 10-min resolution; justified by ACF analysis in Section 2


def load_area_series(df: pd.DataFrame, square_id: int) -> pd.Series:
    """Extract a single area's series as a regularly-spaced 10-min series,
    interpolating any small gaps (consistent with the EDA scripts)."""
    subset = df[df["square_id"] == square_id].sort_values("timestamp")
    s = subset.set_index("timestamp")["internet_traffic"].astype("float32")
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    n_missing = len(full_index) - len(s)
    if n_missing:
        print(f"  square {square_id}: interpolating {n_missing} missing 10-min slots")
    s = s.reindex(full_index).interpolate(limit_direction="both")
    return s


def get_train_test_split(series: pd.Series, seq_len: int = DEFAULT_SEQ_LEN):
    """
    Split into train/test by date. The test portion includes `seq_len` steps of
    context BEFORE 2013-12-16 so the first prediction in the test week still has
    a full lookback window (those context steps are only used as model input,
    never scored as predictions).
    """
    if series.index.min() > TEST_START or series.index.max() < TEST_END:
        raise ValueError(
            f"Series does not fully cover the test window {TEST_START} - {TEST_END}. "
            f"Series covers {series.index.min()} to {series.index.max()}."
        )

    train = series[series.index < TEST_START]
    context_start = TEST_START - pd.Timedelta(minutes=10 * seq_len)
    test_with_context = series[(series.index >= context_start) & (series.index < TEST_END)]
    test_scored = series[(series.index >= TEST_START) & (series.index < TEST_END)]

    return train, test_with_context, test_scored


class ForecastScaler:
    """Simple min-max scaler fit ONLY on training data, to avoid test-set leakage.
    Kept as its own tiny class (rather than sklearn) so it's trivial to invert
    predictions back to the original traffic units for plotting/metrics."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, series: pd.Series):
        self.min_ = float(series.min())
        self.max_ = float(series.max())
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Call fit() on training data first")
        rng = self.max_ - self.min_
        rng = rng if rng > 0 else 1.0
        return ((series.values - self.min_) / rng).astype("float32")

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        rng = self.max_ - self.min_
        rng = rng if rng > 0 else 1.0
        return values * rng + self.min_


def make_windows(values: np.ndarray, seq_len: int):
    """
    Build (X, y) pairs for one-step-ahead supervised learning from a 1D array.
    X[i] = values[i : i+seq_len], y[i] = values[i+seq_len]  (predict the next step)
    Returns X of shape (n_samples, seq_len), y of shape (n_samples,).
    """
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def prepare_area(df: pd.DataFrame, square_id: int, seq_len: int = DEFAULT_SEQ_LEN):
    """
    End-to-end prep for one area: returns a dict with everything a model needs —
    raw train/test series (for SARIMA), scaled windowed arrays (for LSTM/TCN),
    the fitted scaler (to invert predictions), and the timestamps corresponding
    to each scored test prediction (for plotting against ground truth).
    """
    series = load_area_series(df, square_id)
    train, test_with_context, test_scored = get_train_test_split(series, seq_len)

    scaler = ForecastScaler().fit(train)

    train_scaled = scaler.transform(train)
    test_ctx_scaled = scaler.transform(test_with_context)

    X_train, y_train = make_windows(train_scaled, seq_len)
    X_test, y_test = make_windows(test_ctx_scaled, seq_len)

    # timestamps for each scored test point (aligned with y_test / X_test)
    test_timestamps = test_with_context.index[seq_len:]
    assert len(test_timestamps) == len(y_test) == len(test_scored), (
        f"Alignment mismatch: {len(test_timestamps)} timestamps vs "
        f"{len(y_test)} y_test vs {len(test_scored)} test_scored"
    )

    return {
        "square_id": square_id,
        "seq_len": seq_len,
        "series": series,
        "train_raw": train,
        "test_scored_raw": test_scored,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_timestamps": test_timestamps,
    }


if __name__ == "__main__":
    # quick self-check when run directly
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    ranked = pd.read_csv(args.ranked_csv, index_col=0)
    top3 = ranked.index[:3].tolist()

    for sq in top3:
        print(f"\n=== square {sq} ===")
        data = prepare_area(df, int(sq), seq_len=args.seq_len)
        print(f"  train: {len(data['train_raw']):,} steps "
              f"({data['train_raw'].index.min()} to {data['train_raw'].index.max()})")
        print(f"  test (scored): {len(data['test_scored_raw']):,} steps "
              f"({data['test_scored_raw'].index.min()} to {data['test_scored_raw'].index.max()})")
        print(f"  X_train {data['X_train'].shape}, X_test {data['X_test'].shape}")
