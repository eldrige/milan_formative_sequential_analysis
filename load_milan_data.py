"""
Data loading & memory-profiling pipeline for the Milan telecom activity dataset
(Telecom Italia Big Data Challenge — "sms-call-internet-mi" files).

Expected raw format: one .txt file per day, tab-separated, columns:
    square_id, time_interval, country_code, sms_in, sms_out, call_in, call_out, internet_traffic

For this assignment we only need `internet_traffic`, aggregated per (square_id, time_interval)
across country codes (a square/interval can have multiple rows, one per country — sum them).

Usage:
    python load_milan_data.py --raw-dir /path/to/raw_txt_files --out-dir /path/to/output
"""

import argparse
import glob
import os
import time
import tracemalloc

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "square_id",
    "time_interval",   # ms since epoch
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_traffic",
]

# Only these are needed downstream -> drop the rest at read time to save memory.
USE_COLUMNS = ["square_id", "time_interval", "internet_traffic"]


# ---------------------------------------------------------------------------
# Baseline (naive) loader — for the "before" memory measurement
# ---------------------------------------------------------------------------
def load_naive(raw_dir: str, sample_days: int = None) -> pd.DataFrame:
    """Read files with pandas defaults, concatenate, no dtype control.
    This is intentionally the 'bad' version used only to measure baseline memory.

    On a full 62-day / ~20GB dataset this WILL exceed 8GB of RAM if sample_days
    is None. Pass sample_days (e.g. 3) to measure the baseline on a safe subset
    and extrapolate/report that explicitly in the writeup as a methodological
    limitation.
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if sample_days is not None:
        files = files[:sample_days]
        print(
            f"  [naive] sampling {len(files)} of the available files (safety limit)")
    frames = []
    for f in files:
        df = pd.read_csv(f, sep="\t", header=None, names=RAW_COLUMNS)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Optimized loader
# ---------------------------------------------------------------------------
def load_optimized(raw_dir: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    """
    Memory-saving strategies applied:
      1. usecols            -> skip sms/call columns entirely, they're unused here
      2. explicit dtypes    -> float32 instead of float64, uint16 for square_id (max 10000)
      3. chunked reading     -> avoid one giant read_csv call per file
      4. aggregate at load time -> sum internet_traffic across country_code per
                                    (square_id, time_interval) so we never hold
                                    the un-aggregated rows for long
      5. category dtype for square_id after aggregation -> cheap grouping key later
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))

    dtype_map = {
        "square_id": "uint16",
        "time_interval": "int64",   # keep as ms timestamp, convert once at the end
        "country_code": "category",
        "internet_traffic": "float32",
    }

    day_frames = []  # one fully-aggregated frame PER FILE (62 items, not ~180)
    for i, f in enumerate(files, 1):
        t0 = time.perf_counter()
        file_agg = None
        for chunk in pd.read_csv(
            f,
            sep="\t",
            header=None,
            names=RAW_COLUMNS,
            # need country_code to group away
            usecols=USE_COLUMNS + ["country_code"],
            dtype=dtype_map,
            chunksize=chunksize,
        ):
            g = (
                chunk.groupby(["square_id", "time_interval"], observed=True)[
                    "internet_traffic"
                ]
                .sum()
                .reset_index()
            )
            if file_agg is None:
                file_agg = g
            else:
                # merge this chunk's partial sums into the running per-file total —
                # only a handful of chunks per file, so this stays cheap
                file_agg = (
                    pd.concat([file_agg, g], ignore_index=True)
                    .groupby(["square_id", "time_interval"], observed=True)["internet_traffic"]
                    .sum()
                    .reset_index()
                )
        day_frames.append(file_agg)
        elapsed = time.perf_counter() - t0
        print(f"  [optimized] {i}/{len(files)}: {os.path.basename(f)} "
              f"({len(file_agg):,} rows, {elapsed:.1f}s)", end="\r")

    print()  # clear the progress line
    print("  merging all days...")
    combined = pd.concat(day_frames, ignore_index=True)
    # each file is one calendar day, so square/interval pairs shouldn't repeat across
    # files, but this final groupby is a cheap safety net in case of any overlap
    combined = (
        combined.groupby(["square_id", "time_interval"],
                         observed=True)["internet_traffic"]
        .sum()
        .reset_index()
    )

    combined["square_id"] = combined["square_id"].astype("uint16")
    combined["internet_traffic"] = combined["internet_traffic"].astype(
        "float32")
    combined["timestamp"] = pd.to_datetime(
        combined["time_interval"], unit="ms")
    combined = combined.drop(columns=["time_interval"])
    combined = combined.sort_values(
        ["square_id", "timestamp"]).reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Memory profiling helpers
# ---------------------------------------------------------------------------
def profile_loader(name: str, loader_fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()

    df = loader_fn(*args, **kwargs)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    df_mem = df.memory_usage(deep=True).sum() / 1e6  # MB

    print(f"--- {name} ---")
    print(f"  wall time:        {elapsed:.2f} s")
    print(f"  peak traced mem:  {peak / 1e6:.1f} MB")
    print(f"  final df mem:     {df_mem:.1f} MB")
    print(f"  rows:             {len(df):,}")
    print()

    return df, {"elapsed_s": elapsed, "peak_mb": peak / 1e6, "df_mb": df_mem, "rows": len(df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True,
                        help="Directory of raw .txt daily files")
    parser.add_argument("--out-dir", required=True,
                        help="Where to write the processed Parquet file")
    parser.add_argument(
        "--skip-naive",
        action="store_true",
        help="Skip the naive baseline load entirely",
    )
    parser.add_argument(
        "--naive-sample-days",
        type=int,
        default=3,
        help="Only run the naive baseline on this many files, to stay safe on limited RAM "
        "(default: 3). Pass a larger number only if you have enough RAM to hold that many "
        "full days' raw data uncompressed in memory at once (~300-350MB/day for this dataset).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stats = {}

    if not args.skip_naive:
        _, stats["naive"] = profile_loader(
            "Naive load (sample)", load_naive, args.raw_dir, sample_days=args.naive_sample_days
        )

    df_opt, stats["optimized"] = profile_loader(
        "Optimized load", load_optimized, args.raw_dir)

    out_path = os.path.join(args.out_dir, "milan_internet_traffic.parquet")
    df_opt.to_parquet(out_path, index=False)
    print(f"Saved processed dataset -> {out_path}")

    if "naive" in stats:
        # naive ran on a sample of days, optimized ran on all files -> compare on a
        # per-day basis (peak memory / sample_days) rather than raw totals, since the
        # two runs cover different amounts of data.
        n_days = args.naive_sample_days
        naive_per_day = stats["naive"]["peak_mb"] / n_days
        total_days = 62  # adjust if your dataset spans a different number of days
        naive_extrapolated_peak = naive_per_day * total_days
        print(f"\nNaive baseline measured on {n_days} day(s):")
        print(
            f"  peak memory:                {stats['naive']['peak_mb']:.1f} MB")
        print(f"  peak memory per day:        {naive_per_day:.1f} MB/day")
        print(f"  extrapolated to {total_days} days:  ~{naive_extrapolated_peak:.1f} MB "
              f"({naive_extrapolated_peak/1000:.1f} GB) -- NOT run directly, extrapolated")
        print(
            f"\nOptimized run (all {stats['optimized']['rows']:,} rows, {len(glob.glob(os.path.join(args.raw_dir, '*.txt')))} files):")
        print(
            f"  peak memory:                {stats['optimized']['peak_mb']:.1f} MB")
        reduction = 100 * (1 - stats["optimized"]
                           ["peak_mb"] / naive_extrapolated_peak)
        print(
            f"\nEstimated peak-memory reduction (optimized vs extrapolated naive): {reduction:.1f}%")


if __name__ == "__main__":
    main()
