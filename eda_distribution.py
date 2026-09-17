"""
Section 2, item 1-2 (partial): total traffic distribution across Milan's 10,000
areas, and identification of the top-3 highest-traffic areas over the full
observation period.

Reads the Parquet file produced by load_milan_data.py.

Usage:
    python3 eda_distribution.py --parquet milan_out/milan_internet_traffic.parquet --out-dir eda_figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def compute_total_traffic_per_area(df: pd.DataFrame) -> pd.Series:
    """Sum internet_traffic per square_id across the full observation period."""
    return df.groupby("square_id", observed=True)["internet_traffic"].sum().sort_values(
        ascending=False
    )


def plot_distribution(total_traffic: pd.Series, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear-scale histogram
    axes[0].hist(total_traffic.values, bins=60, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Total internet traffic (full period)")
    axes[0].set_ylabel("Number of areas")
    axes[0].set_title("Distribution of total traffic across areas")

    # Log-scale histogram — total traffic across areas is typically very
    # right-skewed (city-center squares dominate), so a log-x view usually
    # shows the shape far more clearly than the linear one
    axes[1].hist(total_traffic.values, bins=60, color="darkorange", edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Total internet traffic (log scale)")
    axes[1].set_ylabel("Number of areas")
    axes[1].set_title("Same distribution, log-x scale")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df):,} rows, {df['square_id'].nunique():,} unique areas, "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")

    total_traffic = compute_total_traffic_per_area(df)

    print("\nSummary statistics of total traffic per area:")
    print(total_traffic.describe())

    skew = total_traffic.skew()
    print(f"\nSkewness: {skew:.2f}  "
          f"({'strongly right-skewed' if skew > 1 else 'moderately/lightly skewed'})")

    top_areas = total_traffic.head(args.top_n)
    print(f"\nTop {args.top_n} areas by total traffic:")
    for sq, val in top_areas.items():
        print(f"  square_id={sq}: {val:,.1f}")

    dist_path = os.path.join(args.out_dir, "traffic_distribution.png")
    plot_distribution(total_traffic, dist_path)
    print(f"\nSaved distribution figure -> {dist_path}")

    # Save the ranked totals too, so later steps (time series plots, model
    # selection area choice) don't need to recompute this from scratch
    ranked_path = os.path.join(args.out_dir, "total_traffic_by_area.csv")
    total_traffic.to_csv(ranked_path, header=["total_internet_traffic"])
    print(f"Saved full ranked totals -> {ranked_path}")


if __name__ == "__main__":
    main()
