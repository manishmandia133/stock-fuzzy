"""Fetch ^GSPC and build raw + processed CSVs. No torch import (runs without it)."""

import os

import pandas as pd

import config as cfg
import data as D
import fuzzy as fz

RAW_CSV = os.path.join("data", "raw_GSPC.csv")
PROC_CSV = os.path.join("data", "processed_GSPC.csv")


def fetch_raw() -> pd.DataFrame:
    """Download the configured universe and return a Date-column raw frame."""
    return D.download_raw_frame(cfg.UNIVERSE[0], cfg.START_DATE, cfg.END_DATE)


def main() -> None:
    """Write raw and processed CSVs, then print shape/balance summaries."""
    os.makedirs("data", exist_ok=True)
    print(
        f"Fetching ^GSPC {cfg.START_DATE}..{cfg.END_DATE} "
        f"(yfinance end exclusive, auto_adjust=True)"
    )
    raw = fetch_raw()
    raw.to_csv(RAW_CSV, index=False)
    print(f"Saved {RAW_CSV}")

    # Re-index on Date for feature engineering.
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")

    engineered = D.engineer_features(df)
    processed, _, _ = D.build_processed_frame(engineered)
    processed["fuzzy_lr"] = [
        fz.fuzzy_lr_vol_only(v) for v in processed["vol_rank"].values
    ]

    out = processed.reset_index()
    out.to_csv(PROC_CSV, index=False)
    print(f"Saved {PROC_CSV}")

    for name, frame in [
        ("raw", pd.read_csv(RAW_CSV)),
        ("processed", pd.read_csv(PROC_CSV)),
    ]:
        print(f"--- {name} ---")
        print(f"shape={frame.shape} NaNs={int(frame.isna().sum().sum())}")
        print("head:")
        print(frame.head(3).to_string())
        print("tail:")
        print(frame.tail(3).to_string())
        if name == "processed":
            print("per-split rows + up-rate:")
            for split in ["train", "val", "test"]:
                sub = frame[frame["split"] == split]
                print(
                    f"  {split}: n={len(sub)} "
                    f"up_rate={sub['target_direction'].mean():.4f}"
                )


if __name__ == "__main__":
    main()
