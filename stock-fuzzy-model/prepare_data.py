"""Fetch data and build inspection CSVs. No torch import.

The ^GSPC processed CSV is written for inspection only. Training data is
rebuilt by main.py from the same cache with the train-only split/scaler rules.
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg
import data as D
import fuzzy as fz

RAW_CSV = os.path.join("data", "raw_GSPC.csv")
PROC_CSV = os.path.join("data", "processed_GSPC.csv")
RAW_10D_CSV = os.path.join("data", "raw_GSPC_10d.csv")
PROC_10D_CSV = os.path.join("data", "processed_GSPC_10d.csv")


def fetch_raw() -> pd.DataFrame:
    """Return the cached ^GSPC OHLCV frame with a Date column."""
    raw = D.fetch_gspc(cfg.START_DATE, cfg.END_DATE).reset_index()
    if "Date" not in raw.columns and "index" in raw.columns:
        raw = raw.rename(columns={"index": "Date"})
    return raw


def write_10day(raw: pd.DataFrame, processed: pd.DataFrame):
    raw10 = raw.tail(10).copy()
    proc10 = processed.tail(10).copy()
    raw10.to_csv(RAW_10D_CSV, index=False)
    proc10.to_csv(PROC_10D_CSV, index=False)
    raw_dates = pd.to_datetime(raw10["Date"])
    proc_dates = pd.to_datetime(proc10["Date"])
    print(f"Saved {RAW_10D_CSV} ({len(raw10)} rows: "
          f"{raw_dates.iloc[0].date()}..{raw_dates.iloc[-1].date()})")
    print(f"Saved {PROC_10D_CSV} ({len(proc10)} rows: "
          f"{proc_dates.iloc[0].date()}..{proc_dates.iloc[-1].date()})")


def prepare_gspc_processed():
    print(f"Fetching ^GSPC {cfg.START_DATE}..{cfg.END_DATE} "
          f"(yfinance end exclusive, auto_adjust=True)")
    raw = fetch_raw()
    raw.to_csv(RAW_CSV, index=False)
    print(f"Saved {RAW_CSV}")

    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    df = D.attach_macro(df, cfg.START_DATE, cfg.END_DATE)

    eng = D.engineer_features(df)
    train_df, val_df, test_df = D.split_data(eng)

    n = len(eng)
    splits = np.array(
        ["train"] * len(train_df)
        + ["val"] * len(val_df)
        + ["test"] * len(test_df)
    )
    if len(splits) != n:
        raise AssertionError(f"Split mismatch: {len(splits)} labels for {n} rows")

    eng = eng.copy()
    eng["split"] = splits

    scaler = StandardScaler()
    scaler.fit(train_df[cfg.ALL_FEATURES].values)
    scaled = np.clip(
        scaler.transform(eng[cfg.ALL_FEATURES].values), -cfg.CLIP, cfg.CLIP
    )
    for i, c in enumerate(cfg.ALL_FEATURES):
        eng[c + "_scaled"] = scaled[:, i]

    train_vol_sorted = np.sort(train_df[cfg.VOL_COL].values)
    eng["vol_rank"] = D._vol_rank(eng[cfg.VOL_COL].values, train_vol_sorted)
    eng["fuzzy_lr"] = [fz.fuzzy_lr(v) for v in eng["vol_rank"].values]

    out = eng.reset_index()
    out.to_csv(PROC_CSV, index=False)
    print(f"Saved {PROC_CSV}")
    print(f"Feature count: {len(cfg.ALL_FEATURES)} -> {cfg.ALL_FEATURES}")

    write_10day(raw, out)

    for name, frame in [("raw", pd.read_csv(RAW_CSV)),
                        ("processed", pd.read_csv(PROC_CSV))]:
        print(f"--- {name} ---")
        print(f"shape={frame.shape} NaNs={int(frame.isna().sum().sum())}")
        print("head:")
        print(frame.head(3).to_string())
        print("tail:")
        print(frame.tail(3).to_string())
        if name == "processed":
            print(f"per-split rows + mean `{cfg.TARGET_COL}`:")
            for s in ["train", "val", "test"]:
                sub = frame[frame["split"] == s]
                print(f"  {s}: n={len(sub)} "
                      f"mean={sub[cfg.TARGET_COL].mean():.6f}")


def main():
    os.makedirs("data", exist_ok=True)
    prepare_gspc_processed()


if __name__ == "__main__":
    main()
