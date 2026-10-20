"""Fetch ^GSPC and build raw + processed CSVs. No torch import (runs without it)."""
import os

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import config as cfg
import data as D
import fuzzy as fz

RAW_CSV = os.path.join("data", "raw_GSPC.csv")
PROC_CSV = os.path.join("data", "processed_GSPC.csv")


def fetch_raw():
    raw = yf.download("^GSPC", start=cfg.START_DATE, end=cfg.END_DATE,
                      auto_adjust=True, progress=False)
    # Flatten MultiIndex exactly as returned, e.g. ('Close','^GSPC') -> 'Close'
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if c[0] in
                       ("Open", "High", "Low", "Close", "Volume") else "_".join(map(str, c))
                       for c in raw.columns]
    raw = raw.reset_index()
    if "Date" not in raw.columns and "index" in raw.columns:
        raw = raw.rename(columns={"index": "Date"})
    return raw


def main():
    os.makedirs("data", exist_ok=True)
    print(f"Fetching ^GSPC {cfg.START_DATE}..{cfg.END_DATE} "
          f"(yfinance end exclusive, auto_adjust=True)")
    raw = fetch_raw()
    raw.to_csv(RAW_CSV, index=False)
    print(f"Saved {RAW_CSV}")

    # Re-index on Date for feature engineering
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")

    eng = D.engineer_features(df)

    # Chronological 70/15/15 split labels
    n = len(eng)
    n_train = int(n * cfg.TRAIN_FRAC)
    n_val = int(n * cfg.VAL_FRAC)
    splits = np.array(["train"] * n_train + ["val"] * n_val +
                      ["test"] * (n - n_train - n_val))
    eng = eng.copy()
    eng["split"] = splits

    # MinMax fit on train only
    scaler = MinMaxScaler()
    scaler.fit(eng.loc[eng["split"] == "train", cfg.FEATURE_COLS].values)
    for i, c in enumerate(cfg.FEATURE_COLS):
        lo, hi = scaler.data_min_[i], scaler.data_max_[i]
        eng[c + "_scaled"] = (eng[c].values - lo) / (hi - lo + 1e-12)

    # vol_rank: percentile vs TRAIN vol distribution (raw vol)
    train_vol_sorted = np.sort(
        eng.loc[eng["split"] == "train", cfg.VOL_COL].values)
    eng["vol_rank"] = D._vol_rank(train_vol_sorted, eng[cfg.VOL_COL].values)
    eng["fuzzy_lr"] = [fz.fuzzy_lr_vol_only(v) for v in eng["vol_rank"].values]

    out = eng.reset_index()
    out.to_csv(PROC_CSV, index=False)
    print(f"Saved {PROC_CSV}")

    for name, frame in [("raw", pd.read_csv(RAW_CSV)),
                        ("processed", pd.read_csv(PROC_CSV))]:
        print(f"--- {name} ---")
        print(f"shape={frame.shape} NaNs={int(frame.isna().sum().sum())}")
        print("head:")
        print(frame.head(3).to_string())
        print("tail:")
        print(frame.tail(3).to_string())
        if name == "processed":
            print("per-split rows + up-rate:")
            for s in ["train", "val", "test"]:
                sub = frame[frame["split"] == s]
                print(f"  {s}: n={len(sub)} "
                      f"up_rate={sub['target_direction'].mean():.4f}")


if __name__ == "__main__":
    main()
