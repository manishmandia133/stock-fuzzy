"""Fetch data and build inspection CSVs. No torch import.

Pooled mode downloads/caches the 100-company notebook universe (60 US + 40
India; ~99 usable symbols). The ^GSPC processed CSV is always written for the
price/volatility figure. Training data itself is rebuilt by main.py from the
same cache with the chronological train-only split/scaler rules.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

import config as cfg
import data as D
import fuzzy as fz

RAW_CSV = os.path.join("data", "raw_GSPC.csv")
PROC_CSV = os.path.join("data", "processed_GSPC.csv")
RAW_10D_CSV = os.path.join("data", "raw_GSPC_10d.csv")
PROC_10D_CSV = os.path.join("data", "processed_GSPC_10d.csv")


def fetch_raw():
    raw = yf.download("^GSPC", start=cfg.START_DATE, end=cfg.END_DATE,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if c[0] in
                       ("Open", "High", "Low", "Close", "Volume") else "_".join(map(str, c))
                       for c in raw.columns]
    raw = raw.reset_index()
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


def prepare_pooled():
    """Download/cache the pooled universe; main.py builds the windows."""
    print(f"Pooled universe: {len(cfg.POOLED_TICKERS)} requested symbols "
          f"({len(cfg.US_TICKERS)} US + {len(cfg.INDIA_TICKERS)} India)")
    frames = D.batch_download(cfg.POOLED_TICKERS, cfg.START_DATE, cfg.END_DATE)
    print(f"Cached {len(frames)}/{len(cfg.POOLED_TICKERS)} symbols")
    usable = 0
    for df in frames.values():
        eng = D.engineer_features(df)
        if len(eng) > cfg.WINDOW + 20:
            usable += 1
    print(f"Symbols with usable engineered features: {usable}")
    print(f"Full feature count: {len(cfg.FULL_FEATURES)}")


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
    df = D.attach_market_features(df)

    eng = D.engineer_features(df)
    D.assert_feature_contract(cfg.FULL_FEATURES)

    n = len(eng)
    n_train = int(n * cfg.TRAIN_FRAC)
    n_val = int(n * cfg.VAL_FRAC)
    splits = np.array(["train"] * n_train + ["val"] * n_val +
                      ["test"] * (n - n_train - n_val))
    eng = eng.copy()
    eng["split"] = splits

    scaler = StandardScaler()
    scaler.fit(eng.loc[eng["split"] == "train", cfg.FULL_FEATURES].values)
    scaled = np.clip(
        scaler.transform(eng[cfg.FULL_FEATURES].values), -cfg.CLIP, cfg.CLIP)
    for i, c in enumerate(cfg.FULL_FEATURES):
        eng[c + "_scaled"] = scaled[:, i]

    train_vol_sorted = np.sort(
        eng.loc[eng["split"] == "train", cfg.VOL_COL].values)
    eng["vol_rank"] = D._vol_rank(train_vol_sorted, eng[cfg.VOL_COL].values)
    eng["fuzzy_lr"] = [fz.fuzzy_vol_lr(v) for v in eng["vol_rank"].values]

    out = eng.reset_index()
    out.to_csv(PROC_CSV, index=False)
    print(f"Saved {PROC_CSV}")
    print(f"Feature count: {len(cfg.FULL_FEATURES)} -> {cfg.FULL_FEATURES}")

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
            print(f"per-split rows + positive rate for `{cfg.TARGET_COL}`:")
            for s in ["train", "val", "test"]:
                sub = frame[frame["split"] == s]
                print(f"  {s}: n={len(sub)} "
                      f"pos_rate={sub[cfg.TARGET_COL].mean():.4f}")


def main():
    os.makedirs("data", exist_ok=True)
    if cfg.UNIVERSE_MODE == "pooled":
        prepare_pooled()
    prepare_gspc_processed()


if __name__ == "__main__":
    main()
