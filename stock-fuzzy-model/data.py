"""Data pipeline (PLAN T1 + T2 + T3).

Rules (do NOT change):
  - Chronological split, scaler fit on train only.
  - Label = Close[t+1] > Close[t] for window end t.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import config as cfg


# ---------------------------------------------------------------- download
def batch_download(tickers, start=cfg.START_DATE, end=cfg.END_DATE, chunk_size=20):
    """Download OHLCV per ticker. Returns {ticker: DataFrame}."""
    frames = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"Downloading {i + 1}-{i + len(chunk)} / {len(tickers)} ...")
        raw = yf.download(
            tickers=chunk, start=start, end=end, group_by="ticker",
            auto_adjust=True, threads=True, progress=False,
        )
        for t in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw[t].copy()
                else:
                    df = raw.copy()
                # flatten any residual MultiIndex (e.g. ('^GSPC','Open') pairs)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
                if len(df) > (cfg.WINDOW + 30):
                    frames[t] = df
                else:
                    print(f"  skipping {t}: insufficient history ({len(df)} rows)")
            except Exception as e:  # noqa: BLE001 -- per-ticker skip, report only
                print(f"  skipping {t}: {e}")
    return frames


# --------------------------------------------------------------- features
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary features only (no raw OHLC). Keeps Close for label checks."""
    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_ret"] = (out["Close"] - out["Open"]) / out["Open"]
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()
    out["volatility_14d"] = out["return_1d"].rolling(14).std()
    out["target_direction"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(
        subset=["return_1d", "hl_range", "oc_ret", "vol_chg", "volatility_14d"]
    ).copy()
    out = out.iloc[:-1]  # last row's target undefined (no "tomorrow")
    return out


def make_windows(feature_arr, target_arr, dates, window):
    """Slice windows. y[k] aligns with window end t; end_dates[k] = date of t."""
    X, y, ends = [], [], []
    for i in range(window, len(feature_arr)):
        X.append(feature_arr[i - window:i])
        y.append(target_arr[i - 1])
        ends.append(dates[i - 1])
    if not X:
        empty_x = np.empty((0, window, feature_arr.shape[1]))
        return empty_x, np.empty((0,)), np.empty((0,), dtype="datetime64[ns]")
    return np.array(X), np.array(y), np.array(ends, dtype="datetime64[ns]")


def _vol_rank(searchsorted_base: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Percentile rank (0-1) of values within TRAIN distribution only."""
    idx = np.searchsorted(searchsorted_base, values, side="right")
    return idx / max(len(searchsorted_base), 1)


def split_scale_window(df: pd.DataFrame, window=cfg.WINDOW,
                       train_frac=cfg.TRAIN_FRAC, val_frac=cfg.VAL_FRAC):
    """Chronological split per ticker; scaler + vol_rank calibrated on train only."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # GATE: chronological order per ticker
    assert train_df.index.max() < val_df.index.min(), "train/val overlap"
    assert val_df.index.max() < test_df.index.min(), "val/test overlap"

    scaler = MinMaxScaler()
    scaler.fit(train_df[cfg.FEATURE_COLS].values)

    train_vol_sorted = np.sort(train_df[cfg.VOL_COL].values)  # raw vol, train only

    # GATE: spot-check 3 windows that y == (Close[end+1] > Close[end])
    closes = df["Close"].values
    didx = {d: i for i, d in enumerate(df.index)}
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(train_df), size=min(3, len(train_df)), replace=False)
    for i in sample_idx:
        end_date = train_df.index[i]
        gi = didx[end_date]
        expected = int(closes[gi + 1] > closes[gi])
        got = int(train_df["target_direction"].iloc[i])
        assert got == expected, f"label mismatch at {end_date}"

    out = {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(part) <= window:
            out[name] = (np.empty((0, window, len(cfg.FEATURE_COLS))),
                         np.empty((0,)),
                         np.empty((0,), dtype="datetime64[ns]"),
                         np.empty((0,)))
            continue
        scaled = scaler.transform(part[cfg.FEATURE_COLS].values)
        X, y, ends = make_windows(
            scaled, part[cfg.TARGET_COL].values,
            part.index.values, window)
        # vol_rank from RAW vol of window last-day, ranked vs train distribution
        raw_vol_last = part[cfg.VOL_COL].values[window - 1:]
        if name == "train":
            raw_vol_last = raw_vol_last[:len(X)]
        vr = _vol_rank(train_vol_sorted, raw_vol_last[:len(X)])
        out[name] = (X, y, ends, vr)
    return out, scaler, train_vol_sorted


def build_dataset(engineered: dict, window=cfg.WINDOW):
    """Per-ticker split/scale, then concatenate. Prints balance + majority baseline."""
    parts = {"train": [], "val": [], "test": []}
    kept = []
    for t, df in engineered.items():
        s, _, _ = split_scale_window(df, window=window)
        if len(s["train"][0]) == 0 or len(s["test"][0]) == 0:
            continue
        kept.append(t)
        for name in parts:
            parts[name].append(s[name])
    print(f"Tickers contributing usable windows: {len(kept)}")

    data = {}
    for name in ["train", "val", "test"]:
        X = np.concatenate([p[0] for p in parts[name]], axis=0)
        y = np.concatenate([p[1] for p in parts[name]], axis=0)
        ends = np.concatenate([p[2] for p in parts[name]], axis=0)
        vr = np.concatenate([p[3] for p in parts[name]], axis=0)
        # Regime-aware batching needs chronological order: sort by end_date
        order = np.argsort(ends)
        data[f"X_{name}"], data[f"y_{name}"] = X[order], y[order]
        data[f"end_{name}"], data[f"vr_{name}"] = ends[order], vr[order]
        print(f"{name}: X={data[f'X_{name}'].shape} y={data[f'y_{name}'].shape}")

    for name in ["train", "val", "test"]:
        y = data[f"y_{name}"]
        pos = y.mean()
        maj = max(pos, 1 - pos)
        print(f"  {name}: pos_rate={pos:.3f} majority_baseline={maj:.3f}")

    # GATE (T3): single-ticker ^GSPC must have >1500 train windows, no NaN
    assert not np.isnan(data["X_train"]).any(), "NaN in X_train"
    if not cfg.EXTENSION:
        assert len(data["X_train"]) > 1500, (
            f"n_train={len(data['X_train'])} <= 1500; check universe/dates")
    return data, kept
