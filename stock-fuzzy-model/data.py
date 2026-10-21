"""Data pipeline (PLAN T1 + T2 + T3).

Rules (do NOT change):
  - Chronological split, scaler fit on train only, clip after transform.
  - Window label at end date t never uses information after t+1.
  - TARGET_MODE="direction": label = Close[t+1] > Close[t].
  - TARGET_MODE="vol": label = next-day hl_range > TRAIN-set median.
  - UNIVERSE_MODE="pooled": one global set of date cut points for all tickers.
"""
import os
import re

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

import config as cfg

# ------------------------------------------------------------------ cache
CACHE_DIR = os.path.join("data", "cache")
VIX_CACHE = os.path.join("data", "raw_VIX.csv")


def _cache_path(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker).strip("_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def _cache_ok(path: str, min_rows: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        tail = pd.read_csv(path, usecols=["Date"]).tail(1)
        if tail.empty:
            return False
        last = pd.to_datetime(tail["Date"].iloc[0])
        # yfinance END_DATE is exclusive; allow a short reporting lag.
        fresh = last >= pd.Timestamp(cfg.END_DATE) - pd.Timedelta(days=10)
        with open(path, encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        return fresh and n >= min_rows
    except Exception:  # noqa: BLE001 -- corrupt cache -> redownload
        return False


# ------------------------------------------------------------------ vix
def fetch_vix(start=cfg.START_DATE, end=cfg.END_DATE):
    """Download ^VIX close once, cache it, and return Date/Close frame."""
    if os.path.exists(VIX_CACHE):
        return pd.read_csv(VIX_CACHE, parse_dates=["Date"])
    os.makedirs("data", exist_ok=True)
    raw = yf.download(cfg.VIX_TICKER, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw[["Close"]].reset_index().rename(columns={"Close": "vix_level"})
    out.to_csv(VIX_CACHE, index=False)
    return out


def attach_vix(df: pd.DataFrame) -> pd.DataFrame:
    """Attach same-day VIX close to an OHLCV frame (no future information)."""
    vix = fetch_vix().set_index("Date")["vix_level"].sort_index()
    vix.index = pd.to_datetime(vix.index)
    out = df.copy()
    out["vix_level"] = vix.reindex(out.index).ffill()
    return out


# ---------------------------------------------------------------- download
def _download_chunk(chunk, start, end):
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
            df = attach_vix(df)
            if len(df) > (cfg.WINDOW + 30):
                df.to_csv(_cache_path(t), index_label="Date")
            else:
                print(f"  skipping {t}: insufficient history ({len(df)} rows)")
        except Exception as e:  # noqa: BLE001 -- per-ticker skip, report only
            print(f"  skipping {t}: {e}")


def batch_download(tickers, start=cfg.START_DATE, end=cfg.END_DATE, chunk_size=20):
    """Load OHLCV per ticker from cache, downloading only missing symbols.

    Returns {ticker: DataFrame} indexed by Date with OHLCV + VIX columns.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    min_rows = cfg.WINDOW + 30
    missing = [t for t in tickers if not _cache_ok(_cache_path(t), min_rows)]
    if missing:
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i:i + chunk_size]
            print(f"Downloading {i + 1}-{i + len(chunk)} / {len(missing)} missing ...")
            _download_chunk(chunk, start, end)

    frames = {}
    for t in tickers:
        p = _cache_path(t)
        if not os.path.exists(p):
            print(f"  skipping {t}: no cached data")
            continue
        df = pd.read_csv(p, index_col="Date", parse_dates=True).sort_index()
        if len(df) > min_rows:
            frames[t] = df
        else:
            print(f"  skipping {t}: insufficient history ({len(df)} rows)")
    return frames


# --------------------------------------------------------------- features
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary features only (no raw OHLC). Keeps Close for label checks.

    Adds both target columns so TARGET_MODE can be switched without re-download:
      target_direction : Close[t+1] > Close[t]
      target_vol_raw   : next-day hl_range (or |return|) before the train median
    """
    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_ret"] = (out["Close"] - out["Open"]) / out["Open"]
    out["vol_chg"] = np.log(out["Volume"].replace(0, np.nan)).diff()
    out["volatility_14d"] = out["return_1d"].rolling(14).std()
    if "vix_level" in out.columns:
        vix = out["vix_level"]
        out["vix_z20"] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()

    out["target_direction"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out["next_hl_range"] = out["hl_range"].shift(-1)
    out["next_abs_return"] = out["return_1d"].abs().shift(-1)
    source = "next_hl_range" if cfg.VOL_TARGET_SOURCE == "hl_range" else "next_abs_return"
    out[cfg.VOL_TARGET_RAW_COL] = out[source]

    out = out.replace([np.inf, -np.inf], np.nan)
    missing = [c for c in cfg.FEATURE_COLS if c not in out.columns]
    if missing:
        raise KeyError(f"missing engineered features: {missing}; attach VIX first")
    out = out.dropna(subset=cfg.FEATURE_COLS).copy()
    out = out.iloc[:-1]  # last row's next-day target is undefined
    if cfg.TARGET_MODE == "vol":
        out = out.dropna(subset=[cfg.VOL_TARGET_RAW_COL]).copy()
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


def _window_targets(part: pd.DataFrame, vol_median) -> np.ndarray:
    if cfg.TARGET_MODE == "direction":
        return part[cfg.DIRECTION_TARGET_COL].values.astype(int)
    if vol_median is None or not np.isfinite(vol_median):
        raise ValueError("vol target requires a finite train-set median")
    return (part[cfg.VOL_TARGET_RAW_COL].values > vol_median).astype(int)


def _label_spotcheck(label_df: pd.DataFrame, full_df: pd.DataFrame,
                     vol_median, n=3):
    """T2: verify labels against the raw next-day quantity (no lookahead)."""
    if len(label_df) == 0:
        return
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(label_df), size=min(n, len(label_df)),
                            replace=False)
    closes = full_df["Close"].values
    highs = full_df["High"].values
    lows = full_df["Low"].values
    for i in sample_idx:
        end_date = label_df.index[i]
        gi = full_df.index.get_loc(end_date)
        if gi + 1 >= len(full_df):
            raise AssertionError(f"no next day for {end_date}")
        if cfg.TARGET_MODE == "direction":
            expected = int(closes[gi + 1] > closes[gi])
        else:
            if cfg.VOL_TARGET_SOURCE == "hl_range":
                actual_raw = (highs[gi + 1] - lows[gi + 1]) / closes[gi + 1]
            else:
                actual_raw = abs(closes[gi + 1] / closes[gi] - 1.0)
            stored_raw = float(full_df[cfg.VOL_TARGET_RAW_COL].iloc[gi])
            assert np.isclose(stored_raw, actual_raw, rtol=1e-9, atol=1e-12), (
                f"vol target misaligned at {end_date}: stored={stored_raw} "
                f"actual next-day={actual_raw}")
            expected = int(actual_raw > vol_median)
        got = int(label_df[cfg.TARGET_COL].iloc[i])
        assert got == expected, f"label mismatch at {end_date}: got={got} expected={expected}"


def _windows_from_part(part: pd.DataFrame, scaler, vol_base, window):
    if len(part) <= window:
        return (np.empty((0, window, len(cfg.FEATURE_COLS))),
                np.empty((0,)),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,)))
    scaled = np.clip(scaler.transform(part[cfg.FEATURE_COLS].values),
                     -cfg.CLIP, cfg.CLIP)
    X, y, ends = make_windows(
        scaled, part[cfg.TARGET_COL].values,
        part.index.values, window)
    # vol_rank from RAW vol of the window's last day, ranked vs TRAIN only
    raw_vol_last = part[cfg.VOL_COL].values[window - 1:]
    vr = _vol_rank(vol_base, raw_vol_last[:len(X)])
    return X, y, ends, vr


def split_scale_window(df: pd.DataFrame, window=cfg.WINDOW,
                       train_frac=cfg.TRAIN_FRAC, val_frac=cfg.VAL_FRAC):
    """Chronological per-ticker split; scaler/median/vol_rank calibrated on train only."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # GATE: chronological order per ticker
    assert train_df.index.max() < val_df.index.min(), "train/val overlap"
    assert val_df.index.max() < test_df.index.min(), "val/test overlap"

    scaler = StandardScaler()
    scaler.fit(train_df[cfg.FEATURE_COLS].values)

    train_vol_sorted = np.sort(train_df[cfg.VOL_COL].values)  # raw vol, train only
    vol_median = (float(np.nanmedian(train_df[cfg.VOL_TARGET_RAW_COL].values))
                  if cfg.TARGET_MODE == "vol" else None)

    labeled = {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        frame = part.copy()
        frame[cfg.TARGET_COL] = _window_targets(part, vol_median)
        labeled[name] = frame

    # GATE (T2): label spot-check on 3 train windows
    _label_spotcheck(labeled["train"], df, vol_median)

    out = {name: _windows_from_part(frame, scaler, train_vol_sorted, window)
           for name, frame in labeled.items()}
    return out, scaler, train_vol_sorted, vol_median


def _global_cut_dates(engineered: dict):
    """One chronological set of cut dates shared by every ticker."""
    all_dates = np.sort(np.unique(np.concatenate(
        [df.index.values for df in engineered.values()])))
    n = len(all_dates)
    tr_i = max(1, int(n * cfg.TRAIN_FRAC)) - 1
    va_i = max(tr_i + 1, int(n * (cfg.TRAIN_FRAC + cfg.VAL_FRAC))) - 1
    va_i = min(va_i, n - 2)
    return pd.Timestamp(all_dates[tr_i]), pd.Timestamp(all_dates[va_i])


def _build_pooled(engineered: dict, window):
    train_cut, val_cut = _global_cut_dates(engineered)
    print(f"pooled date cuts: train<= {train_cut.date()} | "
          f"val<= {val_cut.date()} | test > {val_cut.date()}")

    pieces = []
    for t, df in engineered.items():
        tr = df.loc[df.index <= train_cut]
        va = df.loc[(df.index > train_cut) & (df.index <= val_cut)]
        te = df.loc[df.index > val_cut]
        if len(tr) <= window or len(te) <= window:
            print(f"  skipping {t}: insufficient split windows "
                  f"(train={len(tr)}, test={len(te)})")
            continue
        assert tr.index.max() < va.index.min(), f"{t}: train/val overlap"
        assert va.index.max() < te.index.min(), f"{t}: val/test overlap"
        pieces.append((t, tr, va, te))

    if not pieces:
        raise RuntimeError("no ticker produced usable pooled windows")

    train_feats = np.concatenate([p[1][cfg.FEATURE_COLS].values for p in pieces])
    scaler = StandardScaler()
    scaler.fit(train_feats)
    vol_base = np.sort(np.concatenate([p[1][cfg.VOL_COL].values for p in pieces]))
    vol_median = (float(np.nanmedian(np.concatenate(
        [p[1][cfg.VOL_TARGET_RAW_COL].values for p in pieces])))
        if cfg.TARGET_MODE == "vol" else None)

    parts = {"train": [], "val": [], "test": []}
    kept = []
    for t, tr, va, te in pieces:
        labeled = {}
        for name, part in [("train", tr), ("val", va), ("test", te)]:
            frame = part.copy()
            frame[cfg.TARGET_COL] = _window_targets(part, vol_median)
            labeled[name] = frame
        # T2 spot-check once, on the first usable ticker
        if not kept:
            _label_spotcheck(labeled["train"], engineered[t], vol_median)
        windows = {name: _windows_from_part(frame, scaler, vol_base, window)
                   for name, frame in labeled.items()}
        if len(windows["train"][0]) == 0 or len(windows["test"][0]) == 0:
            continue
        kept.append(t)
        for name, bucket in parts.items():
            bucket.append(windows[name])

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

    print(f"Tickers contributing usable windows: {len(kept)}")
    for name in ["train", "val", "test"]:
        y = data[f"y_{name}"]
        pos = y.mean()
        maj = max(pos, 1 - pos)
        print(f"  {name}: pos_rate={pos:.3f} majority_baseline={maj:.3f}")

    assert not np.isnan(data["X_train"]).any(), "NaN in X_train"
    assert set(np.unique(data["y_train"])) <= {0, 1}, "non-binary labels"
    assert len(data["X_train"]) > 0, "no pooled train windows"
    return data, kept


def _build_per_ticker(engineered: dict, window):
    """Per-ticker split/scale, then concatenate. Keeps the ^GSPC mode available."""
    parts = {"train": [], "val": [], "test": []}
    kept = []
    for t, df in engineered.items():
        s, _, _, _ = split_scale_window(df, window=window)
        if len(s["train"][0]) == 0 or len(s["test"][0]) == 0:
            continue
        kept.append(t)
        for name, bucket in parts.items():
            bucket.append(s[name])
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
    assert len(data["X_train"]) > 1500, (
        f"n_train={len(data['X_train'])} <= 1500; check universe/dates")
    return data, kept


def build_dataset(engineered: dict, window=cfg.WINDOW):
    """Split/scale/window, then concatenate. Mode chosen by UNIVERSE_MODE."""
    if cfg.UNIVERSE_MODE == "pooled":
        return _build_pooled(engineered, window)
    if cfg.UNIVERSE_MODE == "gspc":
        return _build_per_ticker(engineered, window)
    raise ValueError(f"unknown UNIVERSE_MODE: {cfg.UNIVERSE_MODE!r}")
