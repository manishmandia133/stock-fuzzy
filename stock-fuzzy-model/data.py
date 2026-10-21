"""Data pipeline for ^GSPC regression.

Target: log(Close_{t+1} / Close_t) at row t — next-day log return.
Reconstruction: pred_close = Close_t * exp(pred_log_return).

Leakage rules
  • All rolling/EMA features are trailing (look only at data ≤ t).
  • Monthly FRED series (UNRATE, UMCSENT) shifted one month before ffill.
  • StandardScaler fit on TRAIN rows only.
  • Target at t uses Close[t+1]; windows end at t → no future feature leaks.
  • Leakage unit test: perturbing rows after t must not change features at t.
"""
import os
import re

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

import config as cfg

CACHE_DIR = os.path.join("data", "cache")


def _cache(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


# ── Downloads ───────────────────────────────────────────────────────────────

def fetch_gspc(start=cfg.START_DATE, end=cfg.END_DATE) -> pd.DataFrame:
    """Download/cache ^GSPC OHLCV. Returns Date-indexed DataFrame."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache("GSPC")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if len(df) > 200:
            return df
    raw = yf.download(cfg.TICKER, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    df.index.name = "Date"
    df.to_csv(path, index=True)
    return df


def _fetch_price_series(ticker: str, col_name: str,
                        start=cfg.START_DATE, end=cfg.END_DATE) -> pd.Series:
    path = _cache(col_name)
    if os.path.exists(path):
        s = pd.read_csv(path, index_col="Date", parse_dates=True).squeeze()
        if len(s) > 100:
            return s.rename(col_name)
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].rename(col_name)
    s.index.name = "Date"
    s.to_frame().to_csv(path)
    return s


def fetch_fred(series_id: str) -> pd.Series:
    path = _cache(f"FRED_{series_id}")
    if os.path.exists(path):
        try:
            s = pd.read_csv(path, index_col="Date", parse_dates=True).squeeze()
            if len(s) > 10:
                return s.rename(series_id)
        except Exception:
            pass
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = pd.read_csv(url)
    date_col = "observation_date" if "observation_date" in raw.columns else "DATE"
    val_col  = series_id if series_id in raw.columns else raw.columns[-1]
    if date_col not in raw.columns:
        raise ValueError(f"FRED {series_id} returned unexpected format: {raw.columns}")
    df = raw[[date_col, val_col]].copy()
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path)
    return df[series_id]


def attach_macro(df: pd.DataFrame,
                 start=cfg.START_DATE, end=cfg.END_DATE) -> pd.DataFrame:
    """Attach VIX, USDX, EFFR, UNRATE, UMCSENT, T10Y2Y to a Date-indexed DataFrame."""
    out = df.copy()
    idx = out.index

    for ticker, name in [(cfg.VIX_TICKER, "VIX"), (cfg.USDX_TICKER, "USDX")]:
        s = _fetch_price_series(ticker, name, start, end)
        out[name] = s.reindex(idx, method="ffill").ffill()

    for sid in cfg.FRED_SERIES:
        try:
            s = fetch_fred(sid).sort_index()
            if sid in ("UNRATE", "UMCSENT"):
                s = s.shift(1)       # publication-lag guard: use last month's value
            out[sid] = s.reindex(idx, method="ffill").ffill()
        except Exception as exc:
            if sid == "T10Y2Y":
                try:
                    tnx = _fetch_price_series("^TNX", "TNX", start, end)
                    irx = _fetch_price_series("^IRX", "IRX", start, end)
                    spread = (tnx - irx).reindex(idx, method="ffill").ffill()
                    out[sid] = spread
                    print("  Note: FRED T10Y2Y offline; using ^TNX - ^IRX yield spread proxy")
                    continue
                except Exception:
                    pass
            print(f"  WARNING: FRED {sid} failed ({exc}); filling with 0")
            out[sid] = 0.0

    return out


# ── Feature engineering ─────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0.0)
    loss  = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs  = ag / al.where(al > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((al == 0) & (ag > 0), 100.0)
    rsi = rsi.mask((al == 0) & (ag == 0), 50.0)
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr   = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features + target. Fully causal (all look backward only).

    Input df must contain: Open, High, Low, Close, Volume, VIX, USDX,
    EFFR, UNRATE, UMCSENT, T10Y2Y (attach_macro must be called first).

    Returns DataFrame with ALL_FEATURES cols + TARGET_COL + 'Close' (raw).
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    close = out["Close"].astype(float)

    # Core technical (all trailing)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["MACD"] = ema12 - ema26
    out["RSI"]  = _rsi(close)
    out["ATR"]  = _atr(out)

    # Log returns
    log_ret = np.log(close / close.shift(1))
    out["log_ret_1d"]  = log_ret
    out["log_ret_5d"]  = log_ret.rolling(5,  min_periods=5).sum()
    out["log_ret_20d"] = log_ret.rolling(20, min_periods=20).sum()

    # Lagged close (previous day's close — causal)
    out["prev_close"] = close.shift(1)

    # VIX z-score over 20 days
    vix20m = out["VIX"].rolling(20, min_periods=20).mean()
    vix20s = out["VIX"].rolling(20, min_periods=20).std().clip(lower=1e-8)
    out["vix_z20"] = (out["VIX"] - vix20m) / vix20s

    # MA ratios (Close/MA50, MA50/MA200)
    sma50  = close.rolling(50,  min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    out["close_ma50_ratio"]  = close / sma50.where(sma50 > 0)  - 1.0
    out["ma50_ma200_ratio"]  = sma50 / sma200.where(sma200 > 0) - 1.0

    # Volume change
    out["vol_change"] = out["Volume"].pct_change().clip(-5.0, 5.0)

    # Realized volatility (annualised)
    out["realized_vol_20d"] = log_ret.rolling(20, min_periods=20).std() * np.sqrt(252)

    # ── Target: next-day log return (uses only shift(-1) — no look-ahead in features)
    out[cfg.TARGET_COL] = log_ret.shift(-1)

    out = out.replace([np.inf, -np.inf], np.nan)
    needed = cfg.ALL_FEATURES + [cfg.TARGET_COL, "Close"]
    out = out.dropna(subset=[c for c in needed if c in out.columns])
    return out


# ── Leakage unit test ───────────────────────────────────────────────────────

def leakage_unit_test(df_with_macro: pd.DataFrame) -> None:
    """Assert that perturbing rows after t does not change features at t.

    Selects an evaluation date chk_date from feat_orig, finds its raw index
    position t, perturbs strictly rows t+1 onward in raw data, re-computes
    features, and verifies all features at chk_date remain identical.
    Also confirms that the target at chk_date (next-day return) DOES change.
    """
    feat_orig = engineer_features(df_with_macro)
    if len(feat_orig) < 10:
        print("  leakage check: SKIP (insufficient rows)")
        return

    # Choose a check date from the middle of the engineered frame
    check_idx = len(feat_orig) // 2
    chk_date  = feat_orig.index[check_idx]

    # Find the corresponding row in df_with_macro
    raw_loc = df_with_macro.index.get_loc(chk_date)
    perturb_from = raw_loc + 1
    if perturb_from >= len(df_with_macro):
        print("  leakage check: SKIP (chk_date is last row)")
        return

    df_pert = df_with_macro.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df_pert.columns:
            loc = df_pert.columns.get_loc(col)
            df_pert.iloc[perturb_from:, loc] = df_pert.iloc[perturb_from:, loc] * 1.05

    feat_pert = engineer_features(df_pert)

    assert chk_date in feat_pert.index, f"Check date {chk_date} missing in perturbed features"

    for col in cfg.ALL_FEATURES:
        if col not in feat_orig.columns or col not in feat_pert.columns:
            continue
        o = feat_orig.loc[chk_date, col]
        p = feat_pert.loc[chk_date, col]
        if pd.isna(o) and pd.isna(p):
            continue
        if pd.isna(o) or pd.isna(p):
            continue
        diff = abs(float(o) - float(p))
        if diff > 1e-6:
            raise AssertionError(
                f"Leakage detected in feature '{col}' at {chk_date}: "
                f"diff={diff:.2e} (orig={o:.6f}, pert={p:.6f})"
            )

    # Confirm target is NOT in feature_cols
    assert cfg.TARGET_COL not in cfg.ALL_FEATURES, \
        f"Target '{cfg.TARGET_COL}' must not appear in feature_cols!"

    # Verify target DOES reflect the perturbation at t+1
    orig_tgt = feat_orig.loc[chk_date, cfg.TARGET_COL]
    pert_tgt = feat_pert.loc[chk_date, cfg.TARGET_COL]
    assert abs(float(orig_tgt) - float(pert_tgt)) > 1e-5, \
        "Target did not reflect future perturbation at t+1!"

    print("  leakage check: PASS")


# ── Split & window ──────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """Chronological 80/20 split, with val = last 20 % of the 80 % train block.

    Returns (train_df, val_df, test_df).
    """
    n = len(df)
    n_trainval = int(n * cfg.TRAIN_VAL_FRAC)
    n_train    = int(n_trainval * (1.0 - cfg.VAL_FRAC_OF_TRAIN))
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_trainval]
    test_df  = df.iloc[n_trainval:]
    assert train_df.index.max() < val_df.index.min(),  "train/val overlap"
    assert val_df.index.max()   < test_df.index.min(), "val/test overlap"
    return train_df, val_df, test_df


def _make_windows(feat_arr, target_arr, close_arr, dates_arr, window):
    """Sliding windows of length `window`.

    Window k covers rows k .. k+window-1.
    y[k]     = target_arr[k+window-1]   (log-return at end of window)
    close[k] = close_arr[k+window-1]    (Close_t, for price reconstruction)
    date[k]  = dates_arr[k+window-1]
    """
    n = len(feat_arr) - window
    if n <= 0:
        F = feat_arr.shape[1]
        return (np.empty((0, window, F)), np.empty(0),
                np.empty(0), np.empty(0, dtype="datetime64[ns]"))
    X      = np.stack([feat_arr[i:i + window]     for i in range(n)])
    y      = np.array([target_arr[i + window - 1] for i in range(n)])
    closes = np.array([close_arr[i + window - 1]  for i in range(n)])
    dates  = np.array([dates_arr[i + window - 1]  for i in range(n)],
                      dtype="datetime64[ns]")
    return X, y, closes, dates


def _vol_rank(vol_vals: np.ndarray, train_sorted: np.ndarray) -> np.ndarray:
    """Percentile rank (0–1) of vol_vals within train distribution."""
    idx = np.searchsorted(train_sorted, vol_vals, side="right")
    return idx / max(len(train_sorted), 1)


# ── Main pipeline entry point ───────────────────────────────────────────────

def build_dataset(start=cfg.START_DATE, end=cfg.END_DATE,
                  feature_cols=None) -> dict:
    """Full pipeline: download → macro → engineer → split → scale → window.

    Returns dict with keys:
      X_{train,val,test}, y_{train,val,test},
      close_{train,val,test}, vr_{train,val,test}, tr_{train,val,test},
      dates_{train,val,test}, scaler, train_vol_sorted, train_trend_sorted
    """
    feature_cols = list(feature_cols or cfg.ALL_FEATURES)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("── Downloading data ──")
    gspc = fetch_gspc(start, end)
    print(f"  ^GSPC raw rows: {len(gspc)}")

    gspc = attach_macro(gspc, start, end)
    print("  macro attached")

    df = engineer_features(gspc)
    print(f"  after engineering: {len(df)} rows "
          f"({df.index[0].date()} .. {df.index[-1].date()})")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Features missing after engineering: {missing}")

    # Leakage test (on a slice, not full data, to avoid re-downloading)
    leakage_unit_test(gspc)

    # Split
    train_df, val_df, test_df = split_data(df)
    print(f"  train: {len(train_df)} rows  "
          f"({train_df.index[0].date()} .. {train_df.index[-1].date()})")
    print(f"  val:   {len(val_df)} rows  "
          f"({val_df.index[0].date()}  .. {val_df.index[-1].date()})")
    print(f"  test:  {len(test_df)} rows  "
          f"({test_df.index[0].date()} .. {test_df.index[-1].date()})")

    # Scaler fit on train only
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    def scale(part):
        return np.clip(scaler.transform(part[feature_cols].values),
                       -cfg.CLIP, cfg.CLIP)

    # Regime references (vol_rank, trend_rank) from train distribution
    train_vol_sorted   = np.sort(train_df[cfg.VOL_COL].values)
    train_trend_sorted = np.sort(train_df["close_ma50_ratio"].values)

    data = {"scaler": scaler,
            "train_vol_sorted":   train_vol_sorted,
            "train_trend_sorted": train_trend_sorted,
            "feature_cols": feature_cols}

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        feat = scale(part)
        X, y, closes, dates = _make_windows(
            feat, part[cfg.TARGET_COL].values,
            part["Close"].values, part.index.values, cfg.WINDOW)

        # vol_rank and trend_rank at each window's last row
        vol_vals   = part[cfg.VOL_COL].values[cfg.WINDOW - 1:][:len(X)]
        trend_vals = part["close_ma50_ratio"].values[cfg.WINDOW - 1:][:len(X)]
        vr = _vol_rank(vol_vals,   train_vol_sorted)
        tr = _vol_rank(trend_vals, train_trend_sorted)

        data[f"X_{name}"]     = X
        data[f"y_{name}"]     = y
        data[f"close_{name}"] = closes
        data[f"vr_{name}"]    = vr
        data[f"tr_{name}"]    = tr
        data[f"dates_{name}"] = dates
        print(f"  {name}: X={X.shape}  "
              f"y=[{y.min():.4f}, {y.max():.4f}]  "
              f"close=[{closes.min():.1f}, {closes.max():.1f}]")

    # Sanity checks
    assert np.isfinite(data["X_train"]).all(), "NaN/Inf in X_train"
    assert np.isfinite(data["y_train"]).all(), "NaN/Inf in y_train"
    assert len(data["X_train"]) > 200, "Too few training windows"

    return data
