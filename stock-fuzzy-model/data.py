"""Data pipeline (PLAN T1 + T2 + T3).

Rules (do NOT change):
  - Chronological split, scaler fit on train only.
  - Label = Close[t+1] > Close[t] for window end t.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import config as cfg

MIN_HISTORY_ROWS = cfg.WINDOW + 30
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------- download
def _extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract one ticker's OHLCV frame from a possibly multi-index download."""
    if isinstance(raw.columns, pd.MultiIndex):
        frame = raw[ticker].copy()
    else:
        frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    return frame[OHLCV_COLS].dropna(how="all")


def download_raw_frame(
    ticker: str,
    start: str = cfg.START_DATE,
    end: str = cfg.END_DATE,
) -> pd.DataFrame:
    """Download one ticker and return a Date-column raw OHLCV frame."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    # Flatten MultiIndex exactly as returned, e.g. ('Close','^GSPC') -> 'Close'.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            col[0] if col[0] in OHLCV_COLS else "_".join(map(str, col))
            for col in raw.columns
        ]
    raw = raw.reset_index()
    if "Date" not in raw.columns and "index" in raw.columns:
        raw = raw.rename(columns={"index": "Date"})
    return raw


def batch_download(
    tickers: Iterable[str],
    start: str = cfg.START_DATE,
    end: str = cfg.END_DATE,
    chunk_size: int = 20,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV per ticker. Returns {ticker: DataFrame}."""
    tickers = list(tickers)
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"Downloading {i + 1}-{i + len(chunk)} / {len(tickers)} ...")
        raw = yf.download(
            tickers=chunk,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        for ticker in chunk:
            try:
                frame = _extract_ticker_frame(raw, ticker)
                if len(frame) > MIN_HISTORY_ROWS:
                    frames[ticker] = frame
                else:
                    print(
                        f"  skipping {ticker}: insufficient history "
                        f"({len(frame)} rows)"
                    )
            except Exception as exc:  # noqa: BLE001 -- per-ticker skip, report only
                print(f"  skipping {ticker}: {exc}")
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


def make_windows(
    feature_arr: np.ndarray,
    target_arr: np.ndarray,
    dates: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice windows. y[k] aligns with window end t; end_dates[k] = date of t."""
    X, y, ends = [], [], []
    for i in range(window, len(feature_arr)):
        X.append(feature_arr[i - window : i])
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


def _split_bounds(
    n: int,
    train_frac: float = cfg.TRAIN_FRAC,
    val_frac: float = cfg.VAL_FRAC,
) -> tuple[int, int]:
    """Return train/validation row counts for a chronological split."""
    return int(n * train_frac), int(n * val_frac)


def _chronological_split(
    df: pd.DataFrame,
    n_train: int,
    n_val: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by row order and assert there is no split overlap."""
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]
    assert train_df.index.max() < val_df.index.min(), "train/val overlap"
    assert val_df.index.max() < test_df.index.min(), "val/test overlap"
    return train_df, val_df, test_df


def _fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """Fit the feature scaler on training rows only."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[cfg.FEATURE_COLS].values)
    return scaler


def _check_labels(df: pd.DataFrame, train_df: pd.DataFrame) -> None:
    """Spot-check 3 windows that y == (Close[end+1] > Close[end])."""
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


def _empty_windows(
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return empty arrays with the same shapes as a usable split."""
    return (
        np.empty((0, window, len(cfg.FEATURE_COLS))),
        np.empty((0,)),
        np.empty((0,), dtype="datetime64[ns]"),
        np.empty((0,)),
    )


def _split_windows(
    part: pd.DataFrame,
    scaler: MinMaxScaler,
    train_vol_sorted: np.ndarray,
    name: str,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale one split and build its windows and train-calibrated vol ranks."""
    if len(part) <= window:
        return _empty_windows(window)
    scaled = scaler.transform(part[cfg.FEATURE_COLS].values)
    X, y, ends = make_windows(
        scaled,
        part[cfg.TARGET_COL].values,
        part.index.values,
        window,
    )
    # vol_rank from RAW vol of window last-day, ranked vs train distribution.
    raw_vol_last = part[cfg.VOL_COL].values[window - 1 :]
    if name == "train":
        raw_vol_last = raw_vol_last[: len(X)]
    vr = _vol_rank(train_vol_sorted, raw_vol_last[: len(X)])
    return X, y, ends, vr


def split_scale_window(
    df: pd.DataFrame,
    window: int = cfg.WINDOW,
    train_frac: float = cfg.TRAIN_FRAC,
    val_frac: float = cfg.VAL_FRAC,
) -> tuple[dict[str, tuple[np.ndarray, ...]], MinMaxScaler, np.ndarray]:
    """Chronological split per ticker; scaler + vol_rank calibrated on train only."""
    n_train, n_val = _split_bounds(len(df), train_frac, val_frac)
    train_df, val_df, test_df = _chronological_split(df, n_train, n_val)
    scaler = _fit_scaler(train_df)
    train_vol_sorted = np.sort(train_df[cfg.VOL_COL].values)  # raw vol, train only
    _check_labels(df, train_df)

    out = {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out[name] = _split_windows(part, scaler, train_vol_sorted, name, window)
    return out, scaler, train_vol_sorted


def build_dataset(
    engineered: dict[str, pd.DataFrame],
    window: int = cfg.WINDOW,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Per-ticker split/scale, then concatenate. Prints balance + majority baseline."""
    parts: dict[str, list[tuple[np.ndarray, ...]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    kept = []
    for ticker, df in engineered.items():
        split_data, _, _ = split_scale_window(df, window=window)
        if len(split_data["train"][0]) == 0 or len(split_data["test"][0]) == 0:
            continue
        kept.append(ticker)
        for name, chunks in parts.items():
            chunks.append(split_data[name])
    print(f"Tickers contributing usable windows: {len(kept)}")

    data: dict[str, np.ndarray] = {}
    for name in ["train", "val", "test"]:
        X = np.concatenate([p[0] for p in parts[name]], axis=0)
        y = np.concatenate([p[1] for p in parts[name]], axis=0)
        ends = np.concatenate([p[2] for p in parts[name]], axis=0)
        vr = np.concatenate([p[3] for p in parts[name]], axis=0)
        # Regime-aware batching needs chronological order: sort by end_date.
        order = np.argsort(ends)
        data[f"X_{name}"], data[f"y_{name}"] = X[order], y[order]
        data[f"end_{name}"], data[f"vr_{name}"] = ends[order], vr[order]
        print(f"{name}: X={data[f'X_{name}'].shape} y={data[f'y_{name}'].shape}")

    for name in ["train", "val", "test"]:
        y = data[f"y_{name}"]
        pos = y.mean()
        maj = max(pos, 1 - pos)
        print(f"  {name}: pos_rate={pos:.3f} majority_baseline={maj:.3f}")

    # GATE (T3): single-ticker ^GSPC must have >1500 train windows, no NaN.
    assert not np.isnan(data["X_train"]).any(), "NaN in X_train"
    if not cfg.EXTENSION:
        assert (
            len(data["X_train"]) > 1500
        ), f"n_train={len(data['X_train'])} <= 1500; check universe/dates"
    return data, kept


def build_processed_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, MinMaxScaler, np.ndarray]:
    """Add split labels, train-only scaled features, and train-calibrated vol_rank."""
    n_train, n_val = _split_bounds(len(df))
    out = df.copy()
    out["split"] = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * (len(out) - n_train - n_val)
    )
    scaler = _fit_scaler(out.loc[out["split"] == "train"])
    for i, col in enumerate(cfg.FEATURE_COLS):
        lo, hi = scaler.data_min_[i], scaler.data_max_[i]
        out[col + "_scaled"] = (out[col].values - lo) / (hi - lo + 1e-12)

    # vol_rank: percentile vs TRAIN vol distribution (raw vol).
    train_vol_sorted = np.sort(out.loc[out["split"] == "train", cfg.VOL_COL].values)
    out["vol_rank"] = _vol_rank(train_vol_sorted, out[cfg.VOL_COL].values)
    return out, scaler, train_vol_sorted
