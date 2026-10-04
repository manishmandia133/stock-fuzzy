"""Feature engineering, scaling, and sliding-window sequencing."""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import FEATURE_COLS, TARGET_COL, WINDOW


def engineer_features(df):
    """Add return_1d, volatility_14d and next-day direction target."""
    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["volatility_14d"] = out["return_1d"].rolling(14).std()
    out["target_direction"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out = out.dropna(subset=["return_1d", "volatility_14d"]).copy()
    out = out.iloc[:-1]  # last row's target is undefined (no "tomorrow")
    return out[FEATURE_COLS + [TARGET_COL]]


def make_windows(feature_arr, target_arr, window):
    """Slice (T, F) features + (T,) targets into (N, window, F) windows."""
    X, y = [], []
    for i in range(window, len(feature_arr)):
        X.append(feature_arr[i - window:i])
        y.append(target_arr[i - 1])  # target aligned with last day in window
    if not X:
        return np.empty((0, window, feature_arr.shape[1])), np.empty((0,))
    return np.array(X), np.array(y)


def split_scale_window(df, window=WINDOW, train_frac=0.8, val_frac=0.1):
    """Time-based split per ticker; scaler fit on train only."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLS].values)

    splits = {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(part) <= window:
            splits[name] = (np.empty((0, window, len(FEATURE_COLS))), np.empty((0,)))
            continue
        scaled = scaler.transform(part[FEATURE_COLS].values)
        X, y = make_windows(scaled, part[TARGET_COL].values, window)
        splits[name] = (X, y)
    return splits


def build_all_splits(engineered, window=WINDOW):
    """Run split_scale_window per ticker and concatenate across tickers."""
    all_splits = {"train": [], "val": [], "test": []}
    kept_tickers = []
    for t, df in engineered.items():
        s = split_scale_window(df, window=window)
        if len(s["train"][0]) == 0 or len(s["test"][0]) == 0:
            continue
        kept_tickers.append(t)
        for name in ["train", "val", "test"]:
            X, y = s[name]
            if len(X):
                all_splits[name].append((X, y))

    print(f"Tickers contributing usable windows: {len(kept_tickers)}")

    X_train = np.concatenate([x for x, _ in all_splits["train"]], axis=0)
    y_train = np.concatenate([y for _, y in all_splits["train"]], axis=0)
    X_val = np.concatenate([x for x, _ in all_splits["val"]], axis=0)
    y_val = np.concatenate([y for _, y in all_splits["val"]], axis=0)
    X_test = np.concatenate([x for x, _ in all_splits["test"]], axis=0)
    y_test = np.concatenate([y for _, y in all_splits["test"]], axis=0)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), kept_tickers


def calibrate_volatility_percentiles(X_train, vol_col_idx):
    """Calibrate fuzzy thresholds to the real (scaled) volatility distribution.

    Uses the LAST day of each window — the same statistic train_model()
    feeds the controller per batch.
    """
    train_vol_last = X_train[:, -1, vol_col_idx]
    p33, p50, p66, p90 = np.percentile(train_vol_last, [33, 50, 66, 90])
    print(f"Volatility percentiles -> p33={p33:.4f}  p50={p50:.4f}  "
          f"p66={p66:.4f}  p90={p90:.4f}")
    return p33, p50, p66, p90
