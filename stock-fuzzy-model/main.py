"""End-to-end pipeline: data -> features -> fuzzy-LSTM vs baseline -> evaluation.

Run:
    python main.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from data_loader import batch_download
from evaluation import benchmark, plot_histories
from features import build_all_splits, calibrate_volatility_percentiles, engineer_features
from fuzzy_controller import build_fuzzy_system, fuzzy_lr
from model import DirectionLSTM
from train import train_model


def set_seeds():
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)


def build_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    # Shuffle only training windows (rows are independent 60-day windows;
    # shuffling across tickers/time improves batch diversity for SGD).
    rng = np.random.default_rng(cfg.SEED)
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def main():
    set_seeds()
    print("Using device:", cfg.DEVICE)
    print(f"Universe size: {len(cfg.TICKERS)} tickers "
          f"({len(cfg.US_TICKERS)} US + {len(cfg.INDIA_TICKERS)} India)")

    # Phase 1 — data
    raw_data = batch_download(cfg.TICKERS, cfg.START_DATE, cfg.END_DATE)
    print(f"\nSuccessfully downloaded {len(raw_data)} / {len(cfg.TICKERS)} tickers.")

    engineered = {}
    for t, df in raw_data.items():
        feat_df = engineer_features(df)
        if len(feat_df) > (cfg.WINDOW + 20):
            engineered[t] = feat_df
    print(f"Tickers with usable engineered features: {len(engineered)}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = build_all_splits(engineered)

    # Calibrate fuzzy thresholds to real data distribution
    vol_p33, vol_p50, vol_p66, _ = calibrate_volatility_percentiles(X_train, cfg.VOL_COL_IDX)

    train_loader, val_loader, test_loader = build_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # Phase 2 — fuzzy controller
    lr_ctrl_system = build_fuzzy_system(vol_p33, vol_p50, vol_p66)
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"volatility={v:.2f}  ->  fuzzy_lr={fuzzy_lr(v, lr_ctrl_system):.5f}")

    # Phase 3+4 — train fuzzy model
    n_features = len(cfg.FEATURE_COLS)
    fuzzy_model = DirectionLSTM(n_features)
    fuzzy_model, fuzzy_history = train_model(
        fuzzy_model, train_loader, val_loader,
        epochs=cfg.EPOCHS, use_fuzzy=True, vol_col_idx=cfg.VOL_COL_IDX,
        control_system=lr_ctrl_system,
    )

    # Phase 5 (part 1) — fixed-LR baseline
    baseline_model = DirectionLSTM(n_features)
    baseline_model, baseline_history = train_model(
        baseline_model, train_loader, val_loader,
        epochs=cfg.EPOCHS, use_fuzzy=False, fixed_lr=cfg.FIXED_LR,
    )

    # Phase 5 — evaluation
    plot_histories(fuzzy_history, baseline_history)
    benchmark(fuzzy_model, baseline_model, test_loader, fuzzy_history, baseline_history)


if __name__ == "__main__":
    main()
