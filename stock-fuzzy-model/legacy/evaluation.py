"""Evaluation & benchmarking: loss curves, LR schedule, test accuracy."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIXED_LR
from train import directional_accuracy


def plot_histories(fuzzy_history, baseline_history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(fuzzy_history["train_loss"], label="Fuzzy-LSTM (train)")
    axes[0].plot(baseline_history["train_loss"], label="Baseline-LSTM (train)")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()

    axes[1].plot(fuzzy_history["val_loss"], label="Fuzzy-LSTM (val)")
    axes[1].plot(baseline_history["val_loss"], label="Baseline-LSTM (val)")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE Loss")
    axes[1].legend()

    axes[2].plot(fuzzy_history["lr"], label="Fuzzy-LSTM (avg lr/epoch)")
    axes[2].axhline(FIXED_LR, linestyle="--", color="gray", label="Baseline fixed lr")
    axes[2].set_title("Learning Rate Over Training")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def benchmark(fuzzy_model, baseline_model, test_loader, fuzzy_history, baseline_history):
    fuzzy_test_acc = directional_accuracy(fuzzy_model, test_loader)
    baseline_test_acc = directional_accuracy(baseline_model, test_loader)

    print(f"Out-of-sample directional accuracy — Fuzzy-LSTM:    {fuzzy_test_acc:.4f}")
    print(f"Out-of-sample directional accuracy — Baseline-LSTM: {baseline_test_acc:.4f}")

    results = pd.DataFrame({
        "Model": ["Fuzzy-LSTM", "Baseline-LSTM"],
        "Final Train Loss": [fuzzy_history["train_loss"][-1], baseline_history["train_loss"][-1]],
        "Final Val Loss": [fuzzy_history["val_loss"][-1], baseline_history["val_loss"][-1]],
        "Val Loss Std (stability)": [np.std(fuzzy_history["val_loss"]), np.std(baseline_history["val_loss"])],
        "Test Directional Accuracy": [fuzzy_test_acc, baseline_test_acc],
    })
    print(results.to_string(index=False))
    return results
