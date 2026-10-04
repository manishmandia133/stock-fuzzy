"""Hybrid training loop: fuzzy-adapted LR vs fixed-LR baseline."""
import numpy as np
import torch
import torch.nn as nn

from config import DEVICE
from fuzzy_controller import fuzzy_lr, new_fuzzy_simulator


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * len(yb)
            correct += ((preds > 0.5).float() == yb).sum().item()
            n += len(yb)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_model(model, train_loader, val_loader, epochs=15,
                use_fuzzy=True, fixed_lr=1e-3, vol_col_idx=0,
                grad_clip_norm=1.0, control_system=None):
    """Train with per-batch fuzzy LR (use_fuzzy=True) or fixed LR baseline."""
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=fixed_lr)
    sim = new_fuzzy_simulator(control_system) if (use_fuzzy and control_system is not None) else None

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        epoch_lrs = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if use_fuzzy:
                if control_system is None:
                    raise ValueError("control_system is required when use_fuzzy=True")
                # last day per sample, averaged across the batch
                batch_vol = xb[:, -1, vol_col_idx].mean().item()
                lr = fuzzy_lr(batch_vol, control_system, sim=sim)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                epoch_lrs.append(lr)
            else:
                epoch_lrs.append(fixed_lr)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            running_loss += loss.item() * len(yb)
            n_seen += len(yb)

        train_loss = running_loss / max(n_seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(float(np.mean(epoch_lrs)))

        tag = "Fuzzy-LSTM" if use_fuzzy else "Baseline-LSTM"
        print(f"[{tag}] epoch {epoch:02d}/{epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | avg_lr={history['lr'][-1]:.5f}")

    return model, history


def directional_accuracy(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = (model(xb) > 0.5).float()
            correct += (preds == yb).sum().item()
            n += len(yb)
    return correct / max(n, 1)
