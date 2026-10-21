"""Regression evaluation: metrics, persistence baseline, WIN/LOSS table.

Primary metric: RMSE (in price units).
All metrics are computed on Close prices (reconstructed from log-return preds).
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

import targets as T


# ── Core metrics ────────────────────────────────────────────────────────────

def reconstruct_close(close_t: np.ndarray, log_ret_pred: np.ndarray) -> np.ndarray:
    """Reconstruct predicted next-day Close from Close_t and predicted log-return."""
    return close_t * np.exp(log_ret_pred)


def regression_metrics(y_true_log: np.ndarray,
                       y_pred_log: np.ndarray,
                       close_t:    np.ndarray) -> dict:
    """Compute regression metrics in price units.

    Args:
        y_true_log: true next-day log-return array
        y_pred_log: predicted log-return array
        close_t:    Close price at end of each window (Close_t)

    Returns dict with MSE, RMSE, MAPE (%), R (Pearson), MAE, dir_acc.
    """
    true_price = reconstruct_close(close_t, y_true_log)
    pred_price = reconstruct_close(close_t, y_pred_log)

    err   = true_price - pred_price
    mse   = float(np.mean(err ** 2))
    rmse  = float(np.sqrt(mse))
    mae   = float(np.mean(np.abs(err)))
    mape  = float(np.mean(np.abs(err / np.where(true_price != 0, true_price, 1e-8))) * 100)

    # Pearson R on price
    if true_price.std() > 0 and pred_price.std() > 0:
        r = float(np.corrcoef(true_price, pred_price)[0, 1])
    else:
        r = float("nan")

    # Directional accuracy (sign of predicted vs true log-return)
    dir_acc = float(np.mean(np.sign(y_pred_log) == np.sign(y_true_log)))

    return {
        "MSE":     mse,
        "RMSE":    rmse,
        "MAPE":    mape,
        "R":       r,
        "MAE":     mae,
        "dir_acc": dir_acc,
    }


# ── Persistence baseline ─────────────────────────────────────────────────────

def persistence_baseline(close_t: np.ndarray, y_true_log: np.ndarray) -> dict:
    """Persistence: predict tomorrow = today (log-return = 0).

    close_t:    Close at t (window end)
    y_true_log: true log-return (log Close_{t+1}/Close_t)
    """
    zeros = np.zeros_like(y_true_log)
    m = regression_metrics(y_true_log, zeros, close_t)
    return {f"persist_{k}": v for k, v in m.items()}


# ── Ensemble predictions ─────────────────────────────────────────────────────

def ensemble_predict(preds_list: list) -> np.ndarray:
    """Average log-return predictions from multiple seeds."""
    return np.mean(np.stack(preds_list, axis=0), axis=0)


# ── WIN / LOSS comparison ────────────────────────────────────────────────────

def win_loss(rmse: float, mape: float, ref_rmse: float, ref_mape: float) -> str:
    """WIN if both RMSE and MAPE beat the reference, else LOSS."""
    if rmse < ref_rmse and mape < ref_mape:
        return "WIN"
    return "LOSS"


# ── Paired statistics ────────────────────────────────────────────────────────

def holm_correction(pvals: list) -> list:
    """Holm step-down correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals, float)
    out = np.full_like(pvals, np.nan)
    idx = np.where(np.isfinite(pvals))[0]
    if len(idx) == 0:
        return out.tolist()
    order = idx[np.argsort(pvals[idx])]
    m, prev = len(order), 0.0
    for rank, i in enumerate(order):
        adj  = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, adj)
        out[i] = prev
    return out.tolist()


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon for H0: mean(a)==mean(b).

    a, b are per-seed RMSE arrays (lower is better).
    Reports mean diff = mean(b) - mean(a): positive means a is better.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": len(a), "mean_diff": float("nan"),
                "p_t": float("nan"), "p_w": float("nan")}
    diff = b - a  # positive → a is better
    try:
        _, p_t = ttest_rel(a, b)
    except Exception:
        p_t = float("nan")
    try:
        _, p_w = wilcoxon(a, b)
    except Exception:
        p_w = float("nan")
    return {"n": int(len(a)), "mean_diff": float(diff.mean()),
            "p_t": float(p_t), "p_w": float(p_w)}


# ── Results table ─────────────────────────────────────────────────────────────

def build_results_table(rows: list, persist_m: dict, ensemble_df: pd.DataFrame = None) -> pd.DataFrame:
    """Aggregate per-seed rows into a summary table.

    Each row dict must contain: variant, seed, RMSE, MAPE, R, MSE, MAE, dir_acc.
    Returns DataFrame with mean±std per variant + WIN/LOSS columns.
    """
    df = pd.DataFrame(rows)
    records = []

    # Persistence row
    records.append({
        "model":    "Persistence",
        "RMSE":     f"{persist_m['persist_RMSE']:.4f}",
        "RMSE_std": "",
        "MAPE%":    f"{persist_m['persist_MAPE']:.4f}",
        "R":        f"{persist_m['persist_R']:.4f}",
        "MSE":      f"{persist_m['persist_MSE']:.2f}",
        "MAE":      f"{persist_m['persist_MAE']:.4f}",
        "dir_acc":  f"{persist_m['persist_dir_acc']:.3f}",
        "vs_paper": win_loss(persist_m["persist_RMSE"], persist_m["persist_MAPE"],
                             T.PAPER_RMSE, T.PAPER_MAPE),
        "vs_persist": "—",
        "n_seeds":  1,
    })

    # Paper row (reference, hardcoded)
    records.append({
        "model":    "Paper (Bhandari 2022)",
        "RMSE":     f"{T.PAPER_RMSE:.4f}",
        "RMSE_std": "",
        "MAPE%":    f"{T.PAPER_MAPE:.4f}",
        "R":        f"{T.PAPER_R:.4f}",
        "MSE":      f"{T.PAPER_MSE:.2f}",
        "MAE":      "—",
        "dir_acc":  "—",
        "vs_paper": "—",
        "vs_persist": win_loss(T.PAPER_RMSE, T.PAPER_MAPE,
                               persist_m["persist_RMSE"], persist_m["persist_MAPE"]),
        "n_seeds":  1,
    })

    # Per-variant rows
    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        rmse_vals = sub["RMSE"].values
        mape_vals = sub["MAPE"].values
        r_vals    = sub["R"].values
        mse_vals  = sub["MSE"].values
        mae_vals  = sub["MAE"].values
        da_vals   = sub["dir_acc"].values

        m_rmse, s_rmse = float(rmse_vals.mean()), float(rmse_vals.std())
        m_mape         = float(mape_vals.mean())

        records.append({
            "model":    variant,
            "RMSE":     f"{m_rmse:.4f}",
            "RMSE_std": f"{s_rmse:.4f}",
            "MAPE%":    f"{m_mape:.4f}",
            "R":        f"{r_vals.mean():.4f}",
            "MSE":      f"{mse_vals.mean():.2f}",
            "MAE":      f"{mae_vals.mean():.4f}",
            "dir_acc":  f"{da_vals.mean():.3f}",
            "vs_paper": win_loss(m_rmse, m_mape, T.PAPER_RMSE, T.PAPER_MAPE),
            "vs_persist": win_loss(m_rmse, m_mape,
                                   persist_m["persist_RMSE"],
                                   persist_m["persist_MAPE"]),
            "n_seeds":  int(len(sub)),
        })

    # Ensemble rows
    if ensemble_df is not None:
        for _, er in ensemble_df.iterrows():
            records.append({
                "model":    er["variant"],
                "RMSE":     f"{er['RMSE']:.4f}",
                "RMSE_std": "—",
                "MAPE%":    f"{er['MAPE']:.4f}",
                "R":        f"{er['R']:.4f}",
                "MSE":      f"{er['MSE']:.2f}",
                "MAE":      f"{er['MAE']:.4f}",
                "dir_acc":  f"{er['dir_acc']:.3f}",
                "vs_paper": win_loss(er["RMSE"], er["MAPE"], T.PAPER_RMSE, T.PAPER_MAPE),
                "vs_persist": win_loss(er["RMSE"], er["MAPE"],
                                       persist_m["persist_RMSE"],
                                       persist_m["persist_MAPE"]),
                "n_seeds":  int(er["n_seeds"]),
            })

    return pd.DataFrame(records)


def write_results_md(table: pd.DataFrame, stats_rows: list, path="RESULTS.md"):
    """Write the short RESULTS.md table."""
    cols_order = ["model", "RMSE", "RMSE_std", "MAPE%", "R",
                  "MSE", "MAE", "dir_acc", "vs_paper", "vs_persist", "n_seeds"]
    tbl = table.reindex(columns=[c for c in cols_order if c in table.columns])
    lines = [
        "# Results",
        "",
        "Ticker: ^GSPC · 2015-01-01..2025-12-31 · "
        "Regression (next-day Close via log-return) · Target: RMSE in price units",
        "",
        "## Main table (mean ± std over seeds, test set)",
        "",
        tbl.to_markdown(index=False),
        "",
        "WIN = beats reference on both RMSE and MAPE.  ",
        "Persistence: log-return = 0 (tomorrow = today).  ",
        "Paper: Bhandari et al. 2022 — ^GSPC 2006-2021.",
        "",
    ]
    if stats_rows:
        lines += [
            "## Paired tests (fuzzy variants vs Fuzzy_MoE baseline, Holm-corrected)",
            "",
            "| comparison | mean_diff_RMSE | p_t | p_w | Holm_p_t |",
            "|---|---|---|---|---|",
        ]
        for r in stats_rows:
            lines.append(
                f"| {r['comparison']} | {r['mean_diff']:.4f} | "
                f"{r['p_t']:.4f} | {r['p_w']:.4f} | {r['holm_p_t']:.4f} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {path}")
