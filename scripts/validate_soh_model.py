import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def build_windows(df: pd.DataFrame, window: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list, y_list, y_prev = [], [], []
    for i in range(len(df) - window):
        xw = df[["current", "temp"]].iloc[i : i + window].values.astype(np.float32)
        yt = float(df["soh"].iloc[i + window])
        yp = float(df["soh"].iloc[i + window - 1])  # persistence baseline
        x_list.append(xw)
        y_list.append(yt)
        y_prev.append(yp)
    return np.array(x_list), np.array(y_list), np.array(y_prev)


def split_time_series(
    x: np.ndarray, y: np.ndarray, y_prev: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15
):
    n = len(x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Not enough samples for train/val/test split.")

    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)

    return (
        (x[idx_train], y[idx_train], y_prev[idx_train]),
        (x[idx_val], y[idx_val], y_prev[idx_val]),
        (x[idx_test], y[idx_test], y_prev[idx_test]),
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    smape = float(np.mean((2.0 * np.abs(err)) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape_pct": mape, "smape_pct": smape}


def fit_linear_calibration(y_true_val: np.ndarray, y_pred_val: np.ndarray) -> Tuple[float, float]:
    # Fit y_true ~= a * y_pred + b
    a, b = np.polyfit(y_pred_val, y_true_val, deg=1)
    return float(a), float(b)


def compute_drift_bins(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 5) -> pd.DataFrame:
    n = len(y_true)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    rows = []
    for i in range(bins):
        s, e = edges[i], edges[i + 1]
        yt = y_true[s:e]
        yp = y_pred[s:e]
        m = metrics(yt, yp)
        rows.append(
            {
                "bin": i + 1,
                "start_idx": int(s),
                "end_idx": int(e - 1),
                "samples": int(max(0, e - s)),
                "mae": m["mae"],
                "rmse": m["rmse"],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["mae_drift_from_bin1"] = df["mae"] - float(df["mae"].iloc[0])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/battery_history.csv")
    parser.add_argument("--model", default="models/fatigue_model.h5")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(data_path)
    required = {"current", "temp", "soh"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input data must contain columns: {sorted(required)}")

    x, y, y_prev = build_windows(df, window=args.window)
    (x_tr, y_tr, yp_tr), (x_val, y_val, yp_val), (x_te, y_te, yp_te) = split_time_series(
        x, y, y_prev, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    model = tf.keras.models.load_model(str(model_path), compile=False)

    yhat_val = model.predict(x_val, verbose=0).reshape(-1)
    yhat_te = model.predict(x_te, verbose=0).reshape(-1)
    yhat_val = np.clip(yhat_val, 0.0, 1.0)
    yhat_te = np.clip(yhat_te, 0.0, 1.0)

    # Calibration on validation split, applied to test split.
    a, b = fit_linear_calibration(y_val, yhat_val)
    yhat_te_cal = np.clip(a * yhat_te + b, 0.0, 1.0)

    # Baseline: persistence (previous timestep SOH).
    yhat_te_base = np.clip(yp_te, 0.0, 1.0)

    m_raw = metrics(y_te, yhat_te)
    m_cal = metrics(y_te, yhat_te_cal)
    m_base = metrics(y_te, yhat_te_base)

    # Domain validity limits from training split.
    tr_current_min = float(np.min(x_tr[:, :, 0]))
    tr_current_max = float(np.max(x_tr[:, :, 0]))
    tr_temp_min = float(np.min(x_tr[:, :, 1]))
    tr_temp_max = float(np.max(x_tr[:, :, 1]))

    # OOD on test windows if any sample in window exceeds training range.
    ood_mask = (
        (np.min(x_te[:, :, 0], axis=1) < tr_current_min)
        | (np.max(x_te[:, :, 0], axis=1) > tr_current_max)
        | (np.min(x_te[:, :, 1], axis=1) < tr_temp_min)
        | (np.max(x_te[:, :, 1], axis=1) > tr_temp_max)
    )
    id_mask = ~ood_mask

    def safe_group_metrics(mask: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        if np.sum(mask) == 0:
            return {"mae": np.nan, "rmse": np.nan, "mape_pct": np.nan, "smape_pct": np.nan}
        return metrics(y_te[mask], pred[mask])

    m_ood = safe_group_metrics(ood_mask, yhat_te_cal)
    m_id = safe_group_metrics(id_mask, yhat_te_cal)

    drift_df = compute_drift_bins(y_te, yhat_te_cal, bins=5)

    run_id = datetime.utcnow().strftime("soh_validation_%Y%m%d_%H%M%S")
    out_dir = Path("data/eval") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detailed predictions
    pred_df = pd.DataFrame(
        {
            "idx": np.arange(len(y_te)),
            "y_true": y_te,
            "y_pred_raw": yhat_te,
            "y_pred_calibrated": yhat_te_cal,
            "y_pred_persistence": yhat_te_base,
            "is_ood": ood_mask.astype(int),
        }
    )
    pred_path = out_dir / "soh_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Metrics table
    metrics_rows = [
        {"model_variant": "rnn_raw", **m_raw},
        {"model_variant": "rnn_calibrated", **m_cal},
        {"model_variant": "persistence_baseline", **m_base},
        {"model_variant": "rnn_calibrated_id_only", **m_id},
        {"model_variant": "rnn_calibrated_ood_only", **m_ood},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / "soh_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    drift_path = out_dir / "soh_error_drift.csv"
    drift_df.to_csv(drift_path, index=False)

    summary = {
        "run_id": run_id,
        "data_path": str(data_path),
        "model_path": str(model_path),
        "window": args.window,
        "split": {"train": len(x_tr), "val": len(x_val), "test": len(x_te)},
        "calibration": {"a": a, "b": b},
        "train_domain": {
            "current_min": tr_current_min,
            "current_max": tr_current_max,
            "temp_min": tr_temp_min,
            "temp_max": tr_temp_max,
        },
        "ood_test_ratio": float(np.mean(ood_mask.astype(float))),
    }
    summary_path = out_dir / "soh_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = out_dir / "soh_validation_report.md"
    lines = [
        "# SOH Validation Report",
        "",
        "## Split",
        "",
        f"- Train windows: {len(x_tr)}",
        f"- Validation windows: {len(x_val)}",
        f"- Test windows: {len(x_te)}",
        "",
        "## Training Domain (Validity Limits)",
        "",
        f"- Current range: [{tr_current_min:.4f}, {tr_current_max:.4f}] A",
        f"- Temperature range: [{tr_temp_min:.4f}, {tr_temp_max:.4f}] C",
        "",
        "## Calibration",
        "",
        f"- Linear calibration: `y_cal = {a:.6f} * y_raw + {b:.6f}`",
        "",
        "## Test Metrics",
        "",
    ]
    try:
        lines.append(metrics_df.to_markdown(index=False))
    except Exception:
        lines.extend(["```text", metrics_df.to_string(index=False), "```"])

    lines.extend(
        [
            "",
            "## Drift Across Test Horizon",
            "",
        ]
    )
    try:
        lines.append(drift_df.to_markdown(index=False))
    except Exception:
        lines.extend(["```text", drift_df.to_string(index=False), "```"])

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Prefer calibrated model if it improves MAE/RMSE vs raw.",
            "- Compare against persistence baseline to justify model utility.",
            "- High OOD ratio indicates weak validity outside training domain.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] metrics: {metrics_path}")
    print(f"[done] drift: {drift_path}")
    print(f"[done] predictions: {pred_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()

