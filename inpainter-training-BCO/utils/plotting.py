import csv
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None


def _to_float(value):
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    return None


def _sanitize_metric_name(metric_name):
    safe = metric_name.replace("/", "_")
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", safe)
    return safe.strip("_") or "metric"


def _rolling_mean(values, window):
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)

    arr = np.asarray(values, dtype=float)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    smooth = (csum[window:] - csum[:-window]) / float(window)
    pad_left = np.full(window - 1, smooth[0])
    return np.concatenate([pad_left, smooth])


def _build_series(metric_history):
    metric_series = {}

    for row in metric_history:
        step = row.get("step")
        if step is None:
            continue

        for metric_name, metric_value in row.items():
            if metric_name == "step":
                continue

            value = _to_float(metric_value)
            if value is None:
                continue

            series = metric_series.setdefault(metric_name, {"steps": [], "values": []})
            series["steps"].append(int(step))
            series["values"].append(value)

    return metric_series


def _write_history_csv(metric_history, out_dir):
    if not metric_history:
        return None

    keys = {"step"}
    for row in metric_history:
        keys.update(row.keys())
    ordered_keys = ["step"] + sorted(k for k in keys if k != "step")

    csv_path = os.path.join(out_dir, "metrics_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        for row in metric_history:
            writer.writerow(row)
    return csv_path


def _plot_group(metric_series, group_name, metric_names, out_dir, smoothing_window):
    available = [name for name in metric_names if name in metric_series]
    if not available:
        return None

    plt.figure(figsize=(12, 6), dpi=160)
    for name in available:
        steps = np.asarray(metric_series[name]["steps"])
        values = np.asarray(metric_series[name]["values"], dtype=float)
        smooth = _rolling_mean(values, smoothing_window)
        plt.plot(steps, values, alpha=0.25, linewidth=1.0)
        plt.plot(steps, smooth, linewidth=2.0, label=name)

    plt.xlabel("Global step")
    plt.ylabel("Metric value")
    plt.title(group_name)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{_sanitize_metric_name(group_name.lower())}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def _plot_single_metric(metric_name, series, out_dir, smoothing_window):
    steps = np.asarray(series["steps"])
    values = np.asarray(series["values"], dtype=float)

    plt.figure(figsize=(10, 5), dpi=160)
    plt.plot(steps, values, alpha=0.3, linewidth=1.0, label="raw")
    smooth = _rolling_mean(values, smoothing_window)
    if len(smooth) == len(values):
        plt.plot(steps, smooth, linewidth=2.0, label=f"smoothed (window={smoothing_window})")

    plt.xlabel("Global step")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs step")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"metric_{_sanitize_metric_name(metric_name)}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_training_metrics(metric_history, out_dir, smoothing_window=5):
    """Create end-of-training metric plots and return generated file paths."""
    os.makedirs(out_dir, exist_ok=True)

    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")

    metric_series = _build_series(metric_history)
    if not metric_series:
        return []

    generated_files = []

    csv_path = _write_history_csv(metric_history, out_dir)
    if csv_path:
        generated_files.append(csv_path)

    grouped_plots = [
        (
            "Training Loss and Stability",
            ["train/loss", "train/mse_gap_avg", "train/grad_norm"],
        ),
        (
            "Preference Reward Signals",
            ["train/reward_gap", "train/log_ratio_pos", "train/log_ratio_neg"],
        ),
        (
            "Optimization Schedule",
            ["train/lr", "train/epoch"],
        ),
        (
            "Label Distribution",
            ["train/label_pos_count", "train/label_neg_count"],
        ),
    ]

    for group_name, metric_names in grouped_plots:
        out_path = _plot_group(metric_series, group_name, metric_names, out_dir, smoothing_window)
        if out_path:
            generated_files.append(out_path)

    for metric_name, series in sorted(metric_series.items()):
        out_path = _plot_single_metric(metric_name, series, out_dir, smoothing_window)
        generated_files.append(out_path)

    return generated_files