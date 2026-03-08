"""Pure-numpy evaluation metrics for orin direction predictions."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    y_true: list[int], y_pred: list[int], n_classes: int = 3
) -> np.ndarray:
    """3x3 confusion matrix. Rows=true, cols=predicted."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def direction_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Per-direction precision, recall, F1.

    Keys: 'down', 'flat', 'up', each with 'precision', 'recall', 'f1'.
    """
    labels = {0: "down", 1: "flat", 2: "up"}
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    result: dict = {}
    for idx, name in labels.items():
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return result


def sector_breakdown(
    records: list[dict], predictions: list[int], actuals: list[int]
) -> dict:
    """Accuracy by sector/source. Returns dict mapping source to accuracy."""
    source_correct: dict[str, list[bool]] = {}
    for rec, pred, true in zip(records, predictions, actuals):
        source = rec.get("source", rec.get("sector", "unknown"))
        source_correct.setdefault(source, []).append(pred == true)
    return {
        src: round(float(np.mean(vals)), 4) for src, vals in source_correct.items()
    }


def calibration_curve(
    confidences: list[float], corrects: list[bool], n_bins: int = 10
) -> dict:
    """Bin confidences, compute fraction correct per bin.

    Returns dict with keys: bins, accuracy, counts, ece.
    ECE = expected calibration error = sum(|acc_bin - conf_bin| * weight_bin).
    """
    conf_arr = np.asarray(confidences, dtype=float)
    corr_arr = np.asarray(corrects, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    accuracies = np.zeros(n_bins)
    avg_confs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            accuracies[i] = float(corr_arr[mask].mean())
            avg_confs[i] = float(conf_arr[mask].mean())

    total = len(confidences) if len(confidences) > 0 else 1
    ece = float(np.sum(np.abs(accuracies - avg_confs) * counts / total))

    return {
        "bins": bin_centers.tolist(),
        "accuracy": accuracies.tolist(),
        "counts": counts.tolist(),
        "ece": round(ece, 4),
    }


def confidence_intervals(
    values: list[float], confidence: float = 0.95, n_bootstrap: int = 1000
) -> dict:
    """Bootstrap confidence intervals.

    Returns dict with keys: mean, lower, upper, std.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}

    rng = np.random.default_rng(42)
    boot_means = np.array(
        [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_bootstrap)]
    )

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))

    return {
        "mean": round(float(arr.mean()), 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "std": round(float(arr.std()), 4),
    }
