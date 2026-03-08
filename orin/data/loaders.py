"""Data loading for financial text environments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file.

    Expected format per line:
        {"text": "...", "ticker": "...", "date": "...",
         "outcome": {"direction": "up", "magnitude": 0.03, "timeframe": "1d"}}
    """
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_sample_data(env_type: str) -> list[dict[str, Any]]:
    """Load bundled sample data for an environment type.

    Args:
        env_type: one of "earnings", "news", "filing", "macro"

    Returns:
        List of sample records.
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "sample"
    path = data_dir / f"{env_type}.jsonl"
    if not path.exists():
        from orin.data.sources import write_sample_data

        write_sample_data(data_dir)
    return load_jsonl(path)


def load_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert records to a DataFrame with flattened outcome columns."""
    df = pd.json_normalize(records)
    return df


def load_huggingface(dataset_name: str, split: str = "train") -> list[dict[str, Any]]:
    """Load data from a HuggingFace dataset.

    Requires the `datasets` package: pip install orin[all]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("HuggingFace datasets not installed. Run: pip install orin[all]")
    ds = load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]


def _resolve_dotted_key(record: dict, dotted_key: str) -> Any:
    """Resolve a dotted key path like ``'outcome.direction'`` on a nested dict."""
    obj: Any = record
    for part in dotted_key.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def split_data(
    records: list[dict],
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    stratify_by: str = "outcome.direction",
) -> tuple[list, list, list]:
    """Split records into train / val / test sets with optional stratification.

    Args:
        records: List of data records.
        train: Fraction for training.
        val: Fraction for validation.
        test: Fraction for testing.
        seed: Random seed for reproducibility.
        stratify_by: Dotted key path used for stratified splitting.  If the
            key is missing from every record, a non-stratified shuffle split
            is performed instead.

    Returns:
        ``(train_records, val_records, test_records)``
    """
    if not records:
        return [], [], []

    rng = np.random.default_rng(seed)

    # Normalise fractions
    total = train + val + test
    train_frac = train / total
    val_frac = val / total
    # test_frac is implicitly the remainder

    # Group by stratification key
    buckets: dict[Any, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        key = _resolve_dotted_key(rec, stratify_by)
        buckets[key].append(idx)

    # If every record resolved to None, fall back to a single bucket
    if list(buckets.keys()) == [None]:
        buckets = {None: list(range(len(records)))}

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _key, indices in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        arr = np.array(indices)
        rng.shuffle(arr)
        n = len(arr)
        n_train = max(1, int(round(n * train_frac))) if n >= 3 else n
        n_val = max(1, int(round(n * val_frac))) if n - n_train >= 2 else 0
        # Ensure we don't exceed total
        if n_train + n_val > n:
            n_val = n - n_train

        train_idx.extend(arr[:n_train].tolist())
        val_idx.extend(arr[n_train : n_train + n_val].tolist())
        test_idx.extend(arr[n_train + n_val :].tolist())

    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )
