"""Data loading for financial text environments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
