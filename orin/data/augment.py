"""Data augmentation utilities for synthetic financial text records.

Provides synonym replacement, sentence reordering, and number perturbation
to increase training data diversity while preserving outcome labels.
"""

from __future__ import annotations

import copy
import random
import re

# -- Synonym replacement pairs --
# Each group contains interchangeable words/phrases.

_SYNONYM_GROUPS: list[list[str]] = [
    ["revenue", "sales"],
    ["grew", "increased", "rose"],
    ["declined", "fell", "dropped"],
    ["beat", "exceeded", "surpassed"],
    ["missed", "fell short of", "underperformed"],
    ["strong", "robust", "solid"],
    ["weak", "soft", "disappointing"],
    ["guidance", "outlook", "forecast"],
    ["raised", "lifted", "increased"],
    ["lowered", "cut", "reduced"],
]

# Build a lookup: word -> list of synonyms (including itself)
_SYNONYM_MAP: dict[str, list[str]] = {}
for group in _SYNONYM_GROUPS:
    for word in group:
        _SYNONYM_MAP[word] = group

# Pattern to find numbers (integers and decimals) in text
_NUMBER_RE = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)")


def _replace_synonyms(text: str, rng: random.Random) -> str:
    """Replace words with random synonyms from the synonym groups."""
    words = text.split(" ")
    result = []
    for word in words:
        # Strip trailing punctuation for matching
        stripped = word.rstrip(".,;:!?%)")
        suffix = word[len(stripped):]
        lower = stripped.lower()
        if lower in _SYNONYM_MAP:
            candidates = [s for s in _SYNONYM_MAP[lower] if s != lower]
            if candidates:
                replacement = rng.choice(candidates)
                # Preserve original capitalization
                if stripped[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement + suffix)
                continue
        result.append(word)
    return " ".join(result)


def _reorder_sentences(text: str, rng: random.Random) -> str:
    """Shuffle middle sentences while keeping first and last in place."""
    parts = text.split(". ")
    if len(parts) <= 2:
        return text
    first = parts[0]
    last = parts[-1]
    middle = parts[1:-1]
    rng.shuffle(middle)
    return ". ".join([first] + middle + [last])


def _perturb_numbers(text: str, rng: random.Random) -> str:
    """Perturb numeric values by +/-5-15%."""

    def _perturb_match(match: re.Match) -> str:
        val = float(match.group(1))
        if val == 0:
            return match.group(0)
        factor = 1.0 + rng.uniform(-0.15, 0.15)
        # Clamp to at least +/-5% change
        if abs(factor - 1.0) < 0.05:
            factor = 1.05 if factor >= 1.0 else 0.95
        new_val = val * factor
        # Preserve format: integer if original was integer, else same decimal places
        if "." in match.group(1):
            decimals = len(match.group(1).split(".")[1])
            return f"{new_val:.{decimals}f}"
        return str(int(round(new_val)))

    return _NUMBER_RE.sub(_perturb_match, text)


def augment_records(
    records: list[dict],
    synonym_replace: bool = True,
    sentence_reorder: bool = True,
    number_perturb: bool = True,
    n_augments: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Augment financial text records with text transformations.

    Each input record can produce ``n_augments`` augmented copies.  The
    returned list contains all original records followed by augmented copies.
    Outcome labels are preserved unchanged.

    Args:
        records: List of orin-format records with at least a "text" key.
        synonym_replace: Whether to apply synonym replacement.
        sentence_reorder: Whether to shuffle middle sentences.
        number_perturb: Whether to perturb numeric values.
        n_augments: Number of augmented copies per record.
        seed: Random seed for reproducibility.

    Returns:
        Original records plus augmented copies.
    """
    rng = random.Random(seed)
    augmented: list[dict] = []

    for record in records:
        for _ in range(n_augments):
            new_record = copy.deepcopy(record)
            text = new_record["text"]

            if synonym_replace:
                text = _replace_synonyms(text, rng)
            if sentence_reorder:
                text = _reorder_sentences(text, rng)
            if number_perturb:
                text = _perturb_numbers(text, rng)

            new_record["text"] = text
            augmented.append(new_record)

    return list(records) + augmented
