"""Load labeled texts from CSV (object descriptions or legacy tickets)."""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from mcd.preprocessing import preprocess_text

logger = logging.getLogger(__name__)


def load_labeled_csv(
    path: str,
    label_column: str = "label",
    text_columns: Sequence[str] | None = None,
) -> Tuple[List[str], List[str], Dict[str, int], Dict[int, str]]:
    """Load texts and labels.

    Accepts:
    - ``description`` + label
    - ``body`` (+ optional ``subject``) + label
    - explicit ``text_columns`` joined with a blank line
    """
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    if text_columns is None:
        if "description" in df.columns:
            text_columns = ["description"]
        elif "subject" in df.columns and "body" in df.columns:
            text_columns = ["subject", "body"]
        elif "body" in df.columns:
            text_columns = ["body"]
        elif "text" in df.columns:
            text_columns = ["text"]
        else:
            raise ValueError(
                "Could not infer text columns. Provide description, body, text, "
                "or subject+body, or pass text_columns."
            )

    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns: {missing}")

    subset = list(text_columns) + [label_column]
    df = df.dropna(subset=subset)
    if df.empty:
        raise ValueError("CSV has no usable rows after dropping missing values")

    texts: List[str] = []
    for _, row in df.iterrows():
        parts = [str(row[c]).strip() for c in text_columns if str(row[c]).strip()]
        text = "\n\n".join(parts)
        texts.append(preprocess_text(text))

    labels = [str(v) for v in df[label_column].tolist()]
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    logger.info("Loaded %s samples with %s unique labels", len(texts), len(unique_labels))
    return texts, labels, label_to_index, index_to_label


def load_labeled_tickets_csv(
    path: str,
    label_column: str,
) -> Tuple[List[str], List[str], Dict[str, int], Dict[int, str]]:
    """Backward-compatible alias: tickets with subject/body if present."""
    return load_labeled_csv(path, label_column=label_column)
