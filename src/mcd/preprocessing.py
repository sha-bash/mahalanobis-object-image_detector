"""Text preprocessing utilities."""

import re
from typing import List


def normalize_ticket_input(text: str) -> str:
    """Strip email-style Subject/Body headers (legacy ticket format)."""
    text = text.strip()
    if not text:
        return text
    subject_m = re.search(
        r"(?is)^\s*Subject:\s*(.+?)(?=^\s*Body:|\Z)",
        text,
        flags=re.MULTILINE,
    )
    body_m = re.search(r"(?is)Body:\s*(.+)$", text)
    if subject_m and body_m:
        subj = subject_m.group(1).strip()
        bod = body_m.group(1).strip()
        if subj or bod:
            return f"{subj}\n\n{bod}".strip()
    return text


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_texts(texts: List[str]) -> List[str]:
    return [preprocess_text(t) for t in texts]
