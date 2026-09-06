import json
import logging
from typing import Any, Dict

import joblib

logger = logging.getLogger(__name__)


def save_artifact(data: Any, path: str) -> None:
    joblib.dump(data, path)
    logger.info("Saved artifact to %s", path)


def load_artifact(path: str) -> Any:
    data = joblib.load(path)
    logger.info("Loaded artifact from %s", path)
    return data


def save_label_mapping(mapping: Dict[str, int], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved label mapping to %s", path)


def load_label_mapping(path: str) -> Dict[str, int]:
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    logger.info("Loaded label mapping from %s", path)
    return mapping
