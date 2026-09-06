from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from mcd.embedding.sbert import SBERT
from mcd.modeling.covariance import CovarianceMode, estimate_covariance, invert_covariance
from mcd.modeling.drift import detect_drift
from mcd.modeling.thresholds import MaxPlusMarginThresholdStrategy, ThresholdStrategy
from mcd.preprocessing import normalize_ticket_input, preprocess_text

logger = logging.getLogger(__name__)

Prediction = Tuple[str, float, float, bool]


class MahalanobisDriftDetector:
    def __init__(
        self,
        embedder: Any = None,
        threshold_quantile: float = 0.99,
        min_cluster_size: int = 1,
        threshold_strategy: ThresholdStrategy | None = None,
        regularization: float = 0.01,
        covariance_mode: CovarianceMode = "diagonal",
        normalize_tickets: bool = False,
        embedder_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.embedder = embedder or SBERT(model_name=embedder_model_name)
        self.embedder_model_name = embedder_model_name
        self.label_to_index: Dict[str, int] = {}
        self.index_to_label: Dict[int, str] = {}
        self.cluster_means: List[NDArray[np.float64]] = []
        self.cluster_covs: List[NDArray[np.float64]] = []
        self.thresholds: List[float] = []
        self.regularization = float(regularization)
        self.threshold_quantile = threshold_quantile
        self.min_cluster_size = min_cluster_size
        self.covariance_mode: CovarianceMode = covariance_mode
        self.normalize_tickets = normalize_tickets
        self.threshold_strategy: ThresholdStrategy = threshold_strategy or MaxPlusMarginThresholdStrategy()

    def _clean_text(self, text: str) -> str:
        if self.normalize_tickets:
            text = normalize_ticket_input(text)
        return preprocess_text(text)

    def _cluster_distance(self, embedding: NDArray[Any], mean: NDArray[Any], cov: NDArray[Any]) -> float:
        diff = embedding - mean
        inv_cov = invert_covariance(cov, self.regularization)
        quad = float(diff.T @ inv_cov @ diff)
        return float(np.sqrt(max(quad, 0.0)))

    def _nearest_cluster(self, embedding: NDArray[Any]) -> Tuple[int, float]:
        if not self.cluster_means:
            raise RuntimeError("Detector has no fitted clusters. Call fit() first.")
        min_dist = float("inf")
        predicted_cluster = -1
        for i, (mean, cov) in enumerate(zip(self.cluster_means, self.cluster_covs)):
            dist = self._cluster_distance(embedding, mean, cov)
            if dist < min_dist:
                min_dist = dist
                predicted_cluster = i
        return predicted_cluster, min_dist

    def fit(self, texts: List[str], labels: List[str]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("fit() requires at least one sample")

        embeddings = np.asarray(self.embedder.embed(texts), dtype=np.float64)
        self.cluster_means = []
        self.cluster_covs = []
        self.thresholds = []
        built_labels: List[str] = []

        label_arr = np.array([str(x) for x in labels])
        unique_labels = sorted(set(label_arr.tolist()))
        for cluster_label in unique_labels:
            mask = label_arr == cluster_label
            cluster_embeddings = embeddings[mask]
            if cluster_embeddings.shape[0] < self.min_cluster_size:
                logger.warning(
                    "Skipping cluster %s with size %s < %s",
                    cluster_label,
                    cluster_embeddings.shape[0],
                    self.min_cluster_size,
                )
                continue

            mean = np.mean(cluster_embeddings, axis=0)
            cov = estimate_covariance(
                cluster_embeddings,
                self.regularization,
                mode=self.covariance_mode,
            )
            distances = [
                self._cluster_distance(emb, mean, cov) for emb in cluster_embeddings
            ]
            feature_dim = int(cluster_embeddings.shape[1])
            threshold = self.threshold_strategy.compute(distances, feature_dim=feature_dim)

            built_labels.append(str(cluster_label))
            self.cluster_means.append(mean)
            self.cluster_covs.append(cov)
            self.thresholds.append(threshold)

        if not self.cluster_means:
            raise ValueError(
                "No clusters fitted: every class had fewer than min_cluster_size samples"
            )
        self.label_to_index = {label: idx for idx, label in enumerate(built_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        logger.info("Fitted model with %s clusters", len(self.cluster_means))

    def set_threshold(self, value: float, cluster_index: int = 0) -> None:
        if not self.thresholds:
            raise RuntimeError("Detector has no fitted clusters. Call fit() first.")
        if not 0 <= cluster_index < len(self.thresholds):
            raise IndexError(f"cluster_index {cluster_index} out of range")
        if value < 0:
            raise ValueError(f"threshold value must be non-negative, got {value}")
        self.thresholds[cluster_index] = float(value)

    def calibrate_from_negatives(self, texts: List[str], *, quantile: float = 0.05) -> float:
        if not texts:
            raise ValueError("calibrate_from_negatives requires at least one negative caption")
        preds = self.predict_batch(texts)
        distances = [float(item[1]) for item in preds]
        from mcd.modeling.thresholds import QuantileThresholdStrategy

        value = QuantileThresholdStrategy(quantile=quantile).compute(distances, feature_dim=1)
        self.set_threshold(value)
        return value

    def predict(self, text: str) -> Prediction:
        embedding = np.asarray(self.embedder.embed([self._clean_text(text)]), dtype=np.float64)[0]
        return self._predict_embedding(embedding)

    def predict_batch(self, texts: List[str]) -> List[Prediction]:
        cleaned = [self._clean_text(t) for t in texts]
        embeddings = np.asarray(self.embedder.embed(cleaned), dtype=np.float64)
        return self.predict_embeddings(embeddings)

    def predict_embeddings(self, embeddings: NDArray[Any] | Sequence[NDArray[Any]]) -> List[Prediction]:
        arr = np.asarray(embeddings, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return [self._predict_embedding(row) for row in arr]

    def _predict_embedding(self, embedding: NDArray[Any]) -> Prediction:
        predicted_cluster, min_dist = self._nearest_cluster(embedding)
        threshold = self.thresholds[predicted_cluster]
        is_drift = detect_drift(min_dist, threshold)
        predicted_label = self.index_to_label[predicted_cluster]
        return predicted_label, min_dist, threshold, is_drift

    def save(self, path: str) -> None:
        from mcd.persistence.artifacts import save_artifact, save_label_mapping

        data: Dict[str, Any] = {
            "label_to_index": self.label_to_index,
            "cluster_means": self.cluster_means,
            "cluster_covs": self.cluster_covs,
            "thresholds": self.thresholds,
            "regularization": self.regularization,
            "threshold_quantile": self.threshold_quantile,
            "min_cluster_size": self.min_cluster_size,
            "covariance_mode": self.covariance_mode,
            "normalize_tickets": self.normalize_tickets,
            "embedder_model_name": self.embedder_model_name,
        }
        save_artifact(data, path)
        mapping_path = path.replace(".joblib", "_mapping.json")
        save_label_mapping(self.label_to_index, mapping_path)

    @classmethod
    def load(cls, path: str, embedder: Any = None) -> MahalanobisDriftDetector:
        from mcd.persistence.artifacts import load_artifact, load_label_mapping

        data = load_artifact(path)
        mapping_path = path.replace(".joblib", "_mapping.json")
        label_to_index = load_label_mapping(mapping_path)
        model_name = data.get("embedder_model_name", "all-MiniLM-L6-v2")
        instance = cls(
            embedder=embedder,
            threshold_quantile=data.get("threshold_quantile", 0.99),
            min_cluster_size=data.get("min_cluster_size", 1),
            regularization=float(data.get("regularization", 0.01)),
            covariance_mode=data.get("covariance_mode", "diagonal"),
            normalize_tickets=bool(data.get("normalize_tickets", False)),
            embedder_model_name=model_name,
        )
        instance.label_to_index = label_to_index
        instance.index_to_label = {int(v): k for k, v in label_to_index.items()}
        instance.cluster_means = data["cluster_means"]
        instance.cluster_covs = data["cluster_covs"]
        instance.thresholds = data["thresholds"]
        return instance
