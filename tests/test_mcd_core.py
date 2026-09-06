from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcd.io import load_labeled_csv
from mcd.modeling.classifier import MahalanobisDriftDetector
from mcd.modeling.covariance import estimate_covariance
from mcd.modeling.mahalanobis import identity_covariance, mahalanobis_between
from mcd.modeling.thresholds import FixedThresholdStrategy, MaxPlusMarginThresholdStrategy
from tests.embedder_utils import LexicalEmbedder


def test_covariance_single_sample_is_finite():
    x = np.ones((1, 4), dtype=np.float64)
    cov = estimate_covariance(x, reg=0.01, mode="diagonal")
    assert cov.shape == (4, 4)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, 0.01 * np.eye(4))


def test_fit_one_sample():
    det = MahalanobisDriftDetector(
        embedder=LexicalEmbedder(),
        min_cluster_size=1,
        regularization=0.01,
        threshold_strategy=MaxPlusMarginThresholdStrategy(margin=2.0),
    )
    det.fit(["object: red truck; color: red"], ["target"])
    assert len(det.cluster_means) == 1
    assert np.all(np.isfinite(det.cluster_covs[0]))
    label, dist, thr, is_drift = det.predict("object: red truck; color: red")
    assert label == "target"
    assert dist <= thr
    assert is_drift is False


def test_predict_batch_and_embeddings():
    det = MahalanobisDriftDetector(embedder=LexicalEmbedder(), min_cluster_size=1)
    det.fit(["object: red truck"], ["target"])
    batch = det.predict_batch(["object: red truck", "object: green terrain"])
    assert len(batch) == 2
    assert batch[0][1] < batch[1][1]
    embs = LexicalEmbedder().embed(["object: red truck"])
    from_emb = det.predict_embeddings(embs)
    assert from_emb[0][0] == "target"


def test_load_description_csv(tmp_path: Path):
    csv_path = tmp_path / "desc.csv"
    pd.DataFrame(
        {"description": ["red truck cab"], "label": ["target"]}
    ).to_csv(csv_path, index=False)
    texts, labels, *_ = load_labeled_csv(str(csv_path))
    assert texts == ["red truck cab"]
    assert labels == ["target"]


def test_mahalanobis_between_identity():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    d = mahalanobis_between(a, b, identity_covariance(2), reg=0.0)
    assert d == pytest.approx(5.0)


def test_calibrate_from_negatives():
    det = MahalanobisDriftDetector(
        embedder=LexicalEmbedder(),
        min_cluster_size=1,
        regularization=0.01,
        threshold_strategy=MaxPlusMarginThresholdStrategy(margin=50.0),
    )
    det.fit(["object: red truck; color: red"], ["target"])
    wide = det.thresholds[0]
    value = det.calibrate_from_negatives(
        ["object: green terrain tree", "object: green tree"],
        quantile=0.5,
    )
    assert value < wide
    assert det.thresholds[0] == value


def test_fixed_threshold():
    det = MahalanobisDriftDetector(
        embedder=LexicalEmbedder(),
        threshold_strategy=FixedThresholdStrategy(value=0.01),
        min_cluster_size=1,
    )
    det.fit(["object: red truck"], ["target"])
    _label, dist, thr, is_drift = det.predict("object: green terrain tree")
    assert thr == 0.01
    assert is_drift is True
    assert dist > thr
