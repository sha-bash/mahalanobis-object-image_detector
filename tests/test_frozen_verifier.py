import numpy as np
import pytest

from moid.frozen_verifier import FrozenTextVerifier


class Captions:
    def describe(self, image):
        return {'caption': str(image)}


class Embeddings:
    def embed(self, texts):
        return np.array([{'reference': [1., 0.], 'other': [0., 1.]}[t] for t in texts])


def verifier(metric):
    return FrozenTextVerifier('reference', parameters=dict(mean=[0., 0.], components=[[1., 0.], [0., 1.]],
                              covariance=[[4., 1.], [1., 2.]]), metric=metric, threshold=1.,
                              captioner=Captions(), embedder=Embeddings())


def test_frozen_full_covariance_uses_off_diagonal_terms():
    check = verifier('mahalanobis').verify('other')
    # For delta=(-1,1), delta^T [[4,1],[1,2]]^-1 delta = 8/7.
    assert check['distance'] == pytest.approx(np.sqrt(8/7))
    assert not check['accepted']
    assert verifier('mahalanobis').verify('reference')['distance'] == 0


def test_frozen_cosine_accepts_inclusive_boundary_without_refitting():
    model = verifier('cosine')
    before = model.components.copy()
    check = model.verify('other')
    assert check['distance'] == pytest.approx(1.)
    assert check['accepted']
    np.testing.assert_array_equal(model.components, before)
