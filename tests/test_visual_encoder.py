from types import SimpleNamespace

import numpy as np
import torch

from moid.adapters.visual_encoder import VisualEncoder, _to_numpy_vector


def test_to_numpy_vector_from_pooling_output():
    pooled = torch.tensor([[0.25, 0.75]])
    features = SimpleNamespace(pooler_output=pooled, last_hidden_state=None)
    vector = _to_numpy_vector(features)
    np.testing.assert_allclose(vector, [0.25, 0.75])


def test_encode_handles_pooling_output():
    encoder = VisualEncoder.__new__(VisualEncoder)
    encoder.device = "cpu"
    encoder.processor = lambda images, return_tensors: {"pixel_values": torch.zeros(1, 3, 4, 4)}
    encoder.model = SimpleNamespace(
        get_image_features=lambda **_kwargs: SimpleNamespace(
            pooler_output=torch.tensor([[1.0, 2.0, 3.0]])
        )
    )
    vector = encoder.encode(None)
    np.testing.assert_allclose(vector, [1.0, 2.0, 3.0])
