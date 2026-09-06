from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
