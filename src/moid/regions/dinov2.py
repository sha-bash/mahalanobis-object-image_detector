from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from moid.regions.base import BBox

logger = logging.getLogger(__name__)


class Dinov2Proposer:
    """Box proposals via DINOv2 self-detection (unsupervised, no labels needed).
    
    Uses patch-wise self-similarity in DINOv2 feature maps to find object regions.
    Objects appear as clusters of similar patches that are dissimilar from background.
    
    Requires: pip install moid[visual]  (transformers + torch)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        min_area_ratio: float = 0.002,
        max_boxes: int = 20,
        similarity_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.min_area_ratio = min_area_ratio
        self.max_boxes = max_boxes
        self.similarity_threshold = similarity_threshold
        self.device = device
        self._model = None
        self._processor = None

    def propose(self, image: Image.Image) -> list[BBox]:
        """Return candidate boxes using DINOv2 self-detection."""
        if self._model is None:
            self._load_model()
        
        patches, patch_size = self._extract_patches(image)
        if patches is None or len(patches) == 0:
            return []
        
        boxes = self._find_object_regions(patches, image.width, image.height, patch_size)
        return self._filter(image, boxes)

    def _load_model(self) -> None:
        try:
            from transformers import AutoModel, AutoImageProcessor
            logger.info("Loading DINOv2 model for proposal: %s on %s", self.model_name, self.device)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        except Exception:
            raise RuntimeError(
                f"Failed to load DINOv2 model '{self.model_name}'. "
                "Install with: pip install moid[visual]"
            )

    def _extract_patches(
        self, image: Image.Image
    ) -> Tuple[NDArray[np.float32], int] | None:
        """Extract DINOv2 feature patches from image."""
        try:
            import torch
            pil = image.convert("RGB")
            inputs = self._processor(images=pil, return_tensors="pt").to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # last_hidden_state: [1, num_patches, hidden_dim]
                hidden = outputs.last_hidden_state.squeeze(0)  # [num_patches, hidden_dim]
            
            # hidden[0] is [CLS] token, rest are patch tokens
            patch_tokens = hidden[1:]  # [num_patches, hidden_dim]
            
            # Determine grid size from patch shape
            # DINOv2 uses fixed patch size (14 for small)
            patch_size = 14
            num_patches = patch_tokens.shape[0]
            # Infer grid: for 224x224 image with 14px patches -> 16x16 grid (+ CLS)
            sqrt_n = int(round(num_patches ** 0.5))
            if sqrt_n * sqrt_n != num_patches:
                logger.warning("Non-square patch grid: %d patches", num_patches)
                return patch_tokens.cpu().numpy().astype(np.float32), patch_size
            
            patches = patch_tokens.cpu().numpy().astype(np.float32).reshape(sqrt_n, sqrt_n, -1)
            return patches, patch_size
            
        except Exception:
            logger.exception("Failed to extract DINOv2 patches")
            return None

    def _find_object_regions(
        self,
        patches: NDArray[np.float32],
        img_width: int,
        img_height: int,
        patch_size: int,
    ) -> list[BBox]:
        """Find object regions using self-similarity clustering."""
        # patches: [H, W, D]
        h, w, d = patches.shape
        
        # Compute self-similarity between patches
        # Flatten patches
        patch_vecs = patches.reshape(-1, d)  # [H*W, D]
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(patch_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        patch_vecs_norm = patch_vecs / norms
        
        # Compute similarity matrix (only upper triangle for efficiency)
        # For large images, sample subset to avoid O(n^2) memory
        n_patches = h * w
        max_samples = 256
        if n_patches > max_samples:
            indices = np.linspace(0, n_patches - 1, max_samples, dtype=int)
            sim_matrix = np.dot(patch_vecs_norm[indices], patch_vecs_norm.T)
        else:
            sim_matrix = np.dot(patch_vecs_norm, patch_vecs_norm.T)
            indices = np.arange(n_patches)
        
        # For each patch, find its most similar neighbor (excluding self)
        sim_matrix_clean = sim_matrix.copy()
        np.fill_diagonal(sim_matrix_clean, -1)
        max_sim_idx = np.argmax(sim_matrix_clean, axis=1)
        max_sim_vals = np.max(sim_matrix_clean, axis=1)
        
        # Object centers tend to have high self-similarity with neighbors
        # Background patches tend to have low similarity (uniform)
        # Use similarity as a score
        scores = max_sim_vals.reshape(h, w) if n_patches > max_samples else max_sim_vals.reshape(h, w)
        
        # Find local maxima in similarity map (object centers)
        local_maxima = self._find_local_maxima(scores)
        
        # Cluster nearby maxima into regions
        regions = self._cluster_regions(local_maxima, scores, h, w, patch_size)
        
        # Convert to BBox
        boxes = []
        for (y1, x1, y2, x2) in regions:
            bx1 = int(x1 * patch_size)
            by1 = int(y1 * patch_size)
            bx2 = int((x2 + 1) * patch_size)
            by2 = int((y2 + 1) * patch_size)
            boxes.append(BBox(bx1, by1, bx2, by2))
        
        return boxes

    def _find_local_maxima(self, scores: NDArray[np.float32]) -> list[Tuple[int, int]]:
        """Find local maxima in similarity map."""
        maxima = []
        h, w = scores.shape
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if scores[r, c] > scores[r-1, c-1] and scores[r, c] > scores[r-1, c] and scores[r, c] > scores[r-1, c+1] and \
                   scores[r, c] > scores[r, c-1] and scores[r, c] > scores[r, c+1] and \
                   scores[r, c] > scores[r+1, c-1] and scores[r, c] > scores[r+1, c] and scores[r, c] > scores[r+1, c+1]:
                    maxima.append((r, c))
        return maxima

    def _cluster_regions(
        self,
        maxima: list[Tuple[int, int]],
        scores: NDArray[np.float32],
        h: int,
        w: int,
        patch_size: int,
    ) -> list[Tuple[int, int, int, int]]:
        """Cluster nearby maxima into regions."""
        if not maxima:
            return []
        
        # Sort by score (highest first)
        maxima_sorted = sorted(maxima, key=lambda m: scores[m[0], m[1]], reverse=True)
        
        # Non-maximum suppression: cluster maxima that are close together
        clusters: list[list[Tuple[int, int]]] = []
        cluster_threshold = max(3, patch_size // 4)  # ~1 patch distance
        
        for m in maxima_sorted:
            r, c = m
            assigned = False
            for cluster in clusters:
                # Check if this maximum is close to any point in the cluster
                for cr, cc in cluster:
                    if abs(r - cr) <= cluster_threshold and abs(c - cc) <= cluster_threshold:
                        cluster.append(m)
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                clusters.append([m])
        
        # Convert clusters to bounding boxes
        regions = []
        for cluster in clusters:
            rs = [r for r, c in cluster]
            cs = [c for r, c in cluster]
            y1, y2 = min(rs), max(rs)
            x1, x2 = min(cs), max(cs)
            regions.append((y1, x1, y2, x2))
        
        return regions

    def _filter(self, image: Image.Image, boxes: list[BBox]) -> list[BBox]:
        """Filter boxes by minimum area and deduplicate."""
        area = max(image.width * image.height, 1)
        kept: list[BBox] = []
        seen: set[tuple[int, int, int, int]] = set()
        for box in boxes:
            clipped = box.clip(image.width, image.height)
            if clipped.area() < self.min_area_ratio * area:
                continue
            key = clipped.as_tuple()
            if key in seen:
                continue
            seen.add(key)
            kept.append(clipped)
            if len(kept) >= self.max_boxes:
                break
        return kept
